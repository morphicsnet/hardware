use crate::generic::{self, Pass, PassContext, PassError, PassOutcome, PassResult};
use crate::{lower_to_kernels, memory_layout_and_quant, kernel_fusion_and_scheduling, nir};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Diag {
    pub code: String,
    pub message: String,
    pub path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub ok: bool,
    pub errors: Vec<Diag>,
}

pub struct Validation;

impl Pass<nir::Graph> for Validation {
    fn name(&self) -> &'static str { "validation" }

    fn run(&self, module: &mut nir::Graph, ctx: &mut PassContext) -> PassResult {
        let kp: Option<lower_to_kernels::KernelPlan> =
            module.attributes.get("kernel_plan").and_then(|v| serde_json::from_value(v.clone()).ok());
        let lq: Option<memory_layout_and_quant::LayoutAndQuantPlan> =
            module.attributes.get("layout_and_quant_plan").and_then(|v| serde_json::from_value(v.clone()).ok());
        let sp: Option<kernel_fusion_and_scheduling::SchedulePlan> =
            module.attributes.get("schedule_plan").and_then(|v| serde_json::from_value(v.clone()).ok());

        let mut errors: Vec<Diag> = Vec::new();

        if let Some(ref plan) = kp {
            check_kernel_plan(plan, module, &mut errors);
        }
        if let Some(ref plan) = lq {
            check_layout_and_quant(plan, kp.as_ref(), &mut errors);
        }
        if let Some(ref plan) = sp {
            check_schedule(plan, kp.as_ref(), &mut errors);
        }
        check_cross(kp.as_ref(), lq.as_ref(), sp.as_ref(), &mut errors);

        sort_diags(&mut errors);

        // Dump report if requested (guarded by central policy)
        let ok = errors.is_empty();
        let report = ValidationReport { ok, errors: errors.clone() };
        let cfg_map = crate::config_map_from_value(ctx.config.as_ref());
        if crate::should_dump("validation", &cfg_map) {
            let _ = maybe_dump(&report, ctx);
        }
        if crate::metrics_enabled(&cfg_map) {
            tracing::info!(
                ok = report.ok,
                errors_count = report.errors.len(),
                pass = "validation",
                "validation metrics"
            );
        }

        if ok {
            return Ok(PassOutcome::Unchanged);
        }

        // Summarize first few diagnostics deterministically
        let n = report.errors.len();
        let mut parts: Vec<String> = Vec::new();
        for d in report.errors.iter().take(5) {
            parts.push(format!("{} @ {}: {}", d.code, d.path, d.message));
        }
        let summary = format!(
            "validation failed with {} error(s); first: {}",
            n,
            parts.join(" | ")
        );
        Err(PassError::Unsupported(summary))
    }
}

pub fn mk(_: Option<&Value>) -> Box<dyn Pass<nir::Graph>> { Box::new(Validation) }

/// Register this pass as "validation"
pub fn register(reg: &mut generic::Registry<nir::Graph>) {
    reg.register("validation", mk);
}

// ----- helpers -----

fn sort_diags(errors: &mut Vec<Diag>) {
    errors.sort_by(|a, b| {
        match a.code.cmp(&b.code) {
            std::cmp::Ordering::Equal => match a.path.cmp(&b.path) {
                std::cmp::Ordering::Equal => a.message.cmp(&b.message),
                o => o,
            },
            o => o,
        }
    });
}

fn token_of(label: &str) -> Option<String> {
    label.split_once(':').map(|(t, _)| t.to_string())
}

fn normalize_output_label(label: &str) -> String {
    if let Some((node, rest)) = label.split_once(':') {
        if rest.starts_with("out") {
            label.to_string()
        } else {
            format!("{node}:out0")
        }
    } else {
        format!("{label}:out0")
    }
}

fn check_kernel_plan(kp: &lower_to_kernels::KernelPlan, g: &nir::Graph, errors: &mut Vec<Diag>) {
    // KP001: IDs unique, ascending, contiguous 0..N-1
    let n = kp.kernels.len() as u32;
    let mut seen: HashSet<u32> = HashSet::new();
    for (i, k) in kp.kernels.iter().enumerate() {
        if !seen.insert(k.id) {
            errors.push(Diag {
                code: "KP001".into(),
                message: format!("duplicate kernel id {}", k.id),
                path: format!("kernel_plan.kernels[{}].id", i),
            });
        }
    }
    for expect in 0..n {
        if !seen.contains(&expect) {
            errors.push(Diag {
                code: "KP001".into(),
                message: format!("missing kernel id {} in 0..{}", expect, n.saturating_sub(1)),
                path: "kernel_plan.kernels".into(),
            });
        }
    }
    for (i, k) in kp.kernels.iter().enumerate() {
        if k.id != i as u32 {
            errors.push(Diag {
                code: "KP001".into(),
                message: format!("kernel id {} not in ascending order at index {}", k.id, i),
                path: format!("kernel_plan.kernels[{}].id", i),
            });
        }
    }

    // KP002: op names non-empty
    for (i, k) in kp.kernels.iter().enumerate() {
        if k.op.trim().is_empty() {
            errors.push(Diag {
                code: "KP002".into(),
                message: "op name is empty".into(),
                path: format!("kernel_plan.kernels[{}].op", i),
            });
        }
    }

    // KP003: For each consumer input label "X:in{i}" there exists some producer output with "X:out{j}"
    // Only enforce when inputs use the "...:in{i}" form.
    let mut produced_tokens: HashSet<String> = HashSet::new();
    for k in &kp.kernels {
        for o in &k.outputs {
            if let Some(tok) = token_of(o) {
                produced_tokens.insert(tok);
            }
        }
    }
    // Allow graph sources: tokens backed by populations that are not produced by any kernel.
    let pop_names: HashSet<String> = g.populations.iter().map(|p| p.name.clone()).collect();

    for (ki, k) in kp.kernels.iter().enumerate() {
        for (ii, inp) in k.inputs.iter().enumerate() {
            if let Some((tok, rest)) = inp.split_once(':') {
                if rest.starts_with("in") {
                    let t = tok.to_string();
                    let ok = produced_tokens.contains(&t);
                    if !ok {
                        // If it's a known population name and not produced by any kernel, treat as graph input (OK).
                        if !pop_names.contains(&t) {
                            errors.push(Diag {
                                code: "KP003".into(),
                                message: format!("dangling input reference '{}': no matching producer output", inp),
                                path: format!("kernel_plan.kernels[{}].inputs[{}]", ki, ii),
                            });
                        }
                    }
                }
            }
        }
    }
}

fn check_layout_and_quant(
    lq: &memory_layout_and_quant::LayoutAndQuantPlan,
    kp: Option<&lower_to_kernels::KernelPlan>,
    errors: &mut Vec<Diag>,
) {
    // LQ001: buffer indexes unique and contiguous 0..B-1; buffer.name non-empty.
    let b = lq.buffers.len() as u32;
    let mut idx_seen: HashSet<u32> = HashSet::new();
    for (bi, buf) in lq.buffers.iter().enumerate() {
        if buf.name.trim().is_empty() {
            errors.push(Diag {
                code: "LQ001".into(),
                message: "buffer name is empty".into(),
                path: format!("layout_and_quant_plan.buffers[{}].name", bi),
            });
        }
        if !idx_seen.insert(buf.index) {
            errors.push(Diag {
                code: "LQ001".into(),
                message: format!("duplicate buffer index {}", buf.index),
                path: format!("layout_and_quant_plan.buffers[{}].index", bi),
            });
        }
    }
    for expect in 0..b {
        if !idx_seen.contains(&expect) {
            errors.push(Diag {
                code: "LQ001".into(),
                message: format!("missing buffer index {} in 0..{}", expect, b.saturating_sub(1)),
                path: "layout_and_quant_plan.buffers".into(),
            });
        }
    }

    // LQ002: For each buffer.name, there exists a kernel output label in KernelPlan that matches it exactly
    // under normalization used by layout (i.e., normalize_output_label on kernel outputs).
    if let Some(kp) = kp {
        let mut produced: HashSet<String> = HashSet::new();
        for k in &kp.kernels {
            for o in &k.outputs {
                produced.insert(normalize_output_label(o));
            }
        }
        for (bi, buf) in lq.buffers.iter().enumerate() {
            if !produced.contains(&buf.name) {
                errors.push(Diag {
                    code: "LQ002".into(),
                    message: format!("buffer '{}' has no matching kernel output label", buf.name),
                    path: format!("layout_and_quant_plan.buffers[{}].name", bi),
                });
            }
        }
    }

    // LQ003: For each kernel_quant entry, inputs.len == kernel.inputs.len and outputs.len == kernel.outputs.len
    if let Some(kp) = kp {
        let mut arities: HashMap<u32, (usize, usize)> = HashMap::new();
        for k in &kp.kernels {
            arities.insert(k.id, (k.inputs.len(), k.outputs.len()));
        }
        for (qi, q) in lq.quant.iter().enumerate() {
            if let Some((ai, ao)) = arities.get(&q.kernel_id).copied() {
                if q.inputs.len() != ai {
                    errors.push(Diag {
                        code: "LQ003".into(),
                        message: format!("quant inputs len {} does not match kernel {} inputs arity {}", q.inputs.len(), q.kernel_id, ai),
                        path: format!("layout_and_quant_plan.quant[{}].inputs", qi),
                    });
                }
                if q.outputs.len() != ao {
                    errors.push(Diag {
                        code: "LQ003".into(),
                        message: format!("quant outputs len {} does not match kernel {} outputs arity {}", q.outputs.len(), q.kernel_id, ao),
                        path: format!("layout_and_quant_plan.quant[{}].outputs", qi),
                    });
                }
            }
        }
    }
}

fn check_schedule(
    sp: &kernel_fusion_and_scheduling::SchedulePlan,
    kp: Option<&lower_to_kernels::KernelPlan>,
    errors: &mut Vec<Diag>,
) {
    // SC001: group IDs unique and contiguous 0..G-1
    let g = sp.groups.len() as u32;
    let mut gid_seen: HashSet<u32> = HashSet::new();
    for (gi, gr) in sp.groups.iter().enumerate() {
        if !gid_seen.insert(gr.id) {
            errors.push(Diag {
                code: "SC001".into(),
                message: format!("duplicate group id {}", gr.id),
                path: format!("schedule_plan.groups[{}].id", gi),
            });
        }
    }
    for expect in 0..g {
        if !gid_seen.contains(&expect) {
            errors.push(Diag {
                code: "SC001".into(),
                message: format!("missing group id {} in 0..{}", expect, g.saturating_sub(1)),
                path: "schedule_plan.groups".into(),
            });
        }
    }

    // SC002: each kernel appears exactly once across all groups (no duplicates)
    let mut seen: HashSet<u32> = HashSet::new();
    for (gi, gr) in sp.groups.iter().enumerate() {
        for (ki, kid) in gr.kernels.iter().enumerate() {
            if !seen.insert(*kid) {
                errors.push(Diag {
                    code: "SC002".into(),
                    message: format!("kernel {} appears multiple times across groups", kid),
                    path: format!("schedule_plan.groups[{}].kernels[{}]", gi, ki),
                });
            }
        }
    }

    // SC003: within each group, order must respect producer→consumer dependencies implied by KernelPlan labels
    if let Some(kp) = kp {
        let mut producers_by_token: HashMap<String, HashSet<u32>> = HashMap::new();
        for k in &kp.kernels {
            for o in &k.outputs {
                if let Some(tok) = token_of(o) {
                    producers_by_token.entry(tok).or_default().insert(k.id);
                }
            }
        }
        // For consumer dependencies: input tokens
        let mut inputs_by_kernel: HashMap<u32, HashSet<String>> = HashMap::new();
        for k in &kp.kernels {
            for i in &k.inputs {
                if let Some(tok) = token_of(i) {
                    inputs_by_kernel.entry(k.id).or_default().insert(tok);
                }
            }
        }

        for (gi, gr) in sp.groups.iter().enumerate() {
            let mut pos: HashMap<u32, usize> = HashMap::new();
            for (idx, kid) in gr.kernels.iter().enumerate() {
                pos.insert(*kid, idx);
            }
            for (idx_c, &cid) in gr.kernels.iter().enumerate() {
                if let Some(tokens) = inputs_by_kernel.get(&cid) {
                    for tok in tokens {
                        if let Some(prods) = producers_by_token.get(tok) {
                            for pid in prods {
                                if let (Some(&ip), Some(&ic)) = (pos.get(pid), pos.get(&cid)) {
                                    if ic < ip {
                                        errors.push(Diag {
                                            code: "SC003".into(),
                                            message: format!("consumer kernel {} appears before its producer {} for token '{}'", cid, pid, tok),
                                            path: format!("schedule_plan.groups[{}].kernels", gi),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                // Also ensure no self-dependency problems (ignore)
                let _ = idx_c;
            }
        }
    }

    // SC004: group.outputs must match outputs of the last kernel in the group (normalized)
    if let Some(kp) = kp {
        let by_id: HashMap<u32, &lower_to_kernels::Kernel> =
            kp.kernels.iter().map(|k| (k.id, k)).collect();
        for (gi, gr) in sp.groups.iter().enumerate() {
            if let Some(&last_id) = gr.kernels.last() {
                if let Some(k) = by_id.get(&last_id).copied() {
                    let mut outs: Vec<String> = Vec::new();
                    let mut seen: HashSet<String> = HashSet::new();
                    for o in &k.outputs {
                        let n = normalize_output_label(o);
                        if seen.insert(n.clone()) {
                            outs.push(n);
                        }
                    }
                    if outs != gr.outputs {
                        errors.push(Diag {
                            code: "SC004".into(),
                            message: format!("group outputs {:?} do not match last kernel {} normalized outputs {:?}", gr.outputs, last_id, outs),
                            path: format!("schedule_plan.groups[{}].outputs", gi),
                        });
                    }
                }
            }
        }
    }
}

fn check_cross(
    kp: Option<&lower_to_kernels::KernelPlan>,
    lq: Option<&memory_layout_and_quant::LayoutAndQuantPlan>,
    sp: Option<&kernel_fusion_and_scheduling::SchedulePlan>,
    errors: &mut Vec<Diag>,
) {
    // XA001: schedule references kernel IDs not present in KernelPlan
    if let (Some(kp), Some(sp)) = (kp, sp) {
        let kp_ids: HashSet<u32> = kp.kernels.iter().map(|k| k.id).collect();
        for (gi, gr) in sp.groups.iter().enumerate() {
            for (ki, kid) in gr.kernels.iter().enumerate() {
                if !kp_ids.contains(kid) {
                    errors.push(Diag {
                        code: "XA001".into(),
                        message: format!("schedule references unknown kernel id {}", kid),
                        path: format!("schedule_plan.groups[{}].kernels[{}]", gi, ki),
                    });
                }
            }
        }
    }

    // XA002: quant provided for kernels not present in KernelPlan
    if let (Some(kp), Some(lq)) = (kp, lq) {
        let kp_ids: HashSet<u32> = kp.kernels.iter().map(|k| k.id).collect();
        for (qi, q) in lq.quant.iter().enumerate() {
            if !kp_ids.contains(&q.kernel_id) {
                errors.push(Diag {
                    code: "XA002".into(),
                    message: format!("quant provided for unknown kernel id {}", q.kernel_id),
                    path: format!("layout_and_quant_plan.quant[{}].kernel_id", qi),
                });
            }
        }
    }
}

fn maybe_dump(report: &ValidationReport, ctx: &mut PassContext) -> Result<bool, PassError> {
    let dir_opt: Option<String> = match ctx.config.as_ref() {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(_)) => ctx
            .config
            .as_ref()
            .and_then(|v| v.get("dump_dir"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    };
    if let Some(dir) = dir_opt {
        let run_id = ctx.run_id.clone().unwrap_or_else(|| "run".to_string());
        let mut path = PathBuf::from(dir);
        if !path.exists() {
            fs::create_dir_all(&path)?;
        }
        path.push(format!("validation_{}.json", run_id));
        let s = serde_json::to_string_pretty(report).map_err(|e| PassError::Internal(e.to_string()))?;
        fs::write(&path, s)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::{PassContext, PassOutcome, PassSpec, PipelineDescriptor, Registry};
    use serde_json::json;
    use std::fs;

    fn make_chain_add_relu() -> nir::Graph {
        let mut g = nir::Graph::new("df");
        g.dialect = Some(nir::Dialect::Dataflow);
        g.populations.push(nir::Population { name: "x".into(), size: 1, model: "Input".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "Add".into(), size: 1, model: "Add".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "y".into(), size: 1, model: "Tensor".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "ReLU".into(), size: 1, model: "ReLU".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "z".into(), size: 1, model: "Tensor".into(), params: json!({}) });

        g.connections.push(nir::Connection { pre: "x".into(), post: "Add".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "Add".into(), post: "y".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "y".into(), post: "ReLU".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "ReLU".into(), post: "z".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g
    }

    fn build_e2e_then(mut g: nir::Graph) -> nir::Graph {
        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: None },
        ];
        let desc = PipelineDescriptor { passes };
        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        crate::memory_layout_and_quant::register(&mut reg);
        crate::kernel_fusion_and_scheduling::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        let out = pipeline.run(&mut g, &mut ctx).unwrap();
        assert!(matches!(out, PassOutcome::Changed));
        g
    }

    fn run_validation_only(mut g: nir::Graph) -> Result<PassOutcome, crate::generic::PassError> {
        let passes = vec![
            PassSpec { name: "validation".into(), config: None },
        ];
        let desc = PipelineDescriptor { passes };
        let mut reg = Registry::<nir::Graph>::new();
        crate::validation::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();
        let mut ctx = PassContext::default();
        pipeline.run(&mut g, &mut ctx)
    }

    #[test]
    fn ok_smoke() {
        let mut g = make_chain_add_relu();
        g = build_e2e_then(g);

        // Also test dump ok=true
        let dir = std::env::temp_dir().join("nc_passes_validation_ok");
        let _ = fs::create_dir_all(&dir);
        let passes = vec![
            PassSpec { name: "validation".into(), config: Some(json!({ "dump_dir": dir.to_string_lossy() })) },
        ];
        let desc = PipelineDescriptor { passes };
        let mut reg = Registry::<nir::Graph>::new();
        crate::validation::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        ctx.run_id = Some("ok-run-1".into());
        let out = pipeline.run(&mut g, &mut ctx).unwrap();
        assert!(matches!(out, PassOutcome::Unchanged));

        let path = dir.join("validation_ok-run-1.json");
        let s = fs::read_to_string(&path).expect("dump exists");
        let rep: ValidationReport = serde_json::from_str(&s).unwrap();
        assert!(rep.ok);
        assert!(rep.errors.is_empty());
    }

    #[test]
    fn dangling_input_error() {
        let mut g = build_e2e_then(make_chain_add_relu());

        // Mutate kernel_plan: set first kernel input to a dangling "W:in0"
        let mut kp: lower_to_kernels::KernelPlan = serde_json::from_value(g.attributes.get("kernel_plan").cloned().unwrap()).unwrap();
        if !kp.kernels.is_empty() && !kp.kernels[0].inputs.is_empty() {
            kp.kernels[0].inputs[0] = "W:in0".to_string();
        }
        g.attributes.insert("kernel_plan".into(), serde_json::to_value(&kp).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("KP003"));
        assert!(msg.contains("kernel_plan.kernels[0].inputs[0]"));
    }

    #[test]
    fn duplicate_buffer_index_error() {
        let mut g = build_e2e_then(make_chain_add_relu());
        let mut lq: memory_layout_and_quant::LayoutAndQuantPlan = serde_json::from_value(g.attributes.get("layout_and_quant_plan").cloned().unwrap()).unwrap();
        if lq.buffers.len() >= 2 {
            lq.buffers[1].index = lq.buffers[0].index;
        }
        g.attributes.insert("layout_and_quant_plan".into(), serde_json::to_value(&lq).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("LQ001"));
    }

    #[test]
    fn non_contiguous_buffer_index_error() {
        let mut g = build_e2e_then(make_chain_add_relu());
        let mut lq: memory_layout_and_quant::LayoutAndQuantPlan = serde_json::from_value(g.attributes.get("layout_and_quant_plan").cloned().unwrap()).unwrap();
        if lq.buffers.len() >= 2 {
            lq.buffers[1].index = 2; // skip 1
        }
        g.attributes.insert("layout_and_quant_plan".into(), serde_json::to_value(&lq).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("LQ001"));
    }

    #[test]
    fn quant_mismatch_lengths_error() {
        let mut g = build_e2e_then(make_chain_add_relu());
        let mut lq: memory_layout_and_quant::LayoutAndQuantPlan = serde_json::from_value(g.attributes.get("layout_and_quant_plan").cloned().unwrap()).unwrap();
        // Make outputs len = 2 for first kernel quant
        if !lq.quant.is_empty() {
            let q0 = &mut lq.quant[0];
            q0.outputs.push(memory_layout_and_quant::QParams { dtype: "f32".into(), scale: 1.0, zero_point: 0 });
        }
        g.attributes.insert("layout_and_quant_plan".into(), serde_json::to_value(&lq).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("LQ003"));
    }

    #[test]
    fn duplicate_kernel_in_group_error() {
        let mut g = build_e2e_then(make_chain_add_relu());
        let mut sp: kernel_fusion_and_scheduling::SchedulePlan = serde_json::from_value(g.attributes.get("schedule_plan").cloned().unwrap()).unwrap();
        if let Some(gr) = sp.groups.get_mut(0) {
            if !gr.kernels.is_empty() {
                let first = gr.kernels[0];
                gr.kernels.insert(1, first);
            }
        }
        g.attributes.insert("schedule_plan".into(), serde_json::to_value(&sp).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("SC002"));
    }

    #[test]
    fn order_violation_in_group_error() {
        let mut g = build_e2e_then(make_chain_add_relu());
        let mut sp: kernel_fusion_and_scheduling::SchedulePlan = serde_json::from_value(g.attributes.get("schedule_plan").cloned().unwrap()).unwrap();
        if let Some(gr) = sp.groups.get_mut(0) {
            if gr.kernels.len() >= 2 {
                gr.kernels.swap(0, 1);
            }
        }
        g.attributes.insert("schedule_plan".into(), serde_json::to_value(&sp).unwrap());

        let res = run_validation_only(g);
        assert!(res.is_err());
        let Err(PassError::Unsupported(msg)) = res else { panic!("expected Unsupported"); };
        assert!(msg.contains("SC003"));
    }

    #[test]
    fn dump_report_golden() {
        let mut g = build_e2e_then(make_chain_add_relu());
        // Introduce a failure
        let mut sp: kernel_fusion_and_scheduling::SchedulePlan = serde_json::from_value(g.attributes.get("schedule_plan").cloned().unwrap()).unwrap();
        if let Some(gr) = sp.groups.get_mut(0) {
            if gr.kernels.len() >= 2 {
                gr.kernels.swap(0, 1);
            }
        }
        g.attributes.insert("schedule_plan".into(), serde_json::to_value(&sp).unwrap());

        let dir = std::env::temp_dir().join("nc_passes_validation_dump");
        let _ = fs::create_dir_all(&dir);
        let passes = vec![PassSpec { name: "validation".into(), config: Some(json!({ "dump_dir": dir.to_string_lossy() })) }];
        let desc = PipelineDescriptor { passes };
        let mut reg = Registry::<nir::Graph>::new();
        crate::validation::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        ctx.run_id = Some("val-run-1".into());
        let res = pipeline.run(&mut g, &mut ctx);
        assert!(res.is_err());

        let path = dir.join("validation_val-run-1.json");
        let s = fs::read_to_string(&path).expect("dump exists");
        let rep: ValidationReport = serde_json::from_str(&s).unwrap();
        assert!(!rep.ok);
        assert!(!rep.errors.is_empty());
        // Assert at least one expected code present
        let codes: HashSet<String> = rep.errors.iter().map(|d| d.code.clone()).collect();
        assert!(codes.contains("SC003") || codes.contains("SC002") || codes.contains("KP001") || codes.contains("LQ001"));
    }
}
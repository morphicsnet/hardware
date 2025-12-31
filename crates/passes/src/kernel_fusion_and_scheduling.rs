use crate::generic::{self, Pass, PassContext, PassError, PassOutcome, PassResult};
use crate::{lower_to_kernels, memory_layout_and_quant, nir};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FusionGroup {
    pub id: u32,            // contiguous 0..G-1
    pub kernels: Vec<u32>,  // kernel IDs in execution order
    pub ops: Vec<String>,   // ops of kernels in order (debug convenience)
    pub outputs: Vec<String>, // labels of the group's final outputs, deterministic order
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SchedulePlan {
    pub groups: Vec<FusionGroup>, // execution order is vector order
}

pub struct KernelFusionAndScheduling;

impl Pass<nir::Graph> for KernelFusionAndScheduling {
    fn name(&self) -> &'static str { "kernel_fusion_and_scheduling" }

    fn run(&self, module: &mut nir::Graph, ctx: &mut PassContext) -> PassResult {
        // Acquire kernel plan (reconstruct if attribute missing or malformed)
        let kplan: lower_to_kernels::KernelPlan = if let Some(v) = module.attributes.get("kernel_plan") {
            serde_json::from_value(v.clone()).unwrap_or_else(|_| lower_to_kernels::extract_kernel_plan(module))
        } else {
            lower_to_kernels::extract_kernel_plan(module)
        };

        if kplan.kernels.is_empty() {
            return Ok(PassOutcome::Unchanged);
        }

        // Acquire layout+quant (if missing, we conservatively disable fusion)
        let lq: Option<memory_layout_and_quant::LayoutAndQuantPlan> =
            module
                .attributes
                .get("layout_and_quant_plan")
                .and_then(|v| serde_json::from_value(v.clone()).ok());

        // Build schedule with conservative fusion heuristic
        let plan = build_schedule(&kplan, lq.as_ref());

        // Attach attribute
        if let Ok(v) = serde_json::to_value(&plan) {
            module.attributes.insert("schedule_plan".to_string(), v);
        }

        // Optional dump (guarded by central policy)
        let cfg_map = crate::config_map_from_value(ctx.config.as_ref());
        if crate::should_dump("schedule", &cfg_map) {
            let _ = maybe_dump(&plan, ctx);
        }
        if crate::metrics_enabled(&cfg_map) {
            tracing::info!(
                fusion_groups_count = plan.groups.len(),
                pass = "kernel_fusion_and_scheduling",
                "schedule metrics"
            );
        }

        Ok(PassOutcome::Changed)
    }
}

pub fn mk(_: Option<&Value>) -> Box<dyn Pass<nir::Graph>> { Box::new(KernelFusionAndScheduling) }

/// Register this pass into a generic Registry keyed as "kernel_fusion_and_scheduling".
pub fn register(reg: &mut generic::Registry<nir::Graph>) {
    reg.register("kernel_fusion_and_scheduling", mk);
}

fn maybe_dump(plan: &SchedulePlan, ctx: &mut PassContext) -> Result<bool, PassError> {
    // Accept config as string path or object { "dump_dir": "..." }
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
        path.push(format!("schedule_{}.json", run_id));
        let s = serde_json::to_string_pretty(plan).map_err(|e| PassError::Internal(e.to_string()))?;
        fs::write(&path, s)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn token_of(label: &str) -> Option<String> {
    label.split_once(':').map(|(t, _)| t.to_string())
}

fn normalize_output_label(label: &str) -> String {
    // Map "...:in{i}" -> "...:out0" for a stable, consumer-style name; keep "...:out{i}" as-is.
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

fn build_schedule(
    kp: &lower_to_kernels::KernelPlan,
    lq: Option<&memory_layout_and_quant::LayoutAndQuantPlan>,
) -> SchedulePlan {
    // Fusible set S (conservative)
    const FUSIBLE_SET: [&str; 5] = ["Add", "ReLU", "Sigmoid", "Tanh", "Relu6"];
    let fusible: HashSet<&str> = FUSIBLE_SET.iter().copied().collect();
    let is_fusible = |op: &str| -> bool { fusible.contains(op) };

    let is_identity = |q: &memory_layout_and_quant::QParams| -> bool {
        q.dtype == "f32" && (q.scale - 1.0).abs() < f32::EPSILON && q.zero_point == 0
    };

    // Ordered kernel ids (deterministic)
    let mut ordered_ids: Vec<u32> = kp.kernels.iter().map(|k| k.id).collect();
    ordered_ids.sort_unstable();

    // Index by id
    let max_id = ordered_ids.iter().copied().max().unwrap_or(0) as usize;
    let mut by_id: Vec<Option<&lower_to_kernels::Kernel>> = vec![None; max_id + 1];
    for k in &kp.kernels {
        if (k.id as usize) < by_id.len() {
            by_id[k.id as usize] = Some(k);
        }
    }

    // Inputs by token -> Vec<(consumer_kid, consumer_in_idx)>
    let mut inputs_by_token: HashMap<String, Vec<(u32, usize)>> = HashMap::new();
    for k in &kp.kernels {
        for (i, inp) in k.inputs.iter().enumerate() {
            if let Some(tok) = token_of(inp) {
                inputs_by_token.entry(tok).or_default().push((k.id, i));
            }
        }
    }
    for v in inputs_by_token.values_mut() {
        v.sort_by_key(|(kid, idx)| (*kid, *idx));
    }

    // Quant lookup maps (optional)
    let qmap: Option<HashMap<u32, &memory_layout_and_quant::KernelQuant>> =
        lq.map(|plan| plan.quant.iter().map(|kq| (kq.kernel_id, kq)).collect());

    let mut grouped: Vec<bool> = vec![false; max_id + 1];
    let mut groups: Vec<FusionGroup> = Vec::new();

    for kid in ordered_ids {
        let ki = kid as usize;
        if ki >= by_id.len() || by_id[ki].is_none() || grouped[ki] {
            continue;
        }
        let mut kernels: Vec<u32> = Vec::new();
        let mut ops: Vec<String> = Vec::new();

        // seed group with current kernel
        let mut curr_id = kid;
        let mut curr = by_id[curr_id as usize].unwrap();
        kernels.push(curr_id);
        ops.push(curr.op.clone());
        grouped[curr_id as usize] = true;

        // Try to greedily extend downstream
        loop {
            // No fusion if no quant metadata
            if qmap.is_none() {
                break;
            }
            // Both ends must be fusible
            if !is_fusible(&curr.op) {
                break;
            }

            // Find first eligible consumer along any output token, in deterministic order of outputs
            let mut next_choice: Option<(u32, usize, usize)> = None; // (consumer_id, prod_out_idx, cons_in_idx)
            for (out_idx, out_label) in curr.outputs.iter().enumerate() {
                let Some(tok) = token_of(out_label) else { continue };
                let consumers = inputs_by_token.get(&tok).cloned().unwrap_or_default();
                if consumers.len() != 1 {
                    continue;
                }
                let (cid, c_in_idx) = consumers[0];
                let c_ki = cid as usize;
                let Some(cons) = by_id.get(c_ki).and_then(|o| *o) else { continue };
                if grouped.get(c_ki).copied().unwrap_or(false) {
                    continue;
                }
                if !is_fusible(&cons.op) {
                    continue;
                }

                // Quantization identity check on both sides
                let Some(qmap_ref) = &qmap else { continue };
                let Some(kq_p) = qmap_ref.get(&curr_id).copied() else { continue };
                let Some(kq_c) = qmap_ref.get(&cid).copied() else { continue };
                if out_idx >= kq_p.outputs.len() || c_in_idx >= kq_c.inputs.len() {
                    continue;
                }
                if !is_identity(&kq_p.outputs[out_idx]) || !is_identity(&kq_c.inputs[c_in_idx]) {
                    continue;
                }

                next_choice = Some((cid, out_idx, c_in_idx));
                break; // take first deterministic choice
            }

            if let Some((cid, _out_idx, _in_idx)) = next_choice {
                // extend chain
                curr_id = cid;
                curr = by_id[curr_id as usize].unwrap();
                kernels.push(curr_id);
                ops.push(curr.op.clone());
                grouped[curr_id as usize] = true;
                continue;
            } else {
                break;
            }
        }

        // Group outputs: outputs of the last kernel in group, normalized and deduped deterministically
        let last = by_id[kernels.last().copied().unwrap() as usize].unwrap();
        let mut outs: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for o in &last.outputs {
            let n = normalize_output_label(o);
            if seen.insert(n.clone()) {
                outs.push(n);
            }
        }

        let gid = groups.len() as u32;
        groups.push(FusionGroup { id: gid, kernels, ops, outputs: outs });
    }

    SchedulePlan { groups }
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

    fn make_chain_add_matmul() -> nir::Graph {
        let mut g = nir::Graph::new("df");
        g.dialect = Some(nir::Dialect::Dataflow);
        g.populations.push(nir::Population { name: "x".into(), size: 1, model: "Input".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "Add".into(), size: 1, model: "Add".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "y".into(), size: 1, model: "Tensor".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "MatMul".into(), size: 1, model: "MatMul".into(), params: json!({}) });
        g.populations.push(nir::Population { name: "z".into(), size: 1, model: "Tensor".into(), params: json!({}) });

        g.connections.push(nir::Connection { pre: "x".into(), post: "Add".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "Add".into(), post: "y".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "y".into(), post: "MatMul".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g.connections.push(nir::Connection { pre: "MatMul".into(), post: "z".into(), weight: 1.0, delay_ms: 0.0, plasticity: None });
        g
    }

    #[test]
    fn pointwise_chain_fuses() {
        let mut g1 = make_chain_add_relu();

        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: None },
        ];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        crate::memory_layout_and_quant::register(&mut reg);
        super::register(&mut reg);

        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        let out = pipeline.run(&mut g1, &mut ctx).unwrap();
        assert!(matches!(out, PassOutcome::Changed));

        let plan_v = g1.attributes.get("schedule_plan").cloned().expect("schedule_plan");
        let plan: SchedulePlan = serde_json::from_value(plan_v).unwrap();

        assert_eq!(plan.groups.len(), 1);
        let g0 = &plan.groups[0];
        assert_eq!(g0.id, 0);
        assert_eq!(g0.kernels, vec![0, 1]);
        assert_eq!(g0.ops, vec!["Add".to_string(), "ReLU".to_string()]);
        assert_eq!(g0.outputs, vec!["z:out0".to_string()]);

        // Determinism
        let mut g2 = make_chain_add_relu();
        let mut ctx2 = PassContext::default();
        let _ = pipeline.run(&mut g2, &mut ctx2).unwrap();
        let v2 = g2.attributes.get("schedule_plan").cloned().unwrap();
        let plan2: SchedulePlan = serde_json::from_value(v2).unwrap();
        assert_eq!(plan, plan2);
    }

    #[test]
    fn non_fusible_matmul_breaks_chain() {
        let mut g = make_chain_add_matmul();

        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: None },
        ];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        crate::memory_layout_and_quant::register(&mut reg);
        super::register(&mut reg);

        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        let _ = pipeline.run(&mut g, &mut ctx).unwrap();

        let plan_v = g.attributes.get("schedule_plan").cloned().expect("schedule_plan");
        let plan: SchedulePlan = serde_json::from_value(plan_v).unwrap();

        assert_eq!(plan.groups.len(), 2);
        assert_eq!(plan.groups[0].kernels, vec![0]);
        assert_eq!(plan.groups[0].ops, vec!["Add".to_string()]);
        assert_eq!(plan.groups[0].outputs, vec!["y:out0".to_string()]);
        assert_eq!(plan.groups[1].kernels, vec![1]);
        assert_eq!(plan.groups[1].ops, vec!["MatMul".to_string()]);
        assert_eq!(plan.groups[1].outputs, vec!["z:out0".to_string()]);
    }

    #[test]
    fn dump_path_golden() {
        let mut g = make_chain_add_relu();

        let dir = std::env::temp_dir().join("nc_passes_sched_tests");
        let _ = fs::create_dir_all(&dir);
        let sched_cfg = json!({ "dump_dir": dir.to_string_lossy() });

        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: Some(sched_cfg) },
        ];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        crate::memory_layout_and_quant::register(&mut reg);
        super::register(&mut reg);

        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        ctx.run_id = Some("sched-run-1".into());
        let _ = pipeline.run(&mut g, &mut ctx).unwrap();

        let path = dir.join("schedule_sched-run-1.json");
        let s = fs::read_to_string(&path).expect("dump exists");
        let plan_from_file: SchedulePlan = serde_json::from_str(&s).unwrap();

        // Expected golden inline
        let golden = r#"
{
  "groups": [
    { "id": 0, "kernels": [0, 1], "ops": ["Add", "ReLU"], "outputs": ["z:out0"] }
  ]
}"#;
        let golden_plan: SchedulePlan = serde_json::from_str(golden).unwrap();
        assert_eq!(plan_from_file, golden_plan);
    }
}
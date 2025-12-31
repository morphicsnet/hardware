use crate::generic::{self, Pass, PassContext, PassError, PassOutcome, PassResult};
use crate::nir;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub id: u32,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub kernels: Vec<Kernel>,
}

pub struct LowerToKernels;

impl Pass<nir::Graph> for LowerToKernels {
    fn name(&self) -> &'static str { "lower_to_kernels" }

    fn run(&self, module: &mut nir::Graph, ctx: &mut PassContext) -> PassResult {
        let plan = extract_kernel_plan(module);
        if plan.kernels.is_empty() {
            return Ok(PassOutcome::Unchanged);
        }

        // Attach as an annotation for observability by downstream passes/tests.
        if let Ok(v) = serde_json::to_value(&plan) {
            module
                .attributes
                .insert("kernel_plan".to_string(), v);
        }

        // Optional JSON dump if configured (guarded by central policy).
        let cfg_map = crate::config_map_from_value(ctx.config.as_ref());
        if crate::should_dump("lower", &cfg_map) {
            let _ = maybe_dump(&plan, ctx);
        }
        if crate::metrics_enabled(&cfg_map) {
            tracing::info!(kernels_count = plan.kernels.len(), pass = "lower_to_kernels", "lower metrics");
        }

        Ok(PassOutcome::Changed)
    }
}

pub fn mk(_: Option<&Value>) -> Box<dyn Pass<nir::Graph>> { Box::new(LowerToKernels) }

/// Register this pass into a generic Registry keyed as "lower_to_kernels".
pub fn register(reg: &mut generic::Registry<nir::Graph>) {
    reg.register("lower_to_kernels", mk);
}

/// Extract a minimal, deterministic kernel plan from a NIR graph.
/// - Compute nodes are detected conservatively based on population.model being a known op
///   (e.g., "Add", "MatMul", "Conv", "ReLU"). Unknown models are ignored.
/// - Inputs/outputs are stable strings like "src:out0" and "dst:in0", indexed deterministically.
pub fn extract_kernel_plan(g: &nir::Graph) -> KernelPlan {
    // 1) Stable topological order across ALL nodes, then filter to compute nodes.
    let topo = topo_order_all_nodes(g);
    let compute_nodes: Vec<(String, String)> = topo
        .into_iter()
        .filter_map(|name| {
            g.populations
                .iter()
                .find(|p| p.name == name)
                .and_then(|p| detect_op(p).map(|op| (p.name.clone(), op)))
        })
        .collect();

    // 2) Precompute edge indices and stable in/out port numbering.
    let edges: Vec<(usize, &nir::Connection)> = g.connections.iter().enumerate().collect();

    // incoming edge ids per dst, outgoing per src (String keys for simplicity)
    let mut in_eids: HashMap<String, Vec<usize>> = HashMap::new();
    let mut out_eids: HashMap<String, Vec<usize>> = HashMap::new();
    for (eid, c) in &edges {
        in_eids.entry(c.post.clone()).or_default().push(*eid);
        out_eids.entry(c.pre.clone()).or_default().push(*eid);
    }

    // Assign in-slot per destination deterministically: sort by (pre_name, eid)
    let mut in_slot: HashMap<usize, usize> = HashMap::new();
    for (_dst, eids) in &mut in_eids {
        eids.sort_by(|&a, &b| {
            let ca = &g.connections[a];
            let cb = &g.connections[b];
            ca.pre.cmp(&cb.pre).then(a.cmp(&b))
        });
        for (i, eid) in eids.iter().enumerate() {
            in_slot.insert(*eid, i);
        }
    }

    // Assign out-slot per source deterministically: sort by (post_name, eid)
    let mut out_slot: HashMap<usize, usize> = HashMap::new();
    for (_src, eids) in &mut out_eids {
        eids.sort_by(|&a, &b| {
            let ca = &g.connections[a];
            let cb = &g.connections[b];
            ca.post.cmp(&cb.post).then(a.cmp(&b))
        });
        for (i, eid) in eids.iter().enumerate() {
            out_slot.insert(*eid, i);
        }
    }

    // 3) Build kernels in the compute-node topological order.
    let mut kernels: Vec<Kernel> = Vec::new();
    for (kid, (name, op)) in compute_nodes.iter().enumerate() {
        // Inputs: edges into this compute node, labels "src:out{j}", ordered by this node's in-slot.
        let mut ins: Vec<(usize, String)> = g
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.post == *name)
            .map(|(eid, c)| {
                let slot = *out_slot.get(&eid).unwrap_or(&0);
                (in_slot.get(&eid).copied().unwrap_or(0), format!("{}:out{}", c.pre, slot))
            })
            .collect();
        ins.sort_by_key(|(i, _)| *i);
        let inputs: Vec<String> = ins.into_iter().map(|(_, s)| s).collect();

        // Outputs: edges out of this compute node, labels "dst:in{i}", ordered by this node's out-slot.
        let mut outs: Vec<(usize, String)> = g
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.pre == *name)
            .map(|(eid, c)| {
                let slot = *in_slot.get(&eid).unwrap_or(&0);
                (out_slot.get(&eid).copied().unwrap_or(0), format!("{}:in{}", c.post, slot))
            })
            .collect();
        outs.sort_by_key(|(i, _)| *i);
        let outputs: Vec<String> = outs.into_iter().map(|(_, s)| s).collect();

        kernels.push(Kernel {
            id: kid as u32,
            op: op.clone(),
            inputs,
            outputs,
        });
    }

    KernelPlan { kernels }
}

fn detect_op(p: &nir::Population) -> Option<String> {
    // Prefer explicit "op" in params; else use model
    if let Some(op) = p.params.get("op").and_then(|v| v.as_str()) {
        if let Some(norm) = normalize_op(op) {
            return Some(norm.to_string());
        }
    }
    normalize_op(&p.model).map(|s| s.to_string())
}

fn normalize_op(op: &str) -> Option<&'static str> {
    let m = op.to_ascii_lowercase();
    match m.as_str() {
        "add" => Some("Add"),
        "sub" | "subtract" => Some("Sub"),
        "mul" | "multiply" => Some("Mul"),
        "matmul" => Some("MatMul"),
        "conv" | "convolution" => Some("Conv"),
        "relu" => Some("ReLU"),
        "sigmoid" => Some("Sigmoid"),
        "tanh" => Some("Tanh"),
        "batchnorm" | "batch_norm" => Some("BatchNorm"),
        _ => None,
    }
}

fn topo_order_all_nodes(g: &nir::Graph) -> Vec<String> {
    let mut indeg: HashMap<String, usize> = HashMap::new();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for p in &g.populations {
        indeg.entry(p.name.clone()).or_insert(0);
        adj.entry(p.name.clone()).or_insert_with(Vec::new);
    }
    for c in &g.connections {
        indeg.entry(c.post.clone()).and_modify(|v| *v += 1).or_insert(1);
        adj.entry(c.pre.clone()).or_insert_with(Vec::new).push(c.post.clone());
    }
    let mut zeros: Vec<String> = indeg
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(k, _)| k.clone())
        .collect();
    zeros.sort(); // lexicographic for determinism
    let mut q: VecDeque<String> = VecDeque::from(zeros);
    let mut out: Vec<String> = Vec::new();
    let mut indeg_mut = indeg.clone();
    while let Some(n) = q.pop_front() {
        out.push(n.clone());
        if let Some(ns) = adj.get(&n) {
            let mut nexts = ns.clone();
            nexts.sort();
            for m in nexts {
                if let Some(d) = indeg_mut.get_mut(&m) {
                    if *d > 0 {
                        *d -= 1;
                        if *d == 0 {
                            q.push_back(m.clone());
                        }
                    }
                }
            }
        }
    }
    // Fallback: if cycle remains, append remaining nodes in lexicographic order to keep determinism
    if out.len() < g.populations.len() {
        let mut remaining: Vec<String> = g.populations.iter().map(|p| p.name.clone()).collect();
        remaining.sort();
        remaining.retain(|n| !out.contains(n));
        out.extend(remaining);
    }
    out
}

fn maybe_dump(plan: &KernelPlan, ctx: &mut PassContext) -> Result<bool, PassError> {
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
        path.push(format!("kernels_{}.json", run_id));
        let s = serde_json::to_string_pretty(plan)
            .map_err(|e| PassError::Internal(e.to_string()))?;
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

    fn make_test_graph() -> nir::Graph {
        let mut g = nir::Graph::new("df");
        g.dialect = Some(nir::Dialect::Dataflow);
        g.populations.push(nir::Population {
            name: "x".into(),
            size: 1,
            model: "Input".into(),
            params: json!({}),
        });
        g.populations.push(nir::Population {
            name: "Add".into(),
            size: 1,
            model: "Add".into(),
            params: json!({}),
        });
        g.populations.push(nir::Population {
            name: "y".into(),
            size: 1,
            model: "Tensor".into(),
            params: json!({}),
        });
        g.populations.push(nir::Population {
            name: "ReLU".into(),
            size: 1,
            model: "ReLU".into(),
            params: json!({}),
        });
        g.populations.push(nir::Population {
            name: "z".into(),
            size: 1,
            model: "Tensor".into(),
            params: json!({}),
        });
        g.connections.push(nir::Connection {
            pre: "x".into(),
            post: "Add".into(),
            weight: 1.0,
            delay_ms: 0.0,
            plasticity: None,
        });
        g.connections.push(nir::Connection {
            pre: "Add".into(),
            post: "y".into(),
            weight: 1.0,
            delay_ms: 0.0,
            plasticity: None,
        });
        g.connections.push(nir::Connection {
            pre: "y".into(),
            post: "ReLU".into(),
            weight: 1.0,
            delay_ms: 0.0,
            plasticity: None,
        });
        g.connections.push(nir::Connection {
            pre: "ReLU".into(),
            post: "z".into(),
            weight: 1.0,
            delay_ms: 0.0,
            plasticity: None,
        });
        g
    }

    #[test]
    fn kernels_are_extracted_and_deterministic() {
        let mut g = make_test_graph();

        let passes = vec![PassSpec {
            name: "lower_to_kernels".into(),
            config: None,
        }];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        let outcome = pipeline.run(&mut g, &mut ctx).unwrap();
        assert!(matches!(outcome, PassOutcome::Changed));

        let plan_v = g
            .attributes
            .get("kernel_plan")
            .cloned()
            .expect("kernel_plan attribute");
        let plan: KernelPlan = serde_json::from_value(plan_v).unwrap();
        assert_eq!(plan.kernels.len(), 2);
        let ops: Vec<String> = plan.kernels.iter().map(|k| k.op.clone()).collect();
        assert_eq!(ops, vec!["Add".to_string(), "ReLU".to_string()]);
        assert_eq!(plan.kernels[0].inputs.len(), 1);
        assert_eq!(plan.kernels[0].outputs.len(), 1);
        assert_eq!(plan.kernels[1].inputs.len(), 1);
        assert_eq!(plan.kernels[1].outputs.len(), 1);

        // Determinism: recompute directly and compare
        let plan2 = extract_kernel_plan(&g);
        assert_eq!(plan, plan2);
    }

    #[test]
    fn dump_json_matches_golden() {
        let mut g = make_test_graph();

        // Build descriptor with dump_dir
        let dir = std::env::temp_dir().join("nc_passes_ltk_tests");
        let _ = fs::create_dir_all(&dir);
        let cfg = json!({ "dump_dir": dir.to_string_lossy() });
        let passes = vec![PassSpec {
            name: "lower_to_kernels".into(),
            config: Some(cfg),
        }];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        ctx.run_id = Some("test-run-1".into());
        let outcome = pipeline.run(&mut g, &mut ctx).unwrap();
        assert!(matches!(outcome, PassOutcome::Changed));

        let path = dir.join("kernels_test-run-1.json");
        let s = fs::read_to_string(&path).expect("dump exists");
        let plan_from_file: KernelPlan = serde_json::from_str(&s).unwrap();

        // Expected golden inline
        let golden = r#"
{
  "kernels": [
    { "id": 0, "op": "Add", "inputs": ["x:out0"], "outputs": ["y:in0"] },
    { "id": 1, "op": "ReLU", "inputs": ["y:out0"], "outputs": ["z:in0"] }
  ]
}"#;
        let golden_plan: KernelPlan = serde_json::from_str(golden).unwrap();
        assert_eq!(plan_from_file, golden_plan);
    }
}
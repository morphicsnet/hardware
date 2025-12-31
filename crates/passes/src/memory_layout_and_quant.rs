use crate::generic::{self, Pass, PassContext, PassError, PassOutcome, PassResult};
use crate::lower_to_kernels;
use crate::nir;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub name: String,
    pub index: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct QParams {
    pub dtype: String,
    pub scale: f32,
    pub zero_point: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KernelQuant {
    pub kernel_id: u32,
    pub inputs: Vec<QParams>,
    pub outputs: Vec<QParams>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LayoutAndQuantPlan {
    pub buffers: Vec<Buffer>,
    pub quant: Vec<KernelQuant>,
}

/// Minimal pass that assigns stable buffer indices to kernel outputs and identity quant params.
pub struct MemoryLayoutAndQuant;

impl Pass<nir::Graph> for MemoryLayoutAndQuant {
    fn name(&self) -> &'static str { "memory_layout_and_quant" }

    fn run(&self, module: &mut nir::Graph, ctx: &mut PassContext) -> PassResult {
        // 1) Acquire kernel plan from attribute, else conservatively extract (deterministic)
        let kplan: lower_to_kernels::KernelPlan = if let Some(v) = module.attributes.get("kernel_plan") {
            serde_json::from_value(v.clone()).unwrap_or_else(|_| lower_to_kernels::extract_kernel_plan(module))
        } else {
            lower_to_kernels::extract_kernel_plan(module)
        };

        if kplan.kernels.is_empty() {
            return Ok(PassOutcome::Unchanged);
        }

        // 2) Build deterministic layout + quant plan
        let plan = build_plan_from_kernel_plan(&kplan);

        // 3) Attach artifact for downstream consumption
        if let Ok(v) = serde_json::to_value(&plan) {
            module
                .attributes
                .insert("layout_and_quant_plan".to_string(), v);
        }

        // 4) Optional dump (if configured) with central toggle policy
        let cfg_map = crate::config_map_from_value(ctx.config.as_ref());
        if crate::should_dump("layout", &cfg_map) {
            let _ = maybe_dump(&plan, ctx);
        }
        if crate::metrics_enabled(&cfg_map) {
            tracing::info!(
                buffers_count = plan.buffers.len(),
                quant_entries = plan.quant.len(),
                pass = "memory_layout_and_quant",
                "layout metrics"
            );
        }

        Ok(PassOutcome::Changed)
    }
}

fn normalize_output_label(label: &str) -> String {
    // NOTE: lower_to_kernels outputs are "dst:in{i}". We map each produced tensor to "dst:out0"
    // to name the buffer consistently with consumer input labels (and even for sinks).
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

fn build_plan_from_kernel_plan(kp: &lower_to_kernels::KernelPlan) -> LayoutAndQuantPlan {
    // Buffers: first occurrence order over all kernel outputs (stable since kp is deterministic)
    let mut name_to_index: HashMap<String, u32> = HashMap::new();
    let mut buffers: Vec<Buffer> = Vec::new();

    for k in &kp.kernels {
        for out in &k.outputs {
            let buf_name = normalize_output_label(out);
            if !name_to_index.contains_key(&buf_name) {
                let idx = buffers.len() as u32;
                name_to_index.insert(buf_name.clone(), idx);
                buffers.push(Buffer { name: buf_name, index: idx });
            }
        }
    }

    // Identity quantization for all inputs/outputs
    let ident = || QParams {
        dtype: "f32".to_string(),
        scale: 1.0,
        zero_point: 0,
    };

    let mut quant: Vec<KernelQuant> = Vec::with_capacity(kp.kernels.len());
    for k in &kp.kernels {
        quant.push(KernelQuant {
            kernel_id: k.id,
            inputs: (0..k.inputs.len()).map(|_| ident()).collect(),
            outputs: (0..k.outputs.len()).map(|_| ident()).collect(),
        });
    }

    LayoutAndQuantPlan { buffers, quant }
}

fn maybe_dump(plan: &LayoutAndQuantPlan, ctx: &mut PassContext) -> Result<bool, PassError> {
    // Accept config as either a string path or object { "dump_dir": "..." }
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
        path.push(format!("layout_quant_{}.json", run_id));
        let s = serde_json::to_string_pretty(plan).map_err(|e| PassError::Internal(e.to_string()))?;
        fs::write(&path, s)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn mk(_: Option<&Value>) -> Box<dyn Pass<nir::Graph>> { Box::new(MemoryLayoutAndQuant) }

/// Register this pass under name "memory_layout_and_quant"
pub fn register(reg: &mut generic::Registry<nir::Graph>) {
    reg.register("memory_layout_and_quant", mk);
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
    fn plan_contents_and_determinism() {
        let mut g1 = make_test_graph();

        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
        ];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        super::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        let outcome = pipeline.run(&mut g1, &mut ctx).unwrap();
        assert!(matches!(outcome, PassOutcome::Changed));

        // Parse produced plan
        let plan_v = g1
            .attributes
            .get("layout_and_quant_plan")
            .cloned()
            .expect("layout_and_quant_plan attribute");
        let plan: LayoutAndQuantPlan = serde_json::from_value(plan_v).unwrap();

        // Buffers: outputs of Add then ReLU, mapped to ["y:out0", "z:out0"], indices [0,1].
        assert_eq!(plan.buffers.len(), 2);
        assert_eq!(plan.buffers[0].name, "y:out0");
        assert_eq!(plan.buffers[0].index, 0);
        assert_eq!(plan.buffers[1].name, "z:out0");
        assert_eq!(plan.buffers[1].index, 1);

        // Quant: two kernels in kernel_id order with identity params and matching arities.
        assert_eq!(plan.quant.len(), 2);
        assert_eq!(plan.quant[0].kernel_id, 0);
        assert_eq!(plan.quant[1].kernel_id, 1);
        assert_eq!(plan.quant[0].inputs.len(), 1);
        assert_eq!(plan.quant[0].outputs.len(), 1);
        assert_eq!(plan.quant[1].inputs.len(), 1);
        assert_eq!(plan.quant[1].outputs.len(), 1);
        for kq in &plan.quant {
            for q in kq.inputs.iter().chain(kq.outputs.iter()) {
                assert_eq!(q.dtype, "f32");
                assert_eq!(q.scale, 1.0);
                assert_eq!(q.zero_point, 0);
            }
        }

        // Determinism: rerun on a fresh identical graph and compare artifacts.
        let mut g2 = make_test_graph();
        let mut ctx2 = PassContext::default();
        let outcome2 = pipeline.run(&mut g2, &mut ctx2).unwrap();
        assert!(matches!(outcome2, PassOutcome::Changed));
        let plan2_v = g2
            .attributes
            .get("layout_and_quant_plan")
            .cloned()
            .expect("layout_and_quant_plan attribute");
        let plan2: LayoutAndQuantPlan = serde_json::from_value(plan2_v).unwrap();
        assert_eq!(plan, plan2);
    }

    #[test]
    fn dump_json_matches_golden() {
        let mut g = make_test_graph();

        // Build descriptor with dump_dir on memory_layout_and_quant
        let dir = std::env::temp_dir().join("nc_passes_mlq_tests");
        let _ = fs::create_dir_all(&dir);
        let mlq_cfg = serde_json::json!({ "dump_dir": dir.to_string_lossy() });

        let passes = vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: Some(mlq_cfg) },
        ];
        let desc = PipelineDescriptor { passes };

        let mut reg = Registry::<nir::Graph>::new();
        crate::lower_to_kernels::register(&mut reg);
        super::register(&mut reg);
        let pipeline = reg.build_pipeline(&desc).unwrap();

        let mut ctx = PassContext::default();
        ctx.run_id = Some("t-run-1".into());
        let outcome = pipeline.run(&mut g, &mut ctx).unwrap();
        assert!(matches!(outcome, PassOutcome::Changed));

        let path = dir.join("layout_quant_t-run-1.json");
        let s = fs::read_to_string(&path).expect("dump exists");
        let plan_from_file: LayoutAndQuantPlan = serde_json::from_str(&s).unwrap();

        // Expected golden inline
        let golden = r#"
{
  "buffers": [
    { "name": "y:out0", "index": 0 },
    { "name": "z:out0", "index": 1 }
  ],
  "quant": [
    {
      "kernel_id": 0,
      "inputs": [ { "dtype": "f32", "scale": 1.0, "zero_point": 0 } ],
      "outputs": [ { "dtype": "f32", "scale": 1.0, "zero_point": 0 } ]
    },
    {
      "kernel_id": 1,
      "inputs": [ { "dtype": "f32", "scale": 1.0, "zero_point": 0 } ],
      "outputs": [ { "dtype": "f32", "scale": 1.0, "zero_point": 0 } ]
    }
  ]
}"#;
        let golden_plan: LayoutAndQuantPlan = serde_json::from_str(golden).unwrap();
        assert_eq!(plan_from_file, golden_plan);
    }
}
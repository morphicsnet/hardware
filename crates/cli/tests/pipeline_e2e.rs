use std::fs;
use std::path::PathBuf;

use tempfile::tempdir;

use nc_nir as nir;
use nc_passes as passes;
use passes::generic::{PipelineDescriptor, PassSpec, PassContext};

#[test]
fn e2e_pipeline_simple_smoke_produces_dumps() {
    // Locate repository root and input NIR JSON
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_dir = PathBuf::from(manifest_dir);
    let ws_root = crate_dir.parent().and_then(|p| p.parent()).expect("ws root");
    let input = ws_root.join("examples/nir/pipeline_simple.json");

    let src = fs::read_to_string(&input).expect("read examples/nir/simple.json");
    let mut g = nir::Graph::from_json_str(&src).expect("parse NIR");

    // Prepare dump_dir and deterministic run_id
    let tmp = tempdir().expect("tempdir");
    let dump_dir = tmp.path().to_path_buf();
    let run_id = "e2e_simple";

    // Build generic pipeline descriptor with per-pass dump config
    let dump_cfg = serde_json::json!({
        "dump_dir": dump_dir.to_string_lossy(),
        "dump_all": "1",
        "metrics": "0"
    });
    let passes = vec![
        PassSpec { name: "lower_to_kernels".into(), config: Some(dump_cfg.clone()) },
        PassSpec { name: "memory_layout_and_quant".into(), config: Some(dump_cfg.clone()) },
        PassSpec { name: "kernel_fusion_and_scheduling".into(), config: Some(dump_cfg.clone()) },
        PassSpec { name: "validation".into(), config: Some(dump_cfg.clone()) },
    ];
    let desc = PipelineDescriptor { passes };

    // Build registry and pipeline
    let reg = passes::default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).expect("build pipeline");

    // Run with context
    let mut ctx = PassContext::default();
    ctx.run_id = Some(run_id.to_string());
    let _ = pipeline.run(&mut g, &mut ctx).expect("pipeline run ok");

    // Assert dumps exist
    let kernels = dump_dir.join(format!("kernels_{}.json", run_id));
    let layout = dump_dir.join(format!("layout_quant_{}.json", run_id));
    let schedule = dump_dir.join(format!("schedule_{}.json", run_id));
    let validation = dump_dir.join(format!("validation_{}.json", run_id));

    assert!(kernels.exists(), "expected dump file: {}", kernels.display());
    assert!(layout.exists(), "expected dump file: {}", layout.display());
    assert!(schedule.exists(), "expected dump file: {}", schedule.display());
    assert!(validation.exists(), "expected dump file: {}", validation.display());

    // Check validation content
    let s = fs::read_to_string(&validation).expect("read validation dump");
    let v: serde_json::Value = serde_json::from_str(&s).expect("parse validation json");
    assert_eq!(v.get("ok").and_then(|x| x.as_bool()), Some(true), "validation ok not true: {}", s);
    assert!(v.get("errors").and_then(|x| x.as_array()).map(|a| a.is_empty()).unwrap_or(false), "validation errors not empty: {}", s);
}
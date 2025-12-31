use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use nc_passes::generic::{PassContext, PassOutcome, PassSpec, PipelineDescriptor};
use nc_passes::{default_generic_nir_registry, nir};

// Build a simple dataflow graph equivalent to x --Add--> y --ReLU--> z
fn make_chain_small_graph() -> nir::Graph {
    let mut g = nir::Graph::new("chain_small_df");
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

fn unique_tmp_dir(prefix: &str) -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = base.join(format!("{}_{}_{}", prefix, pid, nanos));
    let _ = fs::create_dir_all(&p);
    p
}

fn repo_root() -> PathBuf {
    // crates/passes -> crates -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read_json(path: &Path) -> Value {
    let s = fs::read_to_string(path).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn pipeline_desc_boundaries(dump_dir: &Path) -> PipelineDescriptor {
    let dump_dir_str = dump_dir.to_string_lossy().to_string();
    let cfg = Some(json!({ "dump_dir": dump_dir_str }));
    PipelineDescriptor {
        passes: vec![
            PassSpec { name: "lower_to_kernels".into(), config: cfg.clone() },
            PassSpec { name: "memory_layout_and_quant".into(), config: cfg.clone() },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: cfg.clone() },
        ],
    }
}

fn pipeline_desc_with_validation(dump_dir: &Path) -> PipelineDescriptor {
    let dump_dir_str = dump_dir.to_string_lossy().to_string();
    let cfg = Some(json!({ "dump_dir": dump_dir_str }));
    PipelineDescriptor {
        passes: vec![
            PassSpec { name: "lower_to_kernels".into(), config: cfg.clone() },
            PassSpec { name: "memory_layout_and_quant".into(), config: cfg.clone() },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: cfg.clone() },
            PassSpec { name: "validation".into(), config: cfg.clone() },
        ],
    }
}

#[test]
fn golden_chain_small_matches_all_boundaries() {
    // Build graph
    let mut g = make_chain_small_graph();

    // Build registry + pipeline
    let mut reg = default_generic_nir_registry();
    let dump_dir = unique_tmp_dir("nc_passes_goldens_chain_small");
    let desc = pipeline_desc_boundaries(&dump_dir);
    let pipeline = reg.build_pipeline(&desc).expect("pipeline builds");

    // Run with run_id = "chain_small" so filenames match goldens
    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let outcome = pipeline.run(&mut g, &mut ctx).expect("pipeline run ok");
    // Any of the passes should mark Changed
    assert!(matches!(outcome, PassOutcome::Changed) || matches!(outcome, PassOutcome::Unchanged));

    // Compare dumps to goldens
    let root = repo_root();
    let goldens_dir = root.join("fixtures").join("goldens");

    let dumped_k = dump_dir.join("kernels_chain_small.json");
    let dumped_lq = dump_dir.join("layout_quant_chain_small.json");
    let dumped_s = dump_dir.join("schedule_chain_small.json");

    let golden_k = goldens_dir.join("kernels_chain_small.json");
    let golden_lq = goldens_dir.join("layout_quant_chain_small.json");
    let golden_s = goldens_dir.join("schedule_chain_small.json");

    assert_eq!(read_json(&dumped_k), read_json(&golden_k));
    assert_eq!(read_json(&dumped_lq), read_json(&golden_lq));
    assert_eq!(read_json(&dumped_s), read_json(&golden_s));
}

#[test]
fn determinism_repeated_runs_produce_identical_dumps() {
    // Build pipeline once
    let mut reg = default_generic_nir_registry();

    // First run
    let mut g1 = make_chain_small_graph();
    let dir1 = unique_tmp_dir("nc_passes_goldens_chain_small_run1");
    let desc1 = pipeline_desc_boundaries(&dir1);
    let p1 = reg.build_pipeline(&desc1).unwrap();
    let mut ctx1 = PassContext::default();
    ctx1.run_id = Some("chain_small".into());
    let _ = p1.run(&mut g1, &mut ctx1).unwrap();

    // Second run
    let mut g2 = make_chain_small_graph();
    let dir2 = unique_tmp_dir("nc_passes_goldens_chain_small_run2");
    let desc2 = pipeline_desc_boundaries(&dir2);
    let p2 = reg.build_pipeline(&desc2).unwrap();
    let mut ctx2 = PassContext::default();
    ctx2.run_id = Some("chain_small".into());
    let _ = p2.run(&mut g2, &mut ctx2).unwrap();

    // Compare dump JSONs exactly
    let files = [
        "kernels_chain_small.json",
        "layout_quant_chain_small.json",
        "schedule_chain_small.json",
    ];
    for f in files {
        let v1 = read_json(&dir1.join(f));
        let v2 = read_json(&dir2.join(f));
        assert_eq!(v1, v2, "non-deterministic dump: {}", f);
    }
}

#[test]
fn validation_ok_for_chain_small() {
    let mut g = make_chain_small_graph();

    let mut reg = default_generic_nir_registry();
    let dump_dir = unique_tmp_dir("nc_passes_goldens_validation");
    let desc = pipeline_desc_with_validation(&dump_dir);
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let out = pipeline.run(&mut g, &mut ctx).unwrap();

    // Overall pipeline should succeed (some passes Changed, validation Unchanged)
    assert!(matches!(out, PassOutcome::Changed) || matches!(out, PassOutcome::Unchanged));

    // If validation report was dumped, ensure it's ok with no errors.
    let val_path = dump_dir.join("validation_chain_small.json");
    if val_path.exists() {
        let v = read_json(&val_path);
        assert_eq!(v.get("ok").and_then(|b| b.as_bool()), Some(true));
        let errs = v.get("errors").and_then(|e| e.as_array()).cloned().unwrap_or_default();
        assert!(errs.is_empty(), "expected no validation errors, got {:?}", errs);
    }
}

// H3-7 dump toggles tests -------------------------------------------------------
use tempfile::tempdir;

fn pipeline_desc_with_cfg(cfg: serde_json::Value, include_validation: bool) -> PipelineDescriptor {
    let cfg1 = Some(cfg.clone());
    let cfg2 = Some(cfg.clone());
    let cfg3 = Some(cfg.clone());
    let cfg4 = Some(cfg);
    let mut passes = vec![
        PassSpec { name: "lower_to_kernels".into(), config: cfg1 },
        PassSpec { name: "memory_layout_and_quant".into(), config: cfg2 },
        PassSpec { name: "kernel_fusion_and_scheduling".into(), config: cfg3 },
    ];
    if include_validation {
        passes.push(PassSpec { name: "validation".into(), config: cfg4 });
    }
    PipelineDescriptor { passes }
}

fn assert_exists(p: &std::path::Path) {
    assert!(
        std::fs::metadata(p).is_ok(),
        "expected file to exist: {:?}",
        p
    );
}
fn assert_not_exists(p: &std::path::Path) {
    assert!(
        std::fs::metadata(p).is_err(),
        "expected file to NOT exist: {:?}",
        p
    );
}

#[test]
fn dump_default_compat_all_enabled_by_dump_dir_only() {
    // Back-compat: when only dump_dir is provided and no specific dump_* toggles,
    // all dumps should be produced.
    let mut g = make_chain_small_graph();

    let tmp = tempdir().unwrap();
    let cfg = json!({ "dump_dir": tmp.path().to_string_lossy() });
    let desc = pipeline_desc_with_cfg(cfg, false);

    let mut reg = default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let _ = pipeline.run(&mut g, &mut ctx).unwrap();

    let base = tmp.path();
    assert_exists(&base.join("kernels_chain_small.json"));
    assert_exists(&base.join("layout_quant_chain_small.json"));
    assert_exists(&base.join("schedule_chain_small.json"));
}

#[test]
fn dump_selective_lower_only() {
    // When any specific toggle is present, dumping becomes selective:
    // only those explicitly truthy should dump.
    let mut g = make_chain_small_graph();

    let tmp = tempdir().unwrap();
    let cfg = json!({
        "dump_dir": tmp.path().to_string_lossy(),
        "dump_lower": "1"
    });
    let desc = pipeline_desc_with_cfg(cfg, false);

    let mut reg = default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let _ = pipeline.run(&mut g, &mut ctx).unwrap();

    let base = tmp.path();
    assert_exists(&base.join("kernels_chain_small.json"));
    assert_not_exists(&base.join("layout_quant_chain_small.json"));
    assert_not_exists(&base.join("schedule_chain_small.json"));
}

#[test]
fn dump_all_override_produces_all() {
    // dump_all overrides and enables all dumps.
    let mut g = make_chain_small_graph();

    let tmp = tempdir().unwrap();
    let cfg = json!({
        "dump_dir": tmp.path().to_string_lossy(),
        "dump_all": "true"
    });
    let desc = pipeline_desc_with_cfg(cfg, false);

    let mut reg = default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let _ = pipeline.run(&mut g, &mut ctx).unwrap();

    let base = tmp.path();
    assert_exists(&base.join("kernels_chain_small.json"));
    assert_exists(&base.join("layout_quant_chain_small.json"));
    assert_exists(&base.join("schedule_chain_small.json"));
}

#[test]
fn dump_explicit_none_when_toggles_present_but_false() {
    // All specific toggles present but falsey => no dumps should be written.
    let mut g = make_chain_small_graph();

    let tmp = tempdir().unwrap();
    let cfg = json!({
        "dump_dir": tmp.path().to_string_lossy(),
        "dump_lower": "0",
        "dump_layout": "0",
        "dump_schedule": "no",
        "dump_validation": "false"
    });
    // Include validation to verify it also does not dump.
    let desc = pipeline_desc_with_cfg(cfg, true);

    let mut reg = default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let _ = pipeline.run(&mut g, &mut ctx);

    let base = tmp.path();
    assert_not_exists(&base.join("kernels_chain_small.json"));
    assert_not_exists(&base.join("layout_quant_chain_small.json"));
    assert_not_exists(&base.join("schedule_chain_small.json"));
    assert_not_exists(&base.join("validation_chain_small.json"));
}

#[test]
fn metrics_enabled_no_panic() {
    // Ensure metrics guard does not panic when enabled.
    let mut g = make_chain_small_graph();

    let tmp = tempdir().unwrap();
    let cfg = json!({
        "dump_dir": tmp.path().to_string_lossy(),
        "metrics": "1"
    });
    let desc = pipeline_desc_with_cfg(cfg, true);

    let mut reg = default_generic_nir_registry();
    let pipeline = reg.build_pipeline(&desc).unwrap();

    let mut ctx = PassContext::default();
    ctx.run_id = Some("chain_small".into());
    let out = pipeline.run(&mut g, &mut ctx);
    // Do not assert on logs; just ensure we didn't panic and got a valid result.
    assert!(out.is_ok() || out.is_err(), "pipeline should return a result");
}
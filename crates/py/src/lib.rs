#![doc = r#"
Python SDK (nc) and Rust helpers

This crate exposes:
- A stable Rust helper surface used internally for importing, compiling, simulating, and deploying NIR graphs.
- A minimal, stable Python API surface via pyo3 under the Python module named `nc`.

Stability notes:
- For milestone M2-1, the Python API provides the following functions: version(), compile(), simulate(), deploy().
- version() returns the crate version string.
- compile()/simulate()/deploy() are stubs that raise NcNotImplemented. Implementations land in M2-2.
- Exceptions are registered on the `nc` module: NcError (base), NcValidationError, NcRuntimeError, NcNotImplemented.

This file only modifies the Python SDK crate surface. No other crates are changed.
"#]

use anyhow::Result;
use std::collections::HashMap;
#[cfg(feature = "telemetry")]
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

const VERSION: &str = match option_env!("CARGO_PKG_VERSION") {
    Some(v) => v,
    None => "0.0.0-dev",
};

// Rust API always available
pub fn version() -> &'static str { VERSION }
pub fn list_targets() -> Vec<&'static str> { nc_hal::builtin_targets().to_vec() }

pub fn import_nir_json_str(s: &str) -> Result<nc_nir::Graph> {
    let g = nc_nir::Graph::from_json_str(s)?;
    Ok(g)
}

pub fn import_nir_yaml_str(s: &str) -> Result<nc_nir::Graph> {
    let g = nc_nir::Graph::from_yaml_str(s)?;
    Ok(g)
}

pub fn compile_stub(target: &str) -> Result<String> {
    // Placeholder compile path
    Ok(format!("compile: target={target}"))
}

/// Compile NIR from JSON string for a specific target (feature-gated backends).
pub fn compile_nir_json_str(target: &str, json: &str) -> Result<String> {
    let mut g = nc_nir::Graph::from_json_str(json)?;
    g.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.ensure_version_tag();
    let manifest_path = std::path::PathBuf::from(format!("targets/{target}.toml"));
    let manifest = nc_hal::parse_target_manifest_path(&manifest_path)?;
    nc_hal::validate_manifest(&manifest)?;

    match target {
        "truenorth" => {
            #[cfg(feature = "backend-truenorth")]
            { return nc_backend_truenorth::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-truenorth"))]
            { anyhow::bail!("backend 'truenorth' not enabled; build python crate with feature 'backend-truenorth'"); }
        }
        "dynaps" => {
            #[cfg(feature = "backend-dynaps")]
            { return nc_backend_dynaps::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-dynaps"))]
            { anyhow::bail!("backend 'dynaps' not enabled; build python crate with feature 'backend-dynaps'"); }
        }
        // RISC-V targets (compile-only from Python by default).
        // Runtime execution via QEMU/Renode is controlled out-of-process with env:
        //   NC_RISCV_QEMU_RUN=1  -> attempt run if toolchains are present
        //   NC_RISCV_QEMU_RUN=0  -> compile-only (unit tests use this)
        "riscv64gcv_linux" | "riscv32imac_bare" | "riscv64gc_ctrl" => {
            #[cfg(feature = "backend-riscv")]
            { return nc_backend_riscv::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-riscv"))]
            { anyhow::bail!("backend 'riscv' not enabled; build python crate with feature 'backend-riscv'"); }
        }
        other => anyhow::bail!("unsupported target '{other}'"),
    }
}

/// Compile NIR from YAML string for a specific target (feature-gated backends).
pub fn compile_nir_yaml_str(target: &str, yaml: &str) -> Result<String> {
    let mut g = nc_nir::Graph::from_yaml_str(yaml)?;
    g.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.ensure_version_tag();
    let manifest_path = std::path::PathBuf::from(format!("targets/{target}.toml"));
    let manifest = nc_hal::parse_target_manifest_path(&manifest_path)?;
    nc_hal::validate_manifest(&manifest)?;

    match target {
        "truenorth" => {
            #[cfg(feature = "backend-truenorth")]
            { return nc_backend_truenorth::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-truenorth"))]
            { anyhow::bail!("backend 'truenorth' not enabled; build python crate with feature 'backend-truenorth'"); }
        }
        "dynaps" => {
            #[cfg(feature = "backend-dynaps")]
            { return nc_backend_dynaps::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-dynaps"))]
            { anyhow::bail!("backend 'dynaps' not enabled; build python crate with feature 'backend-dynaps'"); }
        }
        // RISC-V targets (compile-only from Python by default).
        // Runtime execution via QEMU/Renode is controlled out-of-process with env:
        //   NC_RISCV_QEMU_RUN=1  -> attempt run if toolchains are present
        //   NC_RISCV_QEMU_RUN=0  -> compile-only (unit tests use this)
        "riscv64gcv_linux" | "riscv32imac_bare" | "riscv64gc_ctrl" => {
            #[cfg(feature = "backend-riscv")]
            { return nc_backend_riscv::compile(&g, &manifest); }
            #[cfg(not(feature = "backend-riscv"))]
            { anyhow::bail!("backend 'riscv' not enabled; build python crate with feature 'backend-riscv'"); }
        }
        other => anyhow::bail!("unsupported target '{other}'"),
    }
}

/// Compile NIR from a string (auto-detect JSON vs YAML) for a specific target.
pub fn compile_nir_str(target: &str, s: &str) -> Result<String> {
    let t = s.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        compile_nir_json_str(target, s)
    } else {
        // try YAML first; if it fails, fallback to JSON
        compile_nir_yaml_str(target, s).or_else(|_| compile_nir_json_str(target, s))
    }
}


pub fn simulate_stub(sim: &str) -> Result<String> {
    Ok(format!("simulate: simulator={sim}"))
}

/// Simulate NIR from JSON string for a specified simulator, writing artifacts to out_dir (or default).
pub fn simulate_nir_json_str(simulator: &str, json: &str, out_dir: Option<&str>) -> Result<String> {
    let mut g = nc_nir::Graph::from_json_str(json)?;
    g.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.ensure_version_tag();

    let out_path = match out_dir {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(format!("target/sim-{simulator}-py-out")),
    };
    // Keep the variable marked as used even when all simulator features are disabled
    let _ = &out_path;

    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| nc_telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let mut labels = BTreeMap::new();
    #[cfg(feature = "telemetry")]
    {
        labels.insert("simulator".to_string(), simulator.to_string());
        labels.insert("graph".to_string(), g.name.clone());
    }

    #[cfg(feature = "telemetry")]
    let __timer_emit = app.as_ref().map(|a| a.start_timer("py.simulate.emit_ms", labels.clone()));

    let emit_result: anyhow::Result<()> = match simulator {
        "neuron" => {
            #[cfg(feature = "sim-neuron")]
            {
                nc_sim_neuron::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-neuron"))]
            {
                Err(anyhow::anyhow!("simulator 'neuron' not enabled; build python crate with feature 'sim-neuron'"))
            }
        }
        "coreneuron" => {
            #[cfg(feature = "sim-coreneuron")]
            {
                nc_sim_coreneuron::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-coreneuron"))]
            {
                Err(anyhow::anyhow!("simulator 'coreneuron' not enabled; build python crate with feature 'sim-coreneuron'"))
            }
        }
        "arbor" => {
            #[cfg(feature = "sim-arbor")]
            {
                nc_sim_arbor::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-arbor"))]
            {
                Err(anyhow::anyhow!("simulator 'arbor' not enabled; build python crate with feature 'sim-arbor'"))
            }
        }
        other => {
            anyhow::bail!("unsupported simulator '{other}'");
        }
    };

    emit_result?;

    #[cfg(feature = "telemetry")]
    {
        if let Some(a) = &app {
            let _ = a.counter("graph.populations", g.populations.len() as f64, labels.clone());
            let _ = a.counter("graph.connections", g.connections.len() as f64, labels.clone());
            let _ = a.counter("graph.probes", g.probes.len() as f64, labels.clone());
        }
    }

    Ok(out_path.to_string_lossy().to_string())
}

/// Simulate NIR from YAML string for a specified simulator, writing artifacts to out_dir (or default).
pub fn simulate_nir_yaml_str(simulator: &str, yaml: &str, out_dir: Option<&str>) -> Result<String> {
    let mut g = nc_nir::Graph::from_yaml_str(yaml)?;
    g.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    g.ensure_version_tag();

    let out_path = match out_dir {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(format!("target/sim-{simulator}-py-out")),
    };
    // Keep the variable marked as used even when all simulator features are disabled
    let _ = &out_path;

    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| nc_telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let mut labels = BTreeMap::new();
    #[cfg(feature = "telemetry")]
    {
        labels.insert("simulator".to_string(), simulator.to_string());
        labels.insert("graph".to_string(), g.name.clone());
    }

    #[cfg(feature = "telemetry")]
    let __timer_emit = app.as_ref().map(|a| a.start_timer("py.simulate.emit_ms", labels.clone()));

    let emit_result: anyhow::Result<()> = match simulator {
        "neuron" => {
            #[cfg(feature = "sim-neuron")]
            {
                nc_sim_neuron::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-neuron"))]
            {
                Err(anyhow::anyhow!("simulator 'neuron' not enabled; build python crate with feature 'sim-neuron'"))
            }
        }
        "coreneuron" => {
            #[cfg(feature = "sim-coreneuron")]
            {
                nc_sim_coreneuron::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-coreneuron"))]
            {
                Err(anyhow::anyhow!("simulator 'coreneuron' not enabled; build python crate with feature 'sim-coreneuron'"))
            }
        }
        "arbor" => {
            #[cfg(feature = "sim-arbor")]
            {
                nc_sim_arbor::emit_artifacts(&g, &out_path)?;
                Ok(())
            }
            #[cfg(not(feature = "sim-arbor"))]
            {
                Err(anyhow::anyhow!("simulator 'arbor' not enabled; build python crate with feature 'sim-arbor'"))
            }
        }
        other => {
            anyhow::bail!("unsupported simulator '{other}'");
        }
    };

    emit_result?;

    #[cfg(feature = "telemetry")]
    {
        if let Some(a) = &app {
            let _ = a.counter("graph.populations", g.populations.len() as f64, labels.clone());
            let _ = a.counter("graph.connections", g.connections.len() as f64, labels.clone());
            let _ = a.counter("graph.probes", g.probes.len() as f64, labels.clone());
        }
    }

    Ok(out_path.to_string_lossy().to_string())
}

/// Simulate NIR from a string (auto-detect JSON vs YAML) for a specified simulator.
pub fn simulate_nir_str(simulator: &str, s: &str, out_dir: Option<&str>) -> Result<String> {
    let t = s.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        simulate_nir_json_str(simulator, s, out_dir)
    } else {
        // try YAML first; if it fails, fallback to JSON
        simulate_nir_yaml_str(simulator, s, out_dir)
            .or_else(|_| simulate_nir_json_str(simulator, s, out_dir))
    }
}

/// Summarize a JSONL profiling file into CSV metrics: metric,count,avg,min,max
pub fn profile_summary_jsonl(path: &str) -> Result<String> {
    let file = File::open(path)?;
    let rdr = BufReader::new(file);
    let mut stats: HashMap<String, (usize, f64, f64, f64)> = HashMap::new(); // count,sum,min,max
    for l in rdr.lines().map_while(Result::ok) {
        if l.trim().is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
            let metric = v.get("metric").and_then(|m| m.as_str()).unwrap_or("unknown");
            let value = v.get("value").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let e = stats.entry(metric.to_string())
                .or_insert((0, 0.0, f64::INFINITY, f64::NEG_INFINITY));
            e.0 += 1;
            e.1 += value;
            if value < e.2 { e.2 = value; }
            if value > e.3 { e.3 = value; }
        }
    }
    let mut out = String::from("metric,count,avg,min,max\n");
    for (m, (c, sum, min, max)) in stats {
        let avg = if c > 0 { sum / c as f64 } else { 0.0 };
        out.push_str(&format!("{m},{c},{avg:.4},{min:.4},{max:.4}\n"));
    }
    Ok(out)
}

/// Deploy stub (placeholder for runtime-backed deployment)
pub fn deploy_stub(target: &str) -> Result<String> {
    Ok(format!("deploy: target={target}"))
}


#[cfg(feature = "python")]
use pyo3::create_exception;
#[cfg(feature = "python")]
use pyo3::exceptions::PyException;
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python exceptions exposed on the `nc` module.
#[cfg(feature = "python")]
create_exception!(nc, NcError, PyException);
#[cfg(feature = "python")]
create_exception!(nc, NcValidationError, NcError);
#[cfg(feature = "python")]
create_exception!(nc, NcRuntimeError, NcError);
#[cfg(feature = "python")]
create_exception!(nc, NcNotImplemented, NcError);

// -----------------------
// Shared helpers (Rust-only; thin Python wrappers call these)
// -----------------------

use std::path::{Path, PathBuf};
use std::fs;

/// Artifacts produced by the pass pipeline.
#[derive(Debug, Clone)]
pub struct Artifacts {
    pub artifacts_dir: PathBuf,
    pub kernels: PathBuf,
    pub layout_quant: PathBuf,
    pub schedule: PathBuf,
    pub validation: Option<PathBuf>,
}

fn ensure_dir(path: &Path) -> Result<(), std::io::Error> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

fn default_run_id() -> String { "py_run".to_string() }

fn build_pipeline_descriptor() -> nc_passes::generic::PipelineDescriptor {
    use nc_passes::generic::PassSpec;
    nc_passes::generic::PipelineDescriptor {
        passes: vec![
            PassSpec { name: "lower_to_kernels".into(), config: None },
            PassSpec { name: "memory_layout_and_quant".into(), config: None },
            PassSpec { name: "kernel_fusion_and_scheduling".into(), config: None },
            PassSpec { name: "validation".into(), config: None },
        ],
    }
}

/// Execute the generic NIR pipeline and dump artifacts into dump_dir.
/// Returns canonical paths to emitted files.
///
/// Behavior:
/// - Builds a registry with built-in passes from nc-passes
/// - Executes the pipeline with a per-pass config carrying dump toggles
/// - If a validation report exists and ok=false, returns Err with a clear message
pub fn run_pipeline_and_dump_core(
    mut module: nc_nir::Graph,
    dump_dir: &Path,
    run_id: &str,
    dump_all: bool,
    metrics: bool,
) -> anyhow::Result<Artifacts> {
    // Prepare registry and pipeline
    let mut reg = nc_passes::default_generic_nir_registry();
    let desc = build_pipeline_descriptor();
    let pipeline = reg.build_pipeline(&desc)
        .map_err(|e| anyhow::anyhow!("build pipeline: {e}"))?;

    // Per-pass config object with toggles
    let cfg = serde_json::json!({
        "dump_dir": dump_dir.to_string_lossy(),
        "dump_all": dump_all,
        "metrics": metrics,
    });

    // The generic::Pipeline stores per-step configs inside the descriptor,
    // so rebuild steps with the same config for each step.
    let desc_cfg = nc_passes::generic::PipelineDescriptor {
        passes: desc.passes.into_iter().map(|s| {
            nc_passes::generic::PassSpec { name: s.name, config: Some(cfg.clone()) }
        }).collect(),
    };
    let pipeline = reg.build_pipeline(&desc_cfg)
        .map_err(|e| anyhow::anyhow!("build pipeline (cfg): {e}"))?;

    // Ensure module minimally valid and tagged
    module.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    module.ensure_version_tag();

    // Create dump dir and run
    ensure_dir(dump_dir).map_err(|e| anyhow::anyhow!("create dump_dir: {e}"))?;
    let mut ctx = nc_passes::generic::PassContext { config: None, run_id: Some(run_id.to_string()) };

    let run_res = pipeline.run(&mut module, &mut ctx);

    // Artifact paths
    let kernels = dump_dir.join(format!("kernels_{}.json", run_id));
    let layout_quant = dump_dir.join(format!("layout_quant_{}.json", run_id));
    let schedule = dump_dir.join(format!("schedule_{}.json", run_id));
    let validation_path = dump_dir.join(format!("validation_{}.json", run_id));

    // Check validation report first if present
    let mut validation: Option<PathBuf> = None;
    if validation_path.exists() {
        validation = Some(validation_path.clone());
        // Try to parse ok flag
        if let Ok(s) = std::fs::read_to_string(&validation_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if v.get("ok").and_then(|b| b.as_bool()) == Some(false) {
                    // Summarize first few errors if available
                    let summary = v.get("errors")
                        .and_then(|e| e.as_array())
                        .map(|arr| {
                            arr.iter().take(3)
                                .filter_map(|d| d.get("message").and_then(|m| m.as_str()))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        })
                        .unwrap_or_default();
                    return Err(anyhow::anyhow!(format!("validation failed (see {}): {}",
                        validation_path.display(), summary)));
                }
            }
        }
    }

    // If pipeline errored but no validation failure was detected, propagate pipeline error
    if let Err(e) = run_res {
        return Err(anyhow::anyhow!(format!("pipeline execution error: {e}")));
    }

    // Canonicalize paths where possible
    let can = |p: PathBuf| std::fs::canonicalize(&p).unwrap_or(p);

    Ok(Artifacts {
        artifacts_dir: std::fs::canonicalize(dump_dir).unwrap_or(dump_dir.to_path_buf()),
        kernels: can(kernels),
        layout_quant: can(layout_quant),
        schedule: can(schedule),
        validation: validation.map(can),
    })
}

/// Validate presence of expected artifacts in a directory (used by simulate/deploy and unit tests).
pub fn validate_artifacts_dir(dir: &Path) -> anyhow::Result<Artifacts> {
    if !dir.exists() {
        return Err(anyhow::anyhow!(format!("artifacts_dir does not exist: {}", dir.display())));
    }
    // Find any matching files regardless of run_id
    let mut kernels: Option<PathBuf> = None;
    let mut layout: Option<PathBuf> = None;
    let mut schedule: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;

    for entry in std::fs::read_dir(dir).map_err(|e| anyhow::anyhow!(e))? {
        let entry = entry.map_err(|e| anyhow::anyhow!(e))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("kernels_") && name.ends_with(".json") {
            kernels.get_or_insert(entry.path());
        } else if name.starts_with("layout_quant_") && name.ends_with(".json") {
            layout.get_or_insert(entry.path());
        } else if name.starts_with("schedule_") && name.ends_with(".json") {
            schedule.get_or_insert(entry.path());
        } else if name.starts_with("validation_") && name.ends_with(".json") {
            validation.get_or_insert(entry.path());
        }
    }

    let miss = |label: &str| anyhow::anyhow!(format!("missing required artifact '{label}' in {}", dir.display()));
    let kernels = kernels.ok_or_else(|| miss("kernels"))?;
    let layout_quant = layout.ok_or_else(|| miss("layout_quant"))?;
    let schedule = schedule.ok_or_else(|| miss("schedule"))?;

    Ok(Artifacts {
        artifacts_dir: std::fs::canonicalize(dir).unwrap_or(dir.to_path_buf()),
        kernels: std::fs::canonicalize(&kernels).unwrap_or(kernels),
        layout_quant: std::fs::canonicalize(&layout_quant).unwrap_or(layout_quant),
        schedule: std::fs::canonicalize(&schedule).unwrap_or(schedule),
        validation: validation.map(|p| std::fs::canonicalize(&p).unwrap_or(p)),
    })
}

#[cfg(feature = "python")]
fn is_truthy_py(val: &pyo3::PyAny) -> bool {
    if let Ok(b) = val.extract::<bool>() { return b; }
    if let Ok(i) = val.extract::<i64>() { return i != 0; }
    if let Ok(s) = val.str() {
        let s = s.to_string_lossy().to_ascii_lowercase();
        return matches!(s.as_str(), "1" | "true" | "yes");
    }
    false
}

/// Accepts either:
/// - str/Path-like: load JSON/YAML file and parse into nc_nir::Graph
/// - dict: serialize via json.dumps then parse
#[cfg(feature = "python")]
fn parse_nir_input(py: Python<'_>, nir: &pyo3::PyAny) -> PyResult<nc_nir::Graph> {
    use pyo3::types::{PyDict, PyString};

    // If it's a str, interpret as a filesystem path
    if let Ok(s) = nir.downcast::<PyString>() {
        let path = s.to_string_lossy().to_string();
        let content = std::fs::read_to_string(&path)
            .map_err(|e| NcError::new_err(format!("failed to read NIR file '{}': {}", path, e)))?;
        // Auto-detect JSON vs YAML
        let t = content.trim_start();
        let g = if t.starts_with('{') || t.starts_with('[') {
            nc_nir::Graph::from_json_str(&content)
                .map_err(|e| NcError::new_err(format!("invalid NIR JSON: {}", e)))?
        } else {
            nc_nir::Graph::from_yaml_str(&content)
                .or_else(|_| nc_nir::Graph::from_json_str(&content))
                .map_err(|e| NcError::new_err(format!("invalid NIR (YAML/JSON): {}", e)))?
        };
        return Ok(g);
    }

    // Try Path-like via os.fspath
    let os = py.import("os").map_err(|e| NcError::new_err(format!("import os failed: {e}")))?;
    if let Ok(path_obj) = os.getattr("fspath").and_then(|f| f.call1((nir,))) {
        if let Ok(ps) = path_obj.downcast::<PyString>() {
            let path = ps.to_string_lossy().to_string();
            let content = std::fs::read_to_string(&path)
                .map_err(|e| NcError::new_err(format!("failed to read NIR file '{}': {}", path, e)))?;
            let t = content.trim_start();
            let g = if t.starts_with('{') || t.starts_with('[') {
                nc_nir::Graph::from_json_str(&content)
                    .map_err(|e| NcError::new_err(format!("invalid NIR JSON: {}", e)))?
            } else {
                nc_nir::Graph::from_yaml_str(&content)
                    .or_else(|_| nc_nir::Graph::from_json_str(&content))
                    .map_err(|e| NcError::new_err(format!("invalid NIR (YAML/JSON): {}", e)))?
            };
            return Ok(g);
        }
    }

    // Try dict -> json.dumps
    if nir.is_instance_of::<PyDict>()? {
        let json = py.import("json")
            .and_then(|m| m.getattr("dumps"))
            .map_err(|e| NcError::new_err(format!("import json.dumps failed: {e}")))?;
        let s: String = json.call1((nir,))?
            .extract()
            .map_err(|e| NcError::new_err(format!("json.dumps failed: {e}")))?;
        let g = nc_nir::Graph::from_json_str(&s)
            .map_err(|e| NcError::new_err(format!("invalid NIR dict (JSON conversion): {}", e)))?;
        return Ok(g);
    }

    Err(NcError::new_err("unsupported NIR input type; pass a file path (JSON/YAML) or a dict"))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "version")]
/// Return the Python SDK version string.
///
/// Returns
///     str: The crate version string (from Cargo.toml), or "0.0.0-dev" if unavailable.
///
/// Examples
///     >>> import nc
///     >>> isinstance(nc.version(), str)
///     True
fn py_version() -> &'static str {
    crate::version()
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "compile")]
#[pyo3(signature = (nir, *, targets=None, dump_dir=None, run_id=None, dump_all=None, metrics=false))]
/// Compile a NIR schema by running the pass pipeline and emit artifacts.
///
/// Parameters
///     nir (str | dict): Path to NIR JSON/YAML file or an in-memory NIR schema (dict).
///     targets (Optional[List[str]]): Opaque passthrough; echoed back in result.
///     dump_dir (Optional[str | os.PathLike]): Directory to write compilation artifacts. If None, a temp dir is created and persisted.
///     run_id (Optional[str]): Identifier used in artifact filenames. Defaults to "py_run".
///     dump_all (Optional[bool]): If True (default), dump all intermediate artifacts.
///     metrics (bool): If True, enable lightweight metrics logging in the pass pipeline.
///
/// Returns
///     dict:
///         {
///           "artifacts_dir": str,
///           "files": {
///             "kernels": str,
///             "layout_quant": str,
///             "schedule": str,
///             "validation": Optional[str]
///           },
///           "run_id": str,
///           "targets": Optional[List[str]]
///         }
///
/// Raises
///     NcValidationError: When the validation pass reports ok == false.
///     NcRuntimeError: For I/O or pipeline execution errors.
///
fn compile_pyapi(
    nir: pyo3::PyObject,
    targets: Option<Vec<String>>,
    dump_dir: Option<String>,
    run_id: Option<String>,
    dump_all: Option<bool>,
    metrics: bool,
) -> PyResult<pyo3::PyObject> {
    pyo3::Python::with_gil(|py| {
        // Parse NIR
        let any = nir.as_ref(py);
        let module = parse_nir_input(py, any)?;

        // Resolve dump directory (persist tempdir by into_path)
        let artifacts_dir = match dump_dir {
            Some(d) => {
                let p = std::path::PathBuf::from(d);
                ensure_dir(&p).map_err(|e| NcRuntimeError::new_err(format!("create dump_dir: {e}")))?;
                p
            }
            None => {
                let t = tempfile::Builder::new()
                    .prefix("nc_py_artifacts_")
                    .tempdir()
                    .map_err(|e| NcRuntimeError::new_err(format!("tempdir: {e}")))?;
                t.into_path()
            }
        };

        let run_id = run_id.unwrap_or_else(default_run_id);
        let dump_all = dump_all.unwrap_or(true);

        // Execute pipeline
        let res = run_pipeline_and_dump_core(module, &artifacts_dir, &run_id, dump_all, metrics);
        let artifacts = match res {
            Ok(a) => a,
            Err(err) => {
                // Try to detect validation error by checking validation report existence with ok=false
                let vpath = artifacts_dir.join(format!("validation_{}.json", run_id));
                if vpath.exists() {
                    if let Ok(s) = std::fs::read_to_string(&vpath) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                            if v.get("ok").and_then(|b| b.as_bool()) == Some(false) {
                                let msg = v.get("errors")
                                    .and_then(|e| e.as_array())
                                    .map(|arr| {
                                        arr.iter().take(5)
                                            .filter_map(|d| {
                                                let code = d.get("code").and_then(|x| x.as_str()).unwrap_or("?");
                                                let path = d.get("path").and_then(|x| x.as_str()).unwrap_or("?");
                                                let m = d.get("message").and_then(|x| x.as_str()).unwrap_or("");
                                                Some(format!("{} @ {}: {}", code, path, m))
                                            })
                                            .collect::<Vec<_>>()
                                            .join(" | ")
                                    })
                                    .unwrap_or_else(|| err.to_string());
                                return Err(NcValidationError::new_err(msg));
                            }
                        }
                    }
                }
                return Err(NcRuntimeError::new_err(err.to_string()));
            }
        };

        // Build Python dict result
        let py_files = pyo3::types::PyDict::new(py);
        py_files.set_item("kernels", artifacts.kernels.to_string_lossy().to_string())?;
        py_files.set_item("layout_quant", artifacts.layout_quant.to_string_lossy().to_string())?;
        py_files.set_item("schedule", artifacts.schedule.to_string_lossy().to_string())?;
        if let Some(v) = artifacts.validation.as_ref() {
            py_files.set_item("validation", v.to_string_lossy().to_string())?;
        } else {
            py_files.set_item("validation", py.None())?;
        }

        let out = pyo3::types::PyDict::new(py);
        out.set_item("artifacts_dir", artifacts.artifacts_dir.to_string_lossy().to_string())?;
        out.set_item("files", py_files)?;
        out.set_item("run_id", run_id.clone())?;
        if let Some(t) = targets.as_ref() {
            out.set_item("targets", t)?;
        } else {
            out.set_item("targets", py.None())?;
        }

        Ok(out.into_py(py))
    })
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "simulate")]
#[pyo3(signature = (compiled, *, profile=false))]
/// Validate compiled artifacts and return an ok result (minimal simulator stub).
///
/// Parameters
///     compiled (str | dict): Path to artifacts_dir or the dict returned by nc.compile().
///     profile (bool): If True, collect and include profiling data (not implemented; returns None).
///
/// Returns
///     dict: { "ok": True, "profile": None, "metrics": None }
///
/// Raises
///     NcValidationError: If required artifact files are missing.
///
fn simulate_pyapi(
    compiled: pyo3::PyObject,
    profile: bool,
) -> PyResult<pyo3::PyObject> {
    let _ = profile;
    pyo3::Python::with_gil(|py| {
        // Resolve artifacts_dir and/or explicit file list
        let any = compiled.as_ref(py);
        let (artifacts_dir, explicit_files): (std::path::PathBuf, Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf, Option<std::path::PathBuf>)>) = if let Ok(s) = any.extract::<String>() {
            (std::path::PathBuf::from(s), None)
        } else {
            // Expect dict shape from compile()
            let d = any.downcast::<pyo3::types::PyDict>()
                .map_err(|_| NcError::new_err("compiled must be a path (str) or dict with 'artifacts_dir' and 'files'"))?;
            let dir_s: String = d.get_item("artifacts_dir")
                .ok_or_else(|| NcError::new_err("compiled dict missing 'artifacts_dir'"))?
                .extract()
                .map_err(|e| NcError::new_err(format!("extract artifacts_dir: {e}")))?;
            let files = d.get_item("files")
                .ok_or_else(|| NcError::new_err("compiled dict missing 'files'"))?
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| NcError::new_err("'files' must be a dict"))?;
            let k: String = files.get_item("kernels").ok_or_else(|| NcError::new_err("files.kernels missing"))?.extract().map_err(|e| NcError::new_err(format!("files.kernels: {e}")))?;
            let l: String = files.get_item("layout_quant").ok_or_else(|| NcError::new_err("files.layout_quant missing"))?.extract().map_err(|e| NcError::new_err(format!("files.layout_quant: {e}")))?;
            let s: String = files.get_item("schedule").ok_or_else(|| NcError::new_err("files.schedule missing"))?.extract().map_err(|e| NcError::new_err(format!("files.schedule: {e}")))?;
            let v: Option<String> = match files.get_item("validation") {
                Some(x) if !x.is_none() => Some(x.extract().map_err(|e| NcError::new_err(format!("files.validation: {e}")))?),
                _ => None,
            };
            (std::path::PathBuf::from(dir_s),
             Some((std::path::PathBuf::from(k), std::path::PathBuf::from(l), std::path::PathBuf::from(s), v.map(std::path::PathBuf::from))))
        };

        // Validate presence
        let _arts = match explicit_files {
            Some((k, l, s, v)) => {
                if !k.exists() { return Err(NcValidationError::new_err(format!("missing kernels file: {}", k.display()))); }
                if !l.exists() { return Err(NcValidationError::new_err(format!("missing layout_quant file: {}", l.display()))); }
                if !s.exists() { return Err(NcValidationError::new_err(format!("missing schedule file: {}", s.display()))); }
                Artifacts {
                    artifacts_dir: std::fs::canonicalize(&artifacts_dir).unwrap_or(artifacts_dir.clone()),
                    kernels: std::fs::canonicalize(&k).unwrap_or(k),
                    layout_quant: std::fs::canonicalize(&l).unwrap_or(l),
                    schedule: std::fs::canonicalize(&s).unwrap_or(s),
                    validation: v.map(|p| std::fs::canonicalize(&p).unwrap_or(p)),
                }
            }
            None => {
                validate_artifacts_dir(&artifacts_dir)
                    .map_err(|e| NcValidationError::new_err(e.to_string()))?
            }
        };

        // Minimal, successful result
        let out = pyo3::types::PyDict::new(py);
        out.set_item("ok", true)?;
        out.set_item("profile", py.None())?;
        out.set_item("metrics", py.None())?;
        Ok(out.into_py(py))
    })
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "deploy")]
#[pyo3(signature = (compiled, target, *, dry_run=false))]
/// Deploy a compiled artifact to a target.
///
/// Behavior (M2-2):
/// - When dry_run=True, validate inputs and return:
///     { "ok": True, "target": target, "details": { "action": "validated", "dry_run": True } }
/// - When dry_run=False, return NcNotImplemented to indicate runtime wiring is pending.
///
/// Parameters
///     compiled (str | dict): Path to artifacts_dir or the dict returned by nc.compile().
///     target (str): Target identifier (e.g., "riscv64gcv_linux", "sim_hw_specific").
///     dry_run (bool): If True, perform a dry run validation.
///
/// Raises
///     NcValidationError: If required artifacts are missing.
///     NcNotImplemented: For non-dry-run mode (runtime integration pending).
fn deploy_pyapi(
    compiled: pyo3::PyObject,
    target: String,
    dry_run: bool,
) -> PyResult<pyo3::PyObject> {
    pyo3::Python::with_gil(|py| {
        // Validate compiled input similarly to simulate()
        let any = compiled.as_ref(py);
        let artifacts_dir: std::path::PathBuf = if let Ok(s) = any.extract::<String>() {
            std::path::PathBuf::from(s)
        } else {
            let d = any.downcast::<pyo3::types::PyDict>()
                .map_err(|_| NcError::new_err("compiled must be a path (str) or dict with 'artifacts_dir'"))?;
            let dir_s: String = d.get_item("artifacts_dir")
                .ok_or_else(|| NcError::new_err("compiled dict missing 'artifacts_dir'"))?
                .extract()
                .map_err(|e| NcError::new_err(format!("extract artifacts_dir: {e}")))?;
            std::path::PathBuf::from(dir_s)
        };

        // Validate presence of required files
        let _ = validate_artifacts_dir(&artifacts_dir)
            .map_err(|e| NcValidationError::new_err(e.to_string()))?;

        if dry_run {
            let out = pyo3::types::PyDict::new(py);
            let det = pyo3::types::PyDict::new(py);
            det.set_item("action", "validated")?;
            det.set_item("dry_run", true)?;
            out.set_item("ok", true)?;
            out.set_item("target", target)?;
            out.set_item("details", det)?;
            return Ok(out.into_py(py));
        }

        Err(NcNotImplemented::new_err("nc.deploy runtime apply path not wired yet; enable in M2-2 follow-up (runtime integration)"))
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn nc(py: Python, m: &PyModule) -> PyResult<()> {
    // Register exceptions on the module
    m.add("NcError", py.get_type::<NcError>())?;
    m.add("NcValidationError", py.get_type::<NcValidationError>())?;
    m.add("NcRuntimeError", py.get_type::<NcRuntimeError>())?;
    m.add("NcNotImplemented", py.get_type::<NcNotImplemented>())?;

    // Register functions
    m.add_function(wrap_pyfunction!(py_version, m)?)?;
    m.add_function(wrap_pyfunction!(compile_pyapi, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_pyapi, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_pyapi, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Feature-gated Python API test: compile-only for RISC-V (no external tools)
    #[cfg(all(feature = "backend-riscv", feature = "python"))]
    #[test]
    fn py_compile_riscv64gcv_linux_compile_only() {
        std::env::set_var("NC_RISCV_QEMU_RUN", "0"); // ensure no QEMU/Renode invocation
        let nir = std::fs::read_to_string("examples/nir/simple.json").expect("read NIR");
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let m = pyo3::types::PyModule::new(py, "neuro_compiler").expect("module new");
            // Initialize the in-process module with all #[pyfn] exports
            crate::neuro_compiler(py, m).expect("init module");
            let f = m.getattr("compile_nir_str_py").expect("get compile_nir_str_py");
            let art: String = f.call1(("riscv64gcv_linux", nir.as_str()))
                .expect("call ok")
                .extract()
                .expect("extract str");
            if art.starts_with("artifact:") {
                let dir = PathBuf::from(art.trim_start_matches("artifact:"));
                assert!(dir.exists(), "artifact dir should exist: {}", dir.display());
            } else {
                assert!(PathBuf::from(&art).exists(), "artifact path should exist: {}", art);
            }
        });
    }

    // Negative test when RISC-V backend feature is NOT enabled
    #[cfg(not(feature = "backend-riscv"))]
    #[test]
    fn riscv_backend_disabled_has_clear_error() {
        let nir = std::fs::read_to_string("examples/nir/simple.json").expect("read NIR");
        let err = compile_nir_str("riscv64gcv_linux", &nir).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("backend 'riscv' not enabled"), "error: {s}");
        assert!(s.contains("backend-riscv"), "error: {s}");
    }
}

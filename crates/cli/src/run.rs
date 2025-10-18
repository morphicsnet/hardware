use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde_json::{Map, Value};
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(about = "Execute a NIR+EIR plan on a HAL backend", long_about = r#"Execute a NIR+EIR plan on a HAL backend

Examples:
  neuro-compiler run --backend cpu-ref-sim --in plan.nir.json --out out/trace.jsonl --validate-eir
  neuro-compiler run --backend cpu-ref-sim --in plan.nir.json --out out/trace.jsonl --seed 42 --time-unit us --tolerance 1e-4 --duration 0.005
  neuro-compiler run --list-backends
"#)]
pub struct RunArgs {
    /// Backend name to execute (e.g., "cpu-ref-sim")
    #[arg(long = "backend", value_name = "BACKEND_NAME")]
    pub backend: Option<String>,

    /// Input NIR+EIR JSON plan path
    #[arg(long = "in", value_name = "PATH")]
    pub input: Option<PathBuf>,

    /// Output UEC JSONL trace path
    #[arg(long = "out", value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// If provided, runs the EIR validator pass and fails fast on violations
    #[arg(long = "validate-eir")]
    pub validate_eir: bool,

    /// List available HAL backends and exit
    #[arg(long = "list-backends")]
    pub list_backends: bool,

    // Optional passthrough flags forwarded via a JSON opts map (serde_json::Value)
    /// Seed for deterministic execution (u64)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Time unit for the engine: s|ms|us|ns
    #[arg(long = "time-unit", value_name = "UNIT")]
    pub time_unit: Option<String>,

    /// Event-Driven (ED) engine tolerance (float)
    #[arg(long)]
    pub tolerance: Option<f64>,

    /// Discrete-Time (DT) engine step (float)
    #[arg(long)]
    pub step: Option<f64>,

    /// Requested duration in seconds (float)
    #[arg(long)]
    pub duration: Option<f64>,
}

pub fn exec(args: RunArgs) -> Result<()> {
    if args.list_backends {
        let names = list_backends();
        if names.is_empty() {
            println!("No HAL backends are registered. To enable the CPU reference simulator, build with feature: backend-cpu-ref-sim\n  cargo build -F backend-cpu-ref-sim");
        } else {
            for n in names {
                println!("{n}");
            }
        }
        return Ok(());
    }

    // Enforce required flags when not listing backends
    let backend = args
        .backend
        .as_ref()
        .ok_or_else(|| anyhow!("--backend is required unless --list-backends is used"))?;
    let in_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("--in is required unless --list-backends is used"))?;
    let out_path = args
        .out
        .as_ref()
        .ok_or_else(|| anyhow!("--out is required unless --list-backends is used"))?;

    // Load NIR+EIR plan (JSON or YAML)
    let data = fs::read_to_string(in_path)
        .with_context(|| format!("failed to read input plan '{}'", in_path.display()))?;
    let ext = in_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());
    let mut graph: nc_nir::Graph = match ext.as_deref() {
        Some("yaml") | Some("yml") => {
            nc_nir::Graph::from_yaml_str(&data).map_err(|e| anyhow!("parse yaml failed: {e}"))?
        }
        _ => {
            nc_nir::Graph::from_json_str(&data).map_err(|e| anyhow!("parse json failed: {e}"))?
        }
    };

    // EIR validation (non-destructive). If there are violations, fail fast with a clear error.
    if args.validate_eir {
        validate_eir_only(&mut graph)?;
    }

    // Ensure version tag is present (non-invasive metadata).
    graph.ensure_version_tag();

    // Build opts map (only include keys provided)
    let opts = build_opts_json(&args);

    // Ensure output directory exists (so backends have a place to write)
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output directory '{}'", parent.display()))?;
        }
    }

    // Dispatch to HAL by name; feature-gated. Error clearly if backend is unavailable.
    #[cfg(feature = "backend-cpu-ref-sim")]
    {
        let report = nc_hal::backend_registry::run_backend_by_name(
            backend,
            &graph,
            out_path.as_path(),
            opts.as_ref(),
        )
        .with_context(|| format!("HAL backend '{}' failed to execute", backend))?;

        // Success summary
        println!("run ok: backend={} trace={}", backend, out_path.display());

        // Optionally: keep the report around for debugging (silently ignore errors if shape changes).
        let _ = report;
    }

    #[cfg(not(feature = "backend-cpu-ref-sim"))]
    {
        let _ = (&graph, &opts, &backend, &out_path);
        bail!(
            "backend '{}' is not registered. If you intended to use 'cpu-ref-sim', enable the feature and rebuild:\n  cargo build -F backend-cpu-ref-sim",
            backend
        );
    }

    Ok(())
}

// Build a serde_json::Value opts object with only provided keys.
// For cpu-ref-sim (and similar CLI-shim adapters), passthrough flags are forwarded via "extra_args".
fn build_opts_json(args: &RunArgs) -> Option<Value> {
    let mut extra_args: Vec<String> = Vec::new();

    if let Some(seed) = args.seed {
        extra_args.push("--seed".into());
        extra_args.push(seed.to_string());
    }
    if let Some(unit) = args.time_unit.as_deref() {
        extra_args.push("--time-unit".into());
        extra_args.push(unit.to_string());
    }
    if let Some(tol) = args.tolerance {
        extra_args.push("--tolerance".into());
        extra_args.push(tol.to_string());
    }
    if let Some(step) = args.step {
        extra_args.push("--step".into());
        extra_args.push(step.to_string());
    }
    if let Some(dur) = args.duration {
        extra_args.push("--duration".into());
        extra_args.push(dur.to_string());
    }

    if extra_args.is_empty() {
        None
    } else {
        let mut m = Map::new();
        m.insert(
            "extra_args".into(),
            Value::Array(extra_args.into_iter().map(Value::from).collect()),
        );
        Some(Value::Object(m))
    }
}

// Run only the EIR validator pass (no destructive transform to the graph).
fn validate_eir_only(g: &mut nc_nir::Graph) -> Result<()> {
    use nc_passes::generic::{PassContext, PassSpec, PipelineDescriptor, Registry};

    let mut reg = Registry::<nc_nir::Graph>::new();
    nc_passes::eir_validate::register(&mut reg);

    let desc = PipelineDescriptor {
        passes: vec![PassSpec {
            name: "eir_validate".into(),
            config: None,
        }],
    };
    let pipeline = reg
        .build_pipeline(&desc)
        .map_err(|e| anyhow!("failed to construct EIR validator pipeline: {e}"))?;

    let mut ctx = PassContext::default();
    match pipeline.run(g, &mut ctx) {
        Ok(_outcome) => Ok(()),
        Err(e) => bail!("{e}"),
    }
}

// Discover available backends via HAL (feature-gated).
fn list_backends() -> Vec<String> {
    #[cfg(feature = "backend-cpu-ref-sim")]
    {
        nc_hal::backend_registry::available_backends()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    #[cfg(not(feature = "backend-cpu-ref-sim"))]
    {
        Vec::new()
    }
}
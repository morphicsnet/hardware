//! CPU reference simulator backend adapter (UEC CLI shim).
//!
//! Licensing note:
//! - This crate does NOT link or copy any UEC code.
//! - It shells out to an external `uec` executable via std::process::Command.
//!
//! Behavior summary:
//! - Accepts a NIR Graph (with EIR attrs) and validates required EIR fields.
//! - Chooses engine "ed" or "dt" based on EIR profile.mode.
//! - Writes NIR JSON to a temp file.
//! - Invokes `${UEC_CLI:-uec} run --engine <ed|dt> --in <nir.json> --out <trace.jsonl> --seed <seed> [--step X|--tolerance Y]`.
//! - Captures status and returns a minimal RunReport.
//! - Dry-run mode: if UEC_CLI_DRYRUN=1, no process is spawned; we only return the constructed command.

use anyhow::{bail, Context, Result};
use nc_nir as nir;
use nc_nir::eir::EirGraphAttrs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Options that control a backend run (forward-compatible; can be extended).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendRunOpts {
    /// Additional CLI args appended after required flags (if the CLI supports them).
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Additional environment variables to set for the spawned process.
    #[serde(default)]
    pub extra_env: Vec<(String, String)>,
}

/// Minimal run report with helpful diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    /// Engine actually used by the adapter ("ed" or "dt").
    pub engine: String,
    /// Seed used for deterministic execution.
    pub seed: u64,
    /// Path to the JSONL trace emitted by the UEC engine (as requested).
    pub trace_path: PathBuf,
    /// The exact command line that was constructed (debugging and reproducibility).
    pub cmd: String,
    /// "ok" | "dry-run"
    pub status: String,
}

/// Adapter struct. Stateless; construct as needed.
#[derive(Debug, Default)]
pub struct CpuRefSimBackend;

impl CpuRefSimBackend {
    /// The registration/discovery name for this backend.
    pub fn name(&self) -> &'static str {
        "cpu-ref-sim"
    }

    /// Execute the provided NIR+EIR plan through the external UEC CLI.
    ///
    /// Behavior:
    /// - Validates EIR presence on the graph (under keys "eir" or "eir_graph_attrs").
    /// - Chooses engine "ed" or "dt" based on EIR profile.mode.
    /// - Serializes the NIR graph to a temp JSON file.
    /// - Spawns `UEC_CLI` (default "uec") with:
    ///     run --engine <ed|dt> --in <nir.json> --out <trace.jsonl> --seed <seed> --time-unit <unit> [--step|--tolerance]
    /// - If the environment variable `UEC_CLI_DRYRUN=1` is set, does not spawn a process; instead returns a report
    ///   with `status="dry-run"` containing the constructed command string.
    pub fn run(&self, plan: &nir::Graph, out_trace: &Path, opts: &BackendRunOpts) -> Result<RunReport> {
        let attrs = load_graph_eir_attrs(plan).ok_or_else(|| {
            anyhow::anyhow!(
                "Missing EIR graph attributes. Provide graph.attributes['eir'] (preferred) \
                 or 'eir_graph_attrs' with valid EIR fields (profile.mode='ED'|'DT', determinism.seed, timing, etc.)."
            )
        })?;

        let engine = match attrs.profile.mode.as_str() {
            "ED" => "ed",
            "DT" => "dt",
            other => {
                bail!("Unsupported EIR profile.mode='{}'. Expected 'ED' or 'DT'.", other);
            }
        };
        let seed = attrs.determinism.seed;

        // Serialize NIR to a temp file (JSON)
        let tmp_dir = std::env::temp_dir().join(format!("uec_nir_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("failed to create temp dir '{}'", tmp_dir.display()))?;
        let nir_path = tmp_dir.join(format!("{}.nir.json", sanitize_filename(&plan.name)));
        let nir_json = plan
            .to_json_string()
            .map_err(|e| anyhow::anyhow!("failed to serialize NIR to JSON: {}", e))?;
        fs::write(&nir_path, nir_json).with_context(|| {
            format!(
                "failed to write serialized NIR JSON to '{}'",
                nir_path.display()
            )
        })?;

        // CLI path
        let cli = std::env::var("UEC_CLI").unwrap_or_else(|_| "uec".to_string());

        // Construct the command line
        let mut cmd = Command::new(&cli);
        cmd.arg("run")
            .arg("--engine")
            .arg(engine)
            .arg("--in")
            .arg(&nir_path)
            .arg("--out")
            .arg(out_trace)
            .arg("--seed")
            .arg(seed.to_string())
            .arg("--time-unit")
            .arg(attrs.timing.time_unit.clone());

        // Thread timing parameters for engine-specific context
        match engine {
            "ed" => {
                if let Some(tol) = attrs.timing.tolerance {
                    cmd.arg("--tolerance").arg(tol.to_string());
                }
                // In ED, ensure we don't accidentally pass step
            }
            "dt" => {
                if let Some(step) = attrs.timing.step {
                    cmd.arg("--step").arg(step.to_string());
                }
                // In DT, tolerance is optional; skip unless CLI explicitly supports it.
            }
            _ => {}
        }

        // Extra env/args
        for (k, v) in &opts.extra_env {
            cmd.env(k, v);
        }
        for a in &opts.extra_args {
            cmd.arg(a);
        }

        // Dry-run mode
        let dry = std::env::var("UEC_CLI_DRYRUN")
            .ok()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let cmd_str = format!("{cmd:?}");
        if dry {
            return Ok(RunReport {
                engine: engine.to_string(),
                seed,
                trace_path: out_trace.to_path_buf(),
                cmd: cmd_str,
                status: "dry-run".into(),
            });
        }

        // Try primary attempt
        let output = cmd.output().with_context(|| {
            format!(
                "failed to spawn UEC CLI at '{}'. Hint: set UEC_CLI to the full path or install 'uec' on PATH.",
                cli
            )
        })?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Optional fallback: if ED fails, try DT with same inputs (best-effort).
            if engine == "ed" {
                let mut cmd2 = Command::new(&cli);
                cmd2.arg("run")
                    .arg("--engine")
                    .arg("dt")
                    .arg("--in")
                    .arg(&nir_path)
                    .arg("--out")
                    .arg(out_trace)
                    .arg("--seed")
                    .arg(seed.to_string())
                    .arg("--time-unit")
                    .arg(attrs.timing.time_unit.clone());
                if let Some(step) = attrs.timing.step {
                    cmd2.arg("--step").arg(step.to_string());
                }
                for (k, v) in &opts.extra_env {
                    cmd2.env(k, v);
                }
                for a in &opts.extra_args {
                    cmd2.arg(a);
                }
                let cmd2_str = format!("{cmd2:?}");
                match cmd2.output() {
                    Ok(o2) if o2.status.success() => {
                        return Ok(RunReport {
                            engine: "dt".into(),
                            seed,
                            trace_path: out_trace.to_path_buf(),
                            cmd: cmd2_str,
                            status: "ok".into(),
                        });
                    }
                    _ => {
                        bail!(
                            "UEC CLI failed.\n\
                             Primary attempt:\n  cmd: {cmd_str}\n  status: {status}\n  stdout: {stdout}\n  stderr: {stderr}\n\
                             Fallback attempt:\n  cmd: {cmd2}\n\
                             Hint: ensure your UEC CLI supports: `uec run --engine <ed|dt> --in <FILE> --out <FILE> --seed <N>`.\n\
                             You can override the CLI path via UEC_CLI.",
                            status = output.status,
                            cmd2 = cmd2_str
                        );
                    }
                }
            } else {
                bail!(
                    "UEC CLI failed.\ncmd: {cmd}\nstatus: {status}\nstdout: {stdout}\nstderr: {stderr}\n\
                     Hint: ensure your UEC CLI supports: `uec run --engine <dt> --in <FILE> --out <FILE> --seed <N>`.\n\
                     You can override the CLI path via UEC_CLI.",
                    cmd = cmd_str,
                    status = output.status
                );
            }
        }

        Ok(RunReport {
            engine: engine.to_string(),
            seed,
            trace_path: out_trace.to_path_buf(),
            cmd: cmd_str,
            status: "ok".into(),
        })
    }
}

/// Load EIR graph-level attributes from known keys: "eir" (preferred) or "eir_graph_attrs" (legacy).
fn load_graph_eir_attrs(g: &nir::Graph) -> Option<EirGraphAttrs> {
    // Preferred key
    if let Some(v) = g.attributes.get("eir") {
        if let Ok(attrs) = serde_json::from_value::<EirGraphAttrs>(v.clone()) {
            return Some(attrs);
        }
    }
    // Fallback key
    if let Some(v) = g.attributes.get("eir_graph_attrs") {
        if let Ok(attrs) = serde_json::from_value::<EirGraphAttrs>(v.clone()) {
            return Some(attrs);
        }
    }
    None
}

/// Sanitize filenames from graph names (very conservative).
fn sanitize_filename(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => out.push(ch),
            _ => out.push('_'),
        }
    }
    if out.is_empty() {
        "_".into()
    } else {
        out
    }
}

#[cfg(all(test, feature = "backend-cpu-ref-sim"))]
mod tests {
    use super::*;
    use nc_nir::eir;

    fn make_eir_attrs_ed(seed: u64) -> eir::EirGraphAttrs {
        eir::EirGraphAttrs {
            timing: eir::EirTiming {
                time_unit: "ms".into(),
                step: None,
                tolerance: Some(0.001),
            },
            determinism: eir::EirDeterminism::with_defaults(seed),
            units: eir::EirUnits {
                voltage: "mV".into(),
                conductance: "uS".into(),
                current: "nA".into(),
            },
            profile: eir::EirProfile {
                mode: "ED".into(),
                trace_level: "basic".into(),
                delivery_snapping: false,
            },
        }
    }

    #[test]
    fn dry_run_builds_expected_command_for_ed() {
        // Build a minimal NIR graph with EIR attrs
        let mut g = nir::Graph::new("dry_run_test");
        let attrs = make_eir_attrs_ed(42);
        g.attributes
            .insert("eir".to_string(), serde_json::to_value(attrs).unwrap());

        // Output path (no file is actually created in dry-run)
        let out = std::env::temp_dir().join("uec_trace.jsonl");

        // Ensure dry-run
        std::env::set_var("UEC_CLI_DRYRUN", "1");
        // Force a deterministic CLI path in the string for assertions
        std::env::set_var("UEC_CLI", "uec");

        let backend = CpuRefSimBackend::default();
        let report = backend
            .run(&g, &out, &BackendRunOpts::default())
            .expect("dry-run should succeed");

        assert_eq!(report.status, "dry-run");
        assert_eq!(report.engine, "ed");
        assert_eq!(report.seed, 42);
        assert!(report.trace_path.ends_with("uec_trace.jsonl"));

        // Check the command string contains key bits (Debug-format of Command is platform dependent,
        // so we only assert for the presence of substrings).
        let s = report.cmd.to_lowercase();
        assert!(s.contains("uec"));
        assert!(s.contains("--engine"));
        assert!(s.contains("ed"));
        assert!(s.contains("--in"));
        assert!(s.contains(".nir.json"));
        assert!(s.contains("--out"));
        assert!(s.contains("uec_trace.jsonl"));
        assert!(s.contains("--seed"));
        assert!(s.contains("42"));

        // cleanup env
        std::env::remove_var("UEC_CLI_DRYRUN");
        std::env::remove_var("UEC_CLI");
    }
}
//! EIR validation pass (non-invasive).
//!
//! This pass checks EIR-on-NIR graph attributes if present and returns
//! user-friendly diagnostics. When attributes are absent, the pass is a no-op.
//!
//! Supported keys for graph-level attributes (serde JSON objects):
//! - "eir" (preferred)
//! - "eir_graph_attrs" (legacy/fallback)

use crate::generic::{self, Pass, PassContext, PassError, PassOutcome, PassResult};
use crate::nir;
use crate::nir::eir::{self, EirGraphAttrs};

pub struct ValidateEir;

impl Pass<nir::Graph> for ValidateEir {
    fn name(&self) -> &'static str { "eir_validate" }

    fn run(&self, module: &mut nir::Graph, _ctx: &mut PassContext) -> PassResult {
        // Tolerate absence of EIR attributes entirely (no-op).
        let Some(attrs) = load_graph_attrs(module) else {
            return Ok(PassOutcome::Unchanged);
        };

        let mut errors: Vec<String> = Vec::new();

        // time_unit in {"s","ms","us","ns"}
        {
            let ok_units = ["s", "ms", "us", "ns"];
            if !ok_units.contains(&attrs.timing.time_unit.as_str()) {
                errors.push(format!(
                    "timing.time_unit='{}' must be one of {:?}",
                    attrs.timing.time_unit, ok_units
                ));
            }
        }

        // profile.mode in {"ED","DT"} with timing coupling rules
        {
            let m = attrs.profile.mode.as_str();
            match m {
                "ED" => {
                    // tolerance must be Some(>0); step must be None
                    match attrs.timing.tolerance {
                        Some(t) if t > 0.0 => {}
                        Some(_) => errors.push("ED mode requires timing.tolerance > 0".to_string()),
                        None => errors.push("ED mode requires timing.tolerance to be specified".to_string()),
                    }
                    if attrs.timing.step.is_some() {
                        errors.push("ED mode requires timing.step to be None".to_string());
                    }
                }
                "DT" => {
                    // step must be Some(>0); tolerance may be present (informational)
                    match attrs.timing.step {
                        Some(s) if s > 0.0 => {}
                        Some(_) => errors.push("DT mode requires timing.step > 0".to_string()),
                        None => errors.push("DT mode requires timing.step to be specified".to_string()),
                    }
                    // tolerance is optional and informational in DT; no strict check required
                }
                _ => {
                    errors.push(format!(
                        "profile.mode='{}' must be 'ED' or 'DT'",
                        attrs.profile.mode
                    ));
                }
            }
        }

        // determinism.rng == "pcg64"; determinism.schedule == "stable_total_order"
        {
            let expected_rng = eir::default_rng();
            if attrs.determinism.rng.as_str() != expected_rng {
                errors.push(format!(
                    "determinism.rng='{}' must be '{}'",
                    attrs.determinism.rng, expected_rng
                ));
            }
            let expected_sched = eir::default_schedule();
            if attrs.determinism.schedule.as_str() != expected_sched {
                errors.push(format!(
                    "determinism.schedule='{}' must be '{}'",
                    attrs.determinism.schedule, expected_sched
                ));
            }
        }

        // Units strings non-empty (trimmed)
        {
            if attrs.units.voltage.trim().is_empty() {
                errors.push("units.voltage must be a non-empty string".to_string());
            }
            if attrs.units.conductance.trim().is_empty() {
                errors.push("units.conductance must be a non-empty string".to_string());
            }
            if attrs.units.current.trim().is_empty() {
                errors.push("units.current must be a non-empty string".to_string());
            }
        }

        if errors.is_empty() {
            // Pure validator; does not modify the module.
            return Ok(PassOutcome::Unchanged);
        }

        // Summarize errors in a single message consistent with other passes.
        let summary = format!(
            "EIR validation failed with {} error(s): {}",
            errors.len(),
            errors.join(" | ")
        );
        Err(PassError::InvalidInput(summary))
    }
}

/// Try to load graph-level EIR attributes from known keys.
fn load_graph_attrs(g: &nir::Graph) -> Option<EirGraphAttrs> {
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

/// Constructor for registry
pub fn mk(_: Option<&serde_json::Value>) -> Box<dyn Pass<nir::Graph>> {
    Box::new(ValidateEir)
}

/// Register under the pass name "eir_validate".
pub fn register(reg: &mut generic::Registry<nir::Graph>) {
    reg.register("eir_validate", mk);
}
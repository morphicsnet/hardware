//! EIR-on-NIR attribute schema.
//!
//! The types defined here model EIR attributes that are attached to a NIR graph
//! and optionally to nodes. These are serde-serializable and intentionally keep
//! enums as Strings for cross-repo compatibility.

use serde::{Deserialize, Serialize};

/// Returns the default RNG algorithm identifier used by EIR determinism.
pub fn default_rng() -> &'static str { "pcg64" }

/// Returns the default scheduling policy identifier for deterministic execution.
pub fn default_schedule() -> &'static str { "stable_total_order" }

/// Timing configuration for EIR execution.
///
/// - `time_unit`: one of "s", "ms", "us", or "ns".
/// - `step`: for discrete-time (DT) mode, the step size in `time_unit`.
/// - `tolerance`: for event-driven (ED) mode, allowable delivery tolerance in `time_unit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirTiming {
    /// Base unit for time values ("s","ms","us","ns").
    pub time_unit: String,
    /// Fixed step duration (only for DT mode); must be > 0 when present.
    #[serde(default)]
    pub step: Option<f64>,
    /// Delivery tolerance (only for ED mode); must be > 0 when present.
    #[serde(default)]
    pub tolerance: Option<f64>,
}

/// Determinism knobs for reproducible execution.
///
/// All fields are required. `rng` and `schedule` should typically be the
/// helper defaults exposed in this module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirDeterminism {
    /// The random seed used to initialize pseudo-random generators.
    pub seed: u64,
    /// RNG algorithm identifier (e.g., "pcg64").
    pub rng: String,
    /// Scheduling policy identifier (e.g., "stable_total_order").
    pub schedule: String,
}

impl EirDeterminism {
    /// Convenience constructor using module defaults for `rng` and `schedule`.
    pub fn with_defaults(seed: u64) -> Self {
        Self {
            seed,
            rng: default_rng().to_string(),
            schedule: default_schedule().to_string(),
        }
    }
}

/// Unit strings for physical quantities in the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirUnits {
    /// Voltage unit name (e.g., "mV").
    pub voltage: String,
    /// Conductance unit name (e.g., "uS").
    pub conductance: String,
    /// Current unit name (e.g., "nA").
    pub current: String,
}

/// Profile describing execution mode and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirProfile {
    /// Execution mode: "ED" (event-driven) or "DT" (discrete-time).
    pub mode: String,
    /// Trace verbosity level (stringly-typed to remain forward-compatible).
    pub trace_level: String,
    /// If true, delivery times may be snapped to the nearest representable tick.
    pub delivery_snapping: bool,
}

/// Optional attributes that may be associated with a node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirNodeAttrs {
    /// High-level node class/type identifier.
    pub class: String,
    /// Opaque, schema-free parameters specific to the class.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Attributes applied to an entire NIR graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EirGraphAttrs {
    /// Global timing configuration.
    pub timing: EirTiming,
    /// Deterministic execution controls.
    pub determinism: EirDeterminism,
    /// Units for physical quantities.
    pub units: EirUnits,
    /// Execution profile and tracing configuration.
    pub profile: EirProfile,
}
use anyhow::Result;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;
#[cfg(feature = "telemetry")]
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct DeploySpec {
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeStatus {
    pub running: bool,
}

pub fn deploy(_spec: &DeploySpec) -> Result<()> {
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let mut labels = BTreeMap::new();
            labels.insert("target".to_string(), _spec.target.clone());
            Some(a.start_timer("runtime.deploy_ms", labels))
        } else { None }
    };

    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let mut l = BTreeMap::new();
        l.insert("target".to_string(), _spec.target.clone());
        let _ = a.counter("runtime.deploy_requests", 1.0, l);
    }

    Ok(())
}

pub fn start() -> Result<()> {
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());
    #[cfg(feature = "telemetry")]
    let _t = app.as_ref().map(|a| {
        let labels = BTreeMap::new();
        a.start_timer("runtime.start_ms", labels)
    });
    Ok(())
}

pub fn stop() -> Result<()> {
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());
    #[cfg(feature = "telemetry")]
    let _t = app.as_ref().map(|a| {
        let labels = BTreeMap::new();
        a.start_timer("runtime.stop_ms", labels)
    });
    Ok(())
}

pub fn status() -> RuntimeStatus {
    RuntimeStatus { running: false }
}

pub fn version() -> &'static str { "0.0.1" }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lifecycle_stubs_succeed() {
        let spec = DeploySpec { target: "riscv64gcv_linux".to_string() };
        deploy(&spec).expect("deploy ok");
        start().expect("start ok");
        stop().expect("stop ok");
        let s = status();
        assert!(!s.running);
    }
}

pub mod adaptive {
    //! Adaptive runtime decision application.
    //!
    //! This module defines a minimal Decision API and a clear, documented error taxonomy
    //! used by the runtime to apply adaptive control decisions.

    #[cfg(feature = "telemetry")]
    use nc_telemetry as telemetry;
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    /// A lightweight snapshot of relevant runtime resource signals used by policies to decide.
    #[derive(Debug, Clone)]
    pub struct ResourceSnapshot {
        pub utilization_pct: f32,
        pub buffer_occupancy_pct: f32,
    }

    impl ResourceSnapshot {
        pub fn new(utilization_pct: f32, buffer_occupancy_pct: f32) -> Self {
            Self { utilization_pct, buffer_occupancy_pct }
        }
    }

    /// Decision expresses the runtime action to take following a policy evaluation.
    ///
    /// Semantics:
    /// - NoChange: No-op. The current configuration remains. This must not alter idempotency state.
    /// - Repartition: Re-plan the graph partitioning and apply it (feature-gated to the orchestrator).
    /// - Reschedule: Adjust execution ordering or timing. Unavailable in this revision.
    /// - Throttle: Apply rate limiting to execution or IO. Unavailable in this revision.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Decision {
        /// No action required; safe no-op.
        NoChange,
        /// Repartition the workload via the orchestrator integration.
        Repartition,
        /// Reschedule work ordering/time-slicing (not yet supported).
        Reschedule,
        /// Apply throttling/back-pressure (not yet supported).
        Throttle,
    }

    /// Policy encapsulates a strategy that maps a snapshot to a decision.
    pub trait Policy {
        fn name(&self) -> &str;
        fn decide(&self, snapshot: &ResourceSnapshot) -> Decision;
    }

    /// A degenerate policy that always returns NoChange.
    pub struct NoOpPolicy;

    impl Policy for NoOpPolicy {
        fn name(&self) -> &str { "noop-policy" }
        fn decide(&self, _snapshot: &ResourceSnapshot) -> Decision { Decision::NoChange }
    }

    /// Options controlling application of a decision.
    ///
    /// - idempotency_key: When set, this call is idempotent within the current process. If the
    ///   same key is applied again for a mutating decision (Repartition/Reschedule/Throttle), the
    ///   call must fail with RuntimeError::IdempotencyConflict.
    /// - dry_run: When true, no side-effects are performed (telemetry aside).
    #[derive(Debug, Clone, Default)]
    pub struct ApplyOptions {
        /// Optional idempotency key to de-duplicate repeated requests.
        pub idempotency_key: Option<String>,
        /// When true, record intent only without applying side effects.
        pub dry_run: bool,
    }

    /// RuntimeError is the typed error taxonomy for applying runtime decisions.
    ///
    /// Variants:
    /// - InvalidState: The runtime is in a state that cannot accept the decision (precondition failure).
    /// - NotSupported: The decision path exists conceptually but is not implemented/enabled yet.
    /// - IntegrationUnavailable: A required integration is not present (e.g., a feature not enabled).
    /// - IdempotencyConflict: The provided idempotency key was already applied in this process.
    /// - ApplyFailed: A downstream component failed to apply the change.
    /// - RollbackFailed: Failure while attempting to roll back a partial apply.
    /// - ConcurrencyConflict: Concurrent modification detected; the apply should be retried or aborted.
    #[derive(Debug)]
    pub enum RuntimeError {
        InvalidState(String),
        NotSupported(String),
        IntegrationUnavailable(String),
        IdempotencyConflict(String),
        ApplyFailed(String),
        RollbackFailed(String),
        ConcurrencyConflict(String),
    }

    impl std::fmt::Display for RuntimeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RuntimeError::InvalidState(s) => write!(f, "invalid state: {}", s),
                RuntimeError::NotSupported(s) => write!(f, "not supported: {}", s),
                RuntimeError::IntegrationUnavailable(s) => write!(f, "integration unavailable: {}", s),
                RuntimeError::IdempotencyConflict(k) => write!(f, "idempotency conflict for key: {}", k),
                RuntimeError::ApplyFailed(s) => write!(f, "apply failed: {}", s),
                RuntimeError::RollbackFailed(s) => write!(f, "rollback failed: {}", s),
                RuntimeError::ConcurrencyConflict(s) => write!(f, "concurrency conflict: {}", s),
            }
        }
    }

    impl std::error::Error for RuntimeError {}

    /// Result alias for decision application operations.
    pub type Result<T> = std::result::Result<T, RuntimeError>;

    // Simple in-process idempotency registry (thread-safe).
    static IDEM_REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

    fn idem() -> &'static Mutex<HashSet<String>> {
        IDEM_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
    }

    /// Returns true if the key has already been applied for a mutating decision; otherwise records it.
    ///
    /// Note: NoChange does not participate in idempotency tracking.
    fn already_applied_and_mark(decision: &Decision, key: &str) -> bool {
        if matches!(decision, Decision::NoChange) {
            return false;
        }
        let m = idem();
        let mut set = m.lock().expect("idempotency mutex poisoned");
        if set.contains(key) {
            true
        } else {
            set.insert(key.to_string());
            false
        }
    }

    /// Minimal orchestrator shim for the Repartition path.
    ///
    /// This function is feature-gated and will only compile when the "orchestrator" feature
    /// is enabled for the runtime crate. It delegates to the orchestrator crate to compute
    /// a partition plan. Future work will wire the plan into live runtime state.
    #[cfg(feature = "orchestrator")]
    fn orchestrator_repartition_shim() -> Result<()> {
        // Leverage orchestrator API to ensure linkage and provide forward-compatible hook.
        let g = nc_nir::Graph::new("adaptive-repartition-probe");
        let targets = ["riscv64gcv_linux"];
        match nc_orchestrator::partition(&g, &targets) {
            Ok(_plan) => Ok(()),
            Err(e) => Err(RuntimeError::ApplyFailed(format!("orchestrator partition failed: {}", e))),
        }
    }

    /// Apply a decision to the running system with options.
    ///
    /// Behavior:
    /// - Dry-run returns Ok(()) without side-effects.
    /// - NoChange returns Ok(()) and does not alter idempotency state.
    /// - Repartition delegates to the orchestrator when the "orchestrator" feature is enabled.
    ///   Otherwise returns IntegrationUnavailable("feature 'orchestrator' not enabled").
    /// - Reschedule and Throttle return NotSupported with targeted messages.
    /// - For mutating decisions, duplicate idempotency_key returns IdempotencyConflict.
    pub fn apply_with_options(decision: &Decision, opts: &ApplyOptions) -> Result<()> {
        // Optional telemetry: count decisions with labels.
        #[cfg(feature = "telemetry")]
        {
            let app = std::env::var("NC_PROFILE_JSONL")
                .ok()
                .and_then(|p| telemetry::profiling::Appender::open(p).ok());
            if let Some(a) = app.as_ref() {
                let mut labels = std::collections::BTreeMap::new();
                labels.insert("decision".to_string(), format!("{:?}", decision));
                if let Some(k) = &opts.idempotency_key {
                    labels.insert("idem".to_string(), k.clone());
                }
                let _ = a.counter("runtime.decisions", 1.0, labels);
            }
        }

        // Dry-run: no side effects beyond optional telemetry above.
        if opts.dry_run {
            return Ok(());
        }

        // NoChange: explicitly avoid idempotency tracking.
        if matches!(decision, Decision::NoChange) {
            return Ok(());
        }

        // Idempotency for mutating decisions.
        if let Some(k) = &opts.idempotency_key {
            if already_applied_and_mark(decision, k) {
                return Err(RuntimeError::IdempotencyConflict(k.clone()));
            }
        }

        match decision {
            Decision::NoChange => Ok(()), // unreachable due to guard above; keep for exhaustiveness
            Decision::Repartition => {
                #[cfg(feature = "orchestrator")]
                {
                    orchestrator_repartition_shim()
                }
                #[cfg(not(feature = "orchestrator"))]
                {
                    Err(RuntimeError::IntegrationUnavailable(
                        "feature 'orchestrator' not enabled".to_string(),
                    ))
                }
            }
            Decision::Reschedule => Err(RuntimeError::NotSupported(
                "reschedule path not yet available".to_string(),
            )),
            Decision::Throttle => Err(RuntimeError::NotSupported(
                "throttle path not yet available".to_string(),
            )),
        }
    }

    /// Apply a decision to the running system (defaults: no idempotency key, not dry-run).
    pub fn apply(decision: &Decision) -> Result<()> {
        apply_with_options(decision, &ApplyOptions::default())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn noop_policy_decides_no_change() {
            let p = NoOpPolicy;
            let s = ResourceSnapshot::new(50.0, 10.0);
            assert_eq!(p.decide(&s), Decision::NoChange);
        }

        #[test]
        fn no_change_is_noop() {
            let opts = ApplyOptions { idempotency_key: Some("k-noop".into()), dry_run: false };
            apply_with_options(&Decision::NoChange, &opts).expect("no-change ok");

            // Using the same key for a mutating decision must NOT conflict because NoChange
            // must not have mutated the idempotency registry.
            let res = apply_with_options(&Decision::Repartition, &opts);
            if let Err(RuntimeError::IdempotencyConflict(k)) = &res {
                panic!("NoChange mutated idempotency registry; conflict on key {}", k);
            }
        }

        #[test]
        fn idempotency_rejects_duplicate_key() {
            // Use a mutating decision that is currently NotSupported to exercise idempotency.
            let opts = ApplyOptions { idempotency_key: Some("dup-key".into()), dry_run: false };
            let _ = apply_with_options(&Decision::Throttle, &opts); // first attempt marks key
            let res2 = apply_with_options(&Decision::Throttle, &opts);
            match res2 {
                Err(RuntimeError::IdempotencyConflict(k)) => assert_eq!(k, "dup-key"),
                other => panic!("expected IdempotencyConflict, got {:?}", other),
            }
        }

        #[cfg(not(feature = "orchestrator"))]
        #[test]
        fn repartition_feature_disabled_returns_integration_unavailable() {
            let opts = ApplyOptions { idempotency_key: Some("repart-1".into()), dry_run: false };
            let res = apply_with_options(&Decision::Repartition, &opts);
            match res {
                Err(RuntimeError::IntegrationUnavailable(msg)) => {
                    assert_eq!(msg, "feature 'orchestrator' not enabled");
                }
                other => panic!("expected IntegrationUnavailable, got {:?}", other),
            }
        }

        #[cfg(feature = "orchestrator")]
        #[test]
        fn repartition_feature_enabled_path_compiles() {
            let opts = ApplyOptions { idempotency_key: Some("repart-ok".into()), dry_run: false };
            let res = apply_with_options(&Decision::Repartition, &opts);
            // Either we reach the shim and get Ok, or the shim returns a typed error.
            assert!(res.is_ok()
                || matches!(res, Err(RuntimeError::NotSupported(_)) | Err(RuntimeError::ApplyFailed(_))));
        }

        #[test]
        fn reschedule_and_throttle_not_supported() {
            let r = apply_with_options(&Decision::Reschedule, &ApplyOptions::default());
            assert!(matches!(r, Err(RuntimeError::NotSupported(_))));
            let t = apply_with_options(&Decision::Throttle, &ApplyOptions::default());
            assert!(matches!(t, Err(RuntimeError::NotSupported(_))));
        }
    }
}

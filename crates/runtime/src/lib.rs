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

#[cfg(feature = "hal-shims")]
pub struct HalHooks {
    pub reschedule: fn() -> crate::adaptive::Result<()>,
    pub throttle: fn(percent: u8) -> crate::adaptive::Result<()>,
}

#[cfg(all(feature = "hal-shims", not(test)))]
static HAL_HOOKS: std::sync::OnceLock<HalHooks> = std::sync::OnceLock::new();

#[cfg(all(feature = "hal-shims", test))]
static TEST_HAL_HOOKS: std::sync::Mutex<Option<HalHooks>> = std::sync::Mutex::new(None);

#[cfg(feature = "hal-shims")]
#[allow(dead_code)]
pub fn register_hal_hooks(hooks: HalHooks) -> bool {
    #[cfg(test)]
    {
        let mut g = TEST_HAL_HOOKS.lock().expect("TEST_HAL_HOOKS mutex poisoned");
        let is_new = g.is_none();
        *g = Some(hooks);
        return is_new;
    }
    #[cfg(not(test))]
    {
        HAL_HOOKS.set(hooks).is_ok()
    }
}

#[cfg(all(feature = "hal-shims", test))]
#[doc(hidden)]
pub fn __test_clear_hal_hooks() {
    let mut g = TEST_HAL_HOOKS.lock().expect("TEST_HAL_HOOKS mutex poisoned");
    *g = None;
}

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

    #[cfg(feature = "telemetry")]
    use tracing::{event, Level};

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
        /// Optional hint may carry reason/context for auditing.
        Repartition { hint: Option<String> },
        /// Reschedule work ordering/time-slicing (not yet supported).
        /// Optional hint may carry reason/context for auditing.
        Reschedule { hint: Option<String> },
        /// Apply throttling/back-pressure (not yet supported).
        /// Percent in [1, 100]; validated in apply. Values outside the range will
        /// be rejected with InvalidState.
        Throttle { percent: u8 },
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
    static IDEMPOTENCY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

    fn idem() -> &'static Mutex<HashSet<String>> {
        IDEMPOTENCY.get_or_init(|| Mutex::new(HashSet::new()))
    }

    /// Insert the idempotency key if not present. Returns true if this is the first time seen,
    /// false if the key already existed.
    fn idempotency_seen_insert(key: &str) -> bool {
        let m = idem();
        let mut set = m.lock().expect("idempotency mutex poisoned");
        if set.contains(key) {
            false
        } else {
            set.insert(key.to_string());
            true
        }
    }

    /// Remove a previously inserted idempotency key. No-op if not present.
    #[allow(dead_code)]
    fn idempotency_remove(key: &str) {
        let m = idem();
        let mut set = m.lock().expect("idempotency mutex poisoned");
        set.remove(key);
    }

    // --- Telemetry helpers (private) ---
    #[cfg(feature = "telemetry")]
    fn decision_tag(decision: &Decision) -> &'static str {
        match decision {
            Decision::NoChange => "no_change",
            Decision::Repartition { .. } => "repartition",
            Decision::Reschedule { .. } => "reschedule",
            Decision::Throttle { .. } => "throttle",
        }
    }

    #[cfg(feature = "telemetry")]
    fn error_kind(err: &RuntimeError) -> &'static str {
        match err {
            RuntimeError::InvalidState(_) => "invalid_state",
            RuntimeError::NotSupported(_) => "not_supported",
            RuntimeError::IntegrationUnavailable(_) => "integration_unavailable",
            RuntimeError::IdempotencyConflict(_) => "idempotency_conflict",
            RuntimeError::ApplyFailed(_) => "apply_failed",
            RuntimeError::RollbackFailed(_) => "rollback_failed",
            RuntimeError::ConcurrencyConflict(_) => "concurrency_conflict",
        }
    }

    #[cfg(not(feature = "orchestrator"))]
    mod orchestrator_shim {
        use super::RuntimeError;
    
        // NOTE: Default path returns NotSupported; caller handles idempotency reservation/compensation.
        pub fn repartition_minimal() -> std::result::Result<(), RuntimeError> {
            Err(RuntimeError::NotSupported(
                "repartition capability not supported; enable feature 'orchestrator' to plan".to_string(),
            ))
        }
    }
    
    #[cfg(feature = "orchestrator")]
    mod orchestrator_shim {
        use super::RuntimeError;
    
        #[cfg(feature = "telemetry")]
        use nc_telemetry as telemetry;
        #[cfg(feature = "telemetry")]
        use std::collections::BTreeMap;
    
        /// Minimal integration path: invoke the orchestrator's planning flow deterministically.
        /// NOTE: Application of the plan is not wired yet; we return IntegrationUnavailable after planning.
        pub fn repartition_minimal() -> std::result::Result<(), RuntimeError> {
            #[cfg(feature = "telemetry")]
            let app = std::env::var("NC_PROFILE_JSONL")
                .ok()
                .and_then(|p| telemetry::profiling::Appender::open(p).ok());
            #[cfg(feature = "telemetry")]
            let _timer = {
                if let Some(a) = app.as_ref() {
                    let mut labels = BTreeMap::new();
                    labels.insert("path".to_string(), "repartition_minimal".to_string());
                    Some(a.start_timer("runtime.repartition_ms", labels))
                } else {
                    None
                }
            };
    
            // Compose a deterministic, minimal graph to exercise the planner.
            let g = nc_orchestrator::nir::Graph::new("runtime-repartition-minimal");
            let targets: [&str; 0] = [];
    
            match nc_orchestrator::partition(&g, &targets) {
                Ok(plan) => {
                    #[cfg(feature = "telemetry")]
                    if let Some(a) = &app {
                        let mut l = BTreeMap::new();
                        l.insert("parts".to_string(), plan.parts.to_string());
                        let _ = a.counter("runtime.repartition.plan_parts", plan.parts as f64, l);
                    }
                    Err(RuntimeError::IntegrationUnavailable(format!(
                        "repartition planning succeeded (parts={}), but apply path is not wired",
                        plan.parts
                    )))
                }
                Err(e) => Err(RuntimeError::ApplyFailed(format!(
                    "orchestrator planning error: {}",
                    e
                ))),
            }
        }
    }

    /// Apply a decision to the running system with options.
    ///
    /// Behavior:
    /// - Performs validation first for all decisions.
    /// - If dry_run is true, returns Ok(()) without side effects or idempotency recording.
    /// - NoChange returns Ok(()) and does not alter idempotency state.
    /// - Repartition delegates to a feature-gated shim when the "orchestrator" feature is enabled.
    ///   Otherwise returns IntegrationUnavailable("feature 'orchestrator' not enabled").
    /// - Reschedule and Throttle return NotSupported with targeted messages (after validation).
    /// - For mutating decisions, duplicate idempotency_key returns IdempotencyConflict.
    ///
    /// Telemetry (feature "telemetry"):
    /// Emits structured tracing events (target="nc.runtime.apply") with fields: cid, decision_tag,
    /// dry_run, idem_key_present, state transitions (apply.start, apply.validate.ok/err,
    /// apply.idem.{reserved,conflict,compensate}, apply.dispatch.{repartition,reschedule,throttle,err},
    /// and apply.{ok,err} including elapsed_ms and error_kind on failure). Disabled entirely when
    /// the feature is off.
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

        // Structured telemetry (feature-gated)
        #[cfg(feature = "telemetry")]
        let __start = std::time::Instant::now();
        #[cfg(feature = "telemetry")]
        let __decision_tag: &'static str = decision_tag(decision);
        #[cfg(feature = "telemetry")]
        let __cid: String = match &opts.idempotency_key {
            Some(k) => format!("key:{}", k),
            None => {
                let tid = format!("{:?}", std::thread::current().id());
                format!("adhoc:{}", tid)
            }
        };
        #[cfg(feature = "telemetry")]
        {
            event!(target: "nc.runtime.apply", Level::INFO,
                event = "apply.start",
                decision_tag = __decision_tag,
                dry_run = opts.dry_run,
                cid = %__cid,
                idem_key_present = opts.idempotency_key.is_some()
            );
        }

        // Validate decision-specific parameters first.
        if let Decision::Throttle { percent } = decision {
            if *percent == 0 || *percent > 100 {
                #[cfg(feature = "telemetry")]
                {
                    event!(target: "nc.runtime.apply", Level::WARN,
                        event = "apply.validate.err",
                        decision_tag = __decision_tag,
                        cid = %__cid,
                        error = "invalid_state"
                    );
                }
                return Err(RuntimeError::InvalidState(format!(
                    "throttle percent must be 1..=100; got {}",
                    percent
                )));
            }
        }

        // Validation success
        #[cfg(feature = "telemetry")]
        {
            event!(target: "nc.runtime.apply", Level::DEBUG,
                event = "apply.validate.ok",
                decision_tag = __decision_tag,
                cid = %__cid
            );
        }
        // Dry-run: no side effects beyond optional telemetry above, and no idempotency recording.
        if opts.dry_run {
            #[cfg(feature = "telemetry")]
            {
                let elapsed_ms = __start.elapsed().as_millis() as u64;
                event!(target: "nc.runtime.apply", Level::INFO,
                    event = "apply.ok",
                    decision_tag = __decision_tag,
                    cid = %__cid,
                    elapsed_ms = elapsed_ms
                );
            }
            return Ok(());
        }

        // Idempotency reservation only for mutating decisions.
        let is_mutating = matches!(
            decision,
            Decision::Repartition { .. } | Decision::Reschedule { .. } | Decision::Throttle { .. }
        );
        let mut reserved_key: Option<String> = None;
        if is_mutating {
            if let Some(k) = &opts.idempotency_key {
                if !idempotency_seen_insert(k) {
                    #[cfg(feature = "telemetry")]
                    {
                        event!(target: "nc.runtime.apply", Level::INFO,
                            event = "apply.idem.conflict",
                            decision_tag = __decision_tag,
                            cid = %__cid
                        );
                    }
                    return Err(RuntimeError::IdempotencyConflict(k.clone()));
                }
                reserved_key = Some(k.clone());
                #[cfg(feature = "telemetry")]
                {
                    event!(target: "nc.runtime.apply", Level::INFO,
                        event = "apply.idem.reserved",
                        decision_tag = __decision_tag,
                        cid = %__cid
                    );
                }
            }
        }

        // Execute the decision path.
        let res = match decision {
            Decision::NoChange => Ok(()),
            Decision::Repartition { .. } => {
                #[cfg(feature = "telemetry")]
                {
                    #[cfg(feature = "orchestrator")]
                    let feature_enabled = true;
                    #[cfg(not(feature = "orchestrator"))]
                    let feature_enabled = false;

                    event!(target: "nc.runtime.apply", Level::INFO,
                        event = "apply.dispatch.repartition",
                        cid = %__cid,
                        feature_enabled = feature_enabled
                    );
                }
                orchestrator_shim::repartition_minimal()
            }
            Decision::Reschedule { .. } => {
                #[cfg(feature = "telemetry")]
                {
                    #[cfg(feature = "hal-shims")]
                    {
                        let feature_enabled = true;
                        #[cfg(test)]
                        let hooks_registered = crate::TEST_HAL_HOOKS
                            .lock()
                            .expect("TEST_HAL_HOOKS mutex poisoned")
                            .is_some();
                        #[cfg(not(test))]
                        let hooks_registered = crate::HAL_HOOKS.get().is_some();

                        event!(target: "nc.runtime.apply", Level::INFO,
                            event = "apply.dispatch.reschedule",
                            cid = %__cid,
                            feature_enabled = feature_enabled,
                            hooks_registered = hooks_registered
                        );
                    }
                    #[cfg(not(feature = "hal-shims"))]
                    {
                        let feature_enabled = false;
                        event!(target: "nc.runtime.apply", Level::INFO,
                            event = "apply.dispatch.reschedule",
                            cid = %__cid,
                            feature_enabled = feature_enabled
                        );
                    }
                }
                #[cfg(feature = "hal-shims")]
                {
                    #[cfg(test)]
                    {
                        let g = crate::TEST_HAL_HOOKS.lock().expect("TEST_HAL_HOOKS mutex poisoned");
                        if let Some(h) = g.as_ref() {
                            (h.reschedule)()
                        } else {
                            Err(RuntimeError::IntegrationUnavailable("HAL hooks not registered".into()))
                        }
                    }
                    #[cfg(not(test))]
                    {
                        if let Some(h) = crate::HAL_HOOKS.get() {
                            (h.reschedule)()
                        } else {
                            Err(RuntimeError::IntegrationUnavailable("HAL hooks not registered".into()))
                        }
                    }
                }
                #[cfg(not(feature = "hal-shims"))]
                {
                    Err(RuntimeError::NotSupported(
                        "reschedule path not yet available".to_string(),
                    ))
                }
            },
            Decision::Throttle { percent } => {
                let p = *percent;
                #[cfg(feature = "telemetry")]
                {
                    #[cfg(feature = "hal-shims")]
                    {
                        let feature_enabled = true;
                        #[cfg(test)]
                        let hooks_registered = crate::TEST_HAL_HOOKS
                            .lock()
                            .expect("TEST_HAL_HOOKS mutex poisoned")
                            .is_some();
                        #[cfg(not(test))]
                        let hooks_registered = crate::HAL_HOOKS.get().is_some();

                        event!(target: "nc.runtime.apply", Level::INFO,
                            event = "apply.dispatch.throttle",
                            cid = %__cid,
                            feature_enabled = feature_enabled,
                            hooks_registered = hooks_registered,
                            percent = p as u64
                        );
                    }
                    #[cfg(not(feature = "hal-shims"))]
                    {
                        let feature_enabled = false;
                        event!(target: "nc.runtime.apply", Level::INFO,
                            event = "apply.dispatch.throttle",
                            cid = %__cid,
                            feature_enabled = feature_enabled,
                            percent = p as u64
                        );
                    }
                }
                #[cfg(feature = "hal-shims")]
                {
                    #[cfg(test)]
                    {
                        let g = crate::TEST_HAL_HOOKS.lock().expect("TEST_HAL_HOOKS mutex poisoned");
                        if let Some(h) = g.as_ref() {
                            (h.throttle)(p)
                        } else {
                            Err(RuntimeError::IntegrationUnavailable("HAL hooks not registered".into()))
                        }
                    }
                    #[cfg(not(test))]
                    {
                        if let Some(h) = crate::HAL_HOOKS.get() {
                            (h.throttle)(p)
                        } else {
                            Err(RuntimeError::IntegrationUnavailable("HAL hooks not registered".into()))
                        }
                    }
                }
                #[cfg(not(feature = "hal-shims"))]
                {
                    Err(RuntimeError::NotSupported(
                        "throttle path not yet available".to_string(),
                    ))
                }
            },
        };

        // Dispatch error classification (integration unavailable / not supported)
        #[cfg(feature = "telemetry")]
        {
            if let Err(err) = &res {
                let ek = error_kind(err);
                if ek == "integration_unavailable" || ek == "not_supported" {
                    event!(target: "nc.runtime.apply", Level::WARN,
                        event = "apply.dispatch.err",
                        decision_tag = __decision_tag,
                        cid = %__cid,
                        error = ek
                    );
                }
            }
        }

        // Compensation: if we reserved the key in this call and the apply failed, remove it.
        if res.is_err() {
            if let Some(k) = reserved_key.as_deref() {
                #[cfg(feature = "telemetry")]
                {
                    event!(target: "nc.runtime.apply", Level::INFO,
                        event = "apply.idem.compensate",
                        decision_tag = __decision_tag,
                        cid = %__cid
                    );
                }
                idempotency_remove(k);
            }
        }

        // Finalization
        #[cfg(feature = "telemetry")]
        {
            let elapsed_ms = __start.elapsed().as_millis() as u64;
            match &res {
                Ok(()) => {
                    event!(target: "nc.runtime.apply", Level::INFO,
                        event = "apply.ok",
                        decision_tag = __decision_tag,
                        cid = %__cid,
                        elapsed_ms = elapsed_ms
                    );
                }
                Err(err) => {
                    event!(target: "nc.runtime.apply", Level::ERROR,
                        event = "apply.err",
                        decision_tag = __decision_tag,
                        cid = %__cid,
                        error_kind = error_kind(err),
                        elapsed_ms = elapsed_ms
                    );
                }
            }
        }

        res
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
            // default options should be ok
            apply_with_options(&Decision::NoChange, &ApplyOptions::default()).expect("default no-change ok");

            let opts = ApplyOptions { idempotency_key: Some("k-noop".into()), dry_run: false };
            apply_with_options(&Decision::NoChange, &opts).expect("no-change ok");

            // Using the same key for a mutating decision must NOT conflict because NoChange
            // must not have mutated the idempotency registry.
            let res = apply_with_options(&Decision::Repartition { hint: None }, &opts);
            if let Err(RuntimeError::IdempotencyConflict(k)) = &res {
                panic!("NoChange mutated idempotency registry; conflict on key {}", k);
            }
        }

        #[cfg(not(feature = "hal-shims"))]
        #[test]
        fn idempotency_key_removed_on_error() {
            // First attempt with a mutating decision should record idempotency and then fail,
            // and the compensation path should remove the key.
            let opts = ApplyOptions { idempotency_key: Some("k-remove".into()), dry_run: false };
            let r1 = apply_with_options(&Decision::Throttle { percent: 10 }, &opts);
            assert!(matches!(r1, Err(RuntimeError::NotSupported(_))));
            // Immediate retry with the same key should NOT hit IdempotencyConflict; we should
            // see the underlying error again because the key was rolled back.
            let r2 = apply_with_options(&Decision::Throttle { percent: 10 }, &opts);
            assert!(matches!(r2, Err(RuntimeError::NotSupported(_))));
        }

        #[cfg(not(feature = "orchestrator"))]
        #[test]
        fn not_supported_by_default() {
            // NotSupported with a message suggesting enabling the 'orchestrator' feature.
            let opts = ApplyOptions { idempotency_key: Some("repart-ns-1".into()), dry_run: false };
            let res1 = apply_with_options(&Decision::Repartition { hint: None }, &opts);
            match res1 {
                Err(RuntimeError::NotSupported(msg)) => {
                    assert!(
                        msg.contains("repartition") && msg.contains("orchestrator"),
                        "msg={}", msg
                    );
                }
                other => panic!("expected NotSupported, got {:?}", other),
            }
            // Immediate retry should not conflict; compensation must have removed the reservation.
            let res2 = apply_with_options(&Decision::Repartition { hint: None }, &opts);
            assert!(matches!(res2, Err(RuntimeError::NotSupported(_))));
        }

        #[test]
        fn telemetry_compiles_off_repartition_dry_run_ok() {
            // Should compile and run with telemetry disabled; dry_run avoids side effects.
            let res = apply_with_options(
                &Decision::Repartition { hint: None },
                &ApplyOptions { idempotency_key: Some("repart-dry".into()), dry_run: true },
            );
            assert!(res.is_ok());
        }

        #[cfg(feature = "orchestrator")]
        #[test]
        fn feature_on_smoke() {
            let opts = ApplyOptions { idempotency_key: Some("repart-ok".into()), dry_run: false };
            let res = apply_with_options(&Decision::Repartition { hint: Some("test".into()) }, &opts);
            // Expect a non-NotSupported result when orchestrator feature is on.
            match res {
                Ok(()) => {}
                Err(RuntimeError::IntegrationUnavailable(msg)) => {
                    assert!(
                        msg.contains("planning succeeded") || msg.contains("apply path"),
                        "unexpected message: {}", msg
                    );
                }
                other => panic!("expected Ok or IntegrationUnavailable, got {:?}", other),
            }
        }

        #[cfg(not(feature = "hal-shims"))]
        #[test]
        fn reschedule_and_throttle_not_supported() {
            let r = apply_with_options(&Decision::Reschedule { hint: None }, &ApplyOptions::default());
            assert!(matches!(r, Err(RuntimeError::NotSupported(_))));
            let t = apply_with_options(&Decision::Throttle { percent: 50 }, &ApplyOptions::default());
            assert!(matches!(t, Err(RuntimeError::NotSupported(_))));
            // Out-of-range percent should be rejected with InvalidState.
            let invalid = apply_with_options(&Decision::Throttle { percent: 0 }, &ApplyOptions::default());
            assert!(matches!(invalid, Err(RuntimeError::InvalidState(_))));
        }

        #[test]
        fn idempotency_conflict_on_race_smoke() {
            // Simulate an external reservation that races with this call.
            idempotency_remove("race-smoke-1"); // ensure clean slate for this key
            assert!(idempotency_seen_insert("race-smoke-1"), "precondition: key must be new");
            let opts = ApplyOptions { idempotency_key: Some("race-smoke-1".into()), dry_run: false };
            let res = apply_with_options(&Decision::Throttle { percent: 10 }, &opts);
            match res {
                Err(RuntimeError::IdempotencyConflict(k)) => assert_eq!(k, "race-smoke-1"),
                other => panic!("expected IdempotencyConflict, got {:?}", other),
            }
            // cleanup
            idempotency_remove("race-smoke-1");
        }
        #[test]
        fn concurrent_preinserted_key_all_conflict() {
            // Ensure a clean slate and pre-insert a key to simulate an existing reservation.
            let key = "conc-preinserted-all-conflict";
            idempotency_remove(key);
            assert!(idempotency_seen_insert(key), "precondition: key must be newly inserted now");

            // Spawn N threads that will all attempt to apply a mutating decision with the same key.
            let mut handles = Vec::new();
            for _ in 0..4 {
                let k = key.to_string();
                handles.push(std::thread::spawn(move || {
                    let opts = ApplyOptions { idempotency_key: Some(k), dry_run: false };
                    apply_with_options(&Decision::Throttle { percent: 10 }, &opts)
                }));
            }

            // All threads should observe Err(IdempotencyConflict(..)) deterministically.
            for h in handles {
                let res = h.join().expect("thread join ok");
                match res {
                    Err(RuntimeError::IdempotencyConflict(_)) => {}
                    other => panic!("expected IdempotencyConflict for all threads, got {:?}", other),
                }
            }

            // Cleanup
            idempotency_remove(key);
        }

        #[cfg(not(feature = "hal-shims"))]
        #[test]
        fn concurrent_two_threads_conflict_and_compensation_final_state_absent() {
            use std::sync::{Arc, Barrier};
    
            // Fresh key; ensure absence and do NOT pre-insert.
            let key = "conc-two-threads-comp-absent".to_string();
            idempotency_remove(&key);
    
            // Coordinate both threads to start simultaneously (plus main for release).
            let barrier = Arc::new(Barrier::new(3));
    
            let mut handles = Vec::new();
            for _ in 0..2 {
                let b = Arc::clone(&barrier);
                let k = key.clone();
                handles.push(std::thread::spawn(move || {
                    // Release both runners at the same time.
                    b.wait();
                    let opts = ApplyOptions { idempotency_key: Some(k.clone()), dry_run: false };
                    apply_with_options(&Decision::Throttle { percent: 10 }, &opts)
                }));
            }
    
            // Release both threads.
            barrier.wait();
    
            let results: Vec<_> = handles
                .into_iter()
                .map(|h| h.join().expect("thread join"))
                .collect();
    
            // All outcomes are errors (either NotSupported or IdempotencyConflict).
            assert!(results.iter().all(|r| r.is_err()), "all outcomes should be errors");
    
            let idem_conflicts =
                results.iter().filter(|r| matches!(r, Err(RuntimeError::IdempotencyConflict(_)))).count();
            let not_supported =
                results.iter().filter(|r| matches!(r, Err(RuntimeError::NotSupported(_)))).count();
    
            // Deterministic, race-tolerant assertion:
            // - At least one IdempotencyConflict OR both NotSupported (if compensation removed before the second reserve).
            assert!(
                idem_conflicts >= 1 || not_supported == 2,
                "expected at least one IdempotencyConflict or both NotSupported; conflicts={}, not_supported={}, results={:?}",
                idem_conflicts, not_supported, results
            );
    
            // Final registry state should be clean due to compensation.
            assert!(idempotency_seen_insert(&key), "key should be absent after compensation");
            idempotency_remove(&key);
        }
    }
}

#[cfg(test)]
mod hal_shims_tests {
    use crate::adaptive::{apply_with_options, ApplyOptions, Decision, RuntimeError};

    // Feature off: preserve NotSupported behavior for Reschedule/Throttle.
    #[cfg(not(feature = "hal-shims"))]
    #[test]
    fn hal_shims_feature_off_keeps_notsupported_behavior() {
        let r = apply_with_options(&Decision::Reschedule { hint: None }, &ApplyOptions::default());
        assert!(matches!(r, Err(RuntimeError::NotSupported(_))));
        let t = apply_with_options(&Decision::Throttle { percent: 10 }, &ApplyOptions::default());
        assert!(matches!(t, Err(RuntimeError::NotSupported(_))));
    }

    // Feature on tests
    #[cfg(feature = "hal-shims")]
    mod feature_on {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

        static RESCHEDULE_COUNT: AtomicUsize = AtomicUsize::new(0);
        static THROTTLE_COUNT: AtomicUsize = AtomicUsize::new(0);

        fn reschedule_hook() -> crate::adaptive::Result<()> {
            RESCHEDULE_COUNT.fetch_add(1, SeqCst);
            Ok(())
        }
        fn throttle_hook(percent: u8) -> crate::adaptive::Result<()> {
            assert!((1..=100).contains(&percent));
            THROTTLE_COUNT.fetch_add(1, SeqCst);
            Ok(())
        }

        #[test]
        fn hal_shims_feature_on_no_registration_integration_unavailable() {
            crate::__test_clear_hal_hooks();
            let r = apply_with_options(&Decision::Reschedule { hint: None }, &ApplyOptions::default());
            match r {
                Err(RuntimeError::IntegrationUnavailable(msg)) => {
                    assert!(msg.contains("HAL hooks not registered"));
                }
                other => panic!("expected IntegrationUnavailable, got {:?}", other),
            }
            let t = apply_with_options(&Decision::Throttle { percent: 10 }, &ApplyOptions::default());
            match t {
                Err(RuntimeError::IntegrationUnavailable(msg)) => {
                    assert!(msg.contains("HAL hooks not registered"));
                }
                other => panic!("expected IntegrationUnavailable, got {:?}", other),
            }
        }

        #[test]
        fn hal_shims_registration_allows_calls_and_idempotency() {
            crate::__test_clear_hal_hooks();
            RESCHEDULE_COUNT.store(0, SeqCst);
            THROTTLE_COUNT.store(0, SeqCst);

            let hooks = crate::HalHooks { reschedule: reschedule_hook, throttle: throttle_hook };
            assert!(crate::register_hal_hooks(hooks), "hooks newly set");

            // Reschedule ok and increments
            apply_with_options(&Decision::Reschedule { hint: None }, &ApplyOptions::default()).expect("reschedule ok");
            assert_eq!(RESCHEDULE_COUNT.load(SeqCst), 1);

            // Throttle with idempotency: first OK, second is IdempotencyConflict
            let opts = ApplyOptions { idempotency_key: Some("ok-key-hal-test".into()), dry_run: false };
            apply_with_options(&Decision::Throttle { percent: 10 }, &opts).expect("throttle ok");
            assert_eq!(THROTTLE_COUNT.load(SeqCst), 1);

            let second = apply_with_options(&Decision::Throttle { percent: 10 }, &opts);
            match second {
                Err(RuntimeError::IdempotencyConflict(k)) => assert_eq!(k, "ok-key-hal-test"),
                other => panic!("expected IdempotencyConflict, got {:?}", other),
            }
            // Ensure count didn't increase on second call
            assert_eq!(THROTTLE_COUNT.load(SeqCst), 1);
        }
    }
}

#[cfg(test)]
mod adaptive_table_tests {
    use crate::adaptive::{self, apply_with_options, ApplyOptions, Decision, RuntimeError};

    #[derive(Clone)]
    struct Scenario<'a> {
        name: &'a str,
        decision: Decision,
        opts: ApplyOptions,
        expect: Expect,
    }

    #[derive(Debug, Clone)]
    enum Expect {
        Ok,
        Err(ErrVariant),
    }

    #[derive(Debug, Clone)]
    enum ErrVariant {
        InvalidState,
        NotSupported,
        IntegrationUnavailable,
        IdempotencyConflict,
        ApplyFailed,
        RollbackFailed,
        ConcurrencyConflict,
    }

    fn assert_outcome(row: &Scenario, res: &adaptive::Result<()>) {
        match (&row.expect, res) {
            (Expect::Ok, Ok(())) => {}
            (Expect::Ok, other) => panic!("{}: expected Ok(()), got {:?}", row.name, other),

            (Expect::Err(ErrVariant::InvalidState), Err(RuntimeError::InvalidState(_))) => {}
            (Expect::Err(ErrVariant::NotSupported), Err(RuntimeError::NotSupported(_))) => {}
            (Expect::Err(ErrVariant::IntegrationUnavailable), Err(RuntimeError::IntegrationUnavailable(_))) => {}
            (Expect::Err(ErrVariant::IdempotencyConflict), Err(RuntimeError::IdempotencyConflict(_))) => {}
            (Expect::Err(ErrVariant::ApplyFailed), Err(RuntimeError::ApplyFailed(_))) => {}
            (Expect::Err(ErrVariant::RollbackFailed), Err(RuntimeError::RollbackFailed(_))) => {}
            (Expect::Err(ErrVariant::ConcurrencyConflict), Err(RuntimeError::ConcurrencyConflict(_))) => {}
            (Expect::Err(_), other) => panic!("{}: expected {:?}, got {:?}", row.name, row.expect, other),
        }
    }

    fn run(rows: &[Scenario]) {
        for r in rows {
            let res = apply_with_options(&r.decision, &r.opts);
            assert_outcome(r, &res);
        }
    }

    // 1) table_dry_run_success_matrix
    // - All supported decisions validate and succeed under dry_run without recording idempotency.
    // - If an idempotency_key was provided, a subsequent normal call with the same key must NOT
    //   conflict solely due to the dry-run.
    #[test]
    fn table_dry_run_success_matrix() {
        let rows = vec![
            Scenario {
                name: "noop_dry_no_key",
                decision: Decision::NoChange,
                opts: ApplyOptions { idempotency_key: None, dry_run: true },
                expect: Expect::Ok,
            },
            Scenario {
                name: "noop_dry_with_key",
                decision: Decision::NoChange,
                opts: ApplyOptions { idempotency_key: Some("k1".into()), dry_run: true },
                expect: Expect::Ok,
            },
            Scenario {
                name: "reschedule_dry_no_key",
                decision: Decision::Reschedule { hint: None },
                opts: ApplyOptions { idempotency_key: None, dry_run: true },
                expect: Expect::Ok,
            },
            Scenario {
                name: "reschedule_dry_with_key",
                decision: Decision::Reschedule { hint: None },
                opts: ApplyOptions { idempotency_key: Some("k2".into()), dry_run: true },
                expect: Expect::Ok,
            },
            Scenario {
                name: "throttle_dry_no_key",
                decision: Decision::Throttle { percent: 10 },
                opts: ApplyOptions { idempotency_key: None, dry_run: true },
                expect: Expect::Ok,
            },
            Scenario {
                name: "throttle_dry_with_key",
                decision: Decision::Throttle { percent: 10 },
                opts: ApplyOptions { idempotency_key: Some("k3".into()), dry_run: true },
                expect: Expect::Ok,
            },
        ];

        run(&rows);

        // Subsequent normal call must NOT conflict due to the earlier dry-run.
        for r in rows.into_iter().filter(|s| s.opts.idempotency_key.is_some()) {
            let mut opts2 = r.opts.clone();
            opts2.dry_run = false;
            let res2 = apply_with_options(&r.decision, &opts2);
            if let Err(RuntimeError::IdempotencyConflict(k)) = res2 {
                panic!(
                    "{}: dry-run incorrectly recorded idempotency key {}; conflict observed",
                    r.name, k
                );
            }
        }
    }

    // 2) table_invalid_input_validation
    // - Invalid Throttle inputs are rejected with InvalidState in both dry-run and normal modes,
    //   before idempotency.
    #[test]
    fn table_invalid_input_validation() {
        let rows = vec![
            Scenario {
                name: "throttle_percent_0_normal",
                decision: Decision::Throttle { percent: 0 },
                opts: ApplyOptions { idempotency_key: None, dry_run: false },
                expect: Expect::Err(ErrVariant::InvalidState),
            },
            Scenario {
                name: "throttle_percent_0_dry",
                decision: Decision::Throttle { percent: 0 },
                opts: ApplyOptions { idempotency_key: None, dry_run: true },
                expect: Expect::Err(ErrVariant::InvalidState),
            },
            Scenario {
                name: "throttle_percent_101_normal",
                decision: Decision::Throttle { percent: 101 },
                opts: ApplyOptions { idempotency_key: None, dry_run: false },
                expect: Expect::Err(ErrVariant::InvalidState),
            },
            Scenario {
                name: "throttle_percent_101_dry",
                decision: Decision::Throttle { percent: 101 },
                opts: ApplyOptions { idempotency_key: None, dry_run: true },
                expect: Expect::Err(ErrVariant::InvalidState),
            },
        ];

        run(&rows);
    }

    // 3) table_default_features_notsupported_and_compensation
    // - With hal-shims disabled, Reschedule/Throttle return NotSupported and compensation removes
    //   idempotency key so immediate retry with the same key returns the same NotSupported (not a conflict).
    #[cfg(not(feature = "hal-shims"))]
    #[test]
    fn table_default_features_notsupported_and_compensation() {
        let rows = vec![
            Scenario {
                name: "reschedule_notsupported_compensation",
                decision: Decision::Reschedule { hint: None },
                opts: ApplyOptions { idempotency_key: Some("ns-1".into()), dry_run: false },
                expect: Expect::Err(ErrVariant::NotSupported),
            },
            Scenario {
                name: "throttle_notsupported_compensation",
                decision: Decision::Throttle { percent: 10 },
                opts: ApplyOptions { idempotency_key: Some("ns-2".into()), dry_run: false },
                expect: Expect::Err(ErrVariant::NotSupported),
            },
        ];

        // First attempt: NotSupported
        run(&rows);

        // Immediate retry: still NotSupported (NOT IdempotencyConflict), proving compensation removed the key.
        for r in rows {
            let res2 = apply_with_options(&r.decision, &r.opts);
            match res2 {
                Err(RuntimeError::NotSupported(_)) => {}
                Err(RuntimeError::IdempotencyConflict(k)) => {
                    panic!("{}: unexpected IdempotencyConflict for key {}", r.name, k)
                }
                other => panic!("{}: expected NotSupported on retry, got {:?}", r.name, other),
            }
        }
    }
}

#[cfg(all(test, feature = "hal-shims"))]
mod hal_shims_table_tests {
    use crate::adaptive::{self, apply_with_options, ApplyOptions, Decision, RuntimeError};
    use crate::{register_hal_hooks, HalHooks, __test_clear_hal_hooks};
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    static TBL_RESCHEDULE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static TBL_THROTTLE_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn reschedule_hook() -> adaptive::Result<()> {
        TBL_RESCHEDULE_COUNT.fetch_add(1, SeqCst);
        Ok(())
    }
    fn throttle_hook(percent: u8) -> adaptive::Result<()> {
        assert!((1..=100).contains(&percent));
        TBL_THROTTLE_COUNT.fetch_add(1, SeqCst);
        Ok(())
    }

    #[derive(Clone)]
    struct Scenario<'a> {
        name: &'a str,
        decision: Decision,
        opts: ApplyOptions,
    }

    // 4) table_hal_shims_success_and_idempotency (only when feature enabled)
    // - Register hooks that succeed and verify Ok + counter increments, and idempotency behavior.
    #[test]
    fn table_hal_shims_success_and_idempotency() {
        __test_clear_hal_hooks();
        TBL_RESCHEDULE_COUNT.store(0, SeqCst);
        TBL_THROTTLE_COUNT.store(0, SeqCst);

        assert!(
            register_hal_hooks(HalHooks { reschedule: reschedule_hook, throttle: throttle_hook }),
            "test hooks should be newly registered"
        );

        let rows = vec![
            Scenario {
                name: "reschedule_no_key_first_and_second_ok",
                decision: Decision::Reschedule { hint: None },
                opts: ApplyOptions { idempotency_key: None, dry_run: false },
            },
            Scenario {
                name: "reschedule_with_key_idempotent_conflict_on_second",
                decision: Decision::Reschedule { hint: None },
                opts: ApplyOptions { idempotency_key: Some("ok-key-hal-a".into()), dry_run: false },
            },
            Scenario {
                name: "throttle_no_key_first_and_second_ok",
                decision: Decision::Throttle { percent: 10 },
                opts: ApplyOptions { idempotency_key: None, dry_run: false },
            },
            Scenario {
                name: "throttle_with_key_idempotent_conflict_on_second",
                decision: Decision::Throttle { percent: 10 },
                opts: ApplyOptions { idempotency_key: Some("ok-key-hal-b".into()), dry_run: false },
            },
        ];

        for s in rows {
            // First call
            let r_pre = TBL_RESCHEDULE_COUNT.load(SeqCst);
            let t_pre = TBL_THROTTLE_COUNT.load(SeqCst);
            let res1 = apply_with_options(&s.decision, &s.opts);
            assert!(res1.is_ok(), "{}: first call expected Ok(()), got {:?}", s.name, res1);

            match s.decision {
                Decision::Reschedule { .. } => {
                    assert_eq!(
                        TBL_RESCHEDULE_COUNT.load(SeqCst),
                        r_pre + 1,
                        "{}: reschedule hook should increment on first call",
                        s.name
                    );
                    // Second call
                    let res2 = apply_with_options(&s.decision, &s.opts);
                    match &s.opts.idempotency_key {
                        None => {
                            assert!(res2.is_ok(), "{}: second call without key should be Ok(())", s.name);
                            assert_eq!(
                                TBL_RESCHEDULE_COUNT.load(SeqCst),
                                r_pre + 2,
                                "{}: reschedule counter should increment twice when no key",
                                s.name
                            );
                        }
                        Some(_) => {
                            match res2 {
                                Err(RuntimeError::IdempotencyConflict(_)) => {}
                                other => panic!(
                                    "{}: expected IdempotencyConflict on second call with key, got {:?}",
                                    s.name, other
                                ),
                            }
                            assert_eq!(
                                TBL_RESCHEDULE_COUNT.load(SeqCst),
                                r_pre + 1,
                                "{}: reschedule counter must NOT increment on idempotent replay",
                                s.name
                            );
                        }
                    }
                }
                Decision::Throttle { .. } => {
                    assert_eq!(
                        TBL_THROTTLE_COUNT.load(SeqCst),
                        t_pre + 1,
                        "{}: throttle hook should increment on first call",
                        s.name
                    );
                    // Second call
                    let res2 = apply_with_options(&s.decision, &s.opts);
                    match &s.opts.idempotency_key {
                        None => {
                            assert!(res2.is_ok(), "{}: second call without key should be Ok(())", s.name);
                            assert_eq!(
                                TBL_THROTTLE_COUNT.load(SeqCst),
                                t_pre + 2,
                                "{}: throttle counter should increment twice when no key",
                                s.name
                            );
                        }
                        Some(_) => {
                            match res2 {
                                Err(RuntimeError::IdempotencyConflict(_)) => {}
                                other => panic!(
                                    "{}: expected IdempotencyConflict on second call with key, got {:?}",
                                    s.name, other
                                ),
                            }
                            assert_eq!(
                                TBL_THROTTLE_COUNT.load(SeqCst),
                                t_pre + 1,
                                "{}: throttle counter must NOT increment on idempotent replay",
                                s.name
                            );
                        }
                    }
                }
                _ => unreachable!("Only Reschedule and Throttle are covered in this table"),
            }
        }
    }
}

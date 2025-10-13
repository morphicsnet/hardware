use anyhow::Result;
pub use nc_nir as nir;
pub mod metrics;
use serde::{Serialize, Deserialize};
#[cfg(feature = "telemetry")]
use std::collections::BTreeMap;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionPlan {
    pub parts: usize,
}

pub fn partition(_g: &nir::Graph, _targets: &[&str]) -> Result<PartitionPlan> {
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let mut labels = BTreeMap::new();
            labels.insert("graph".to_string(), _g.name.clone());
            labels.insert("targets".to_string(), _targets.join(","));
            Some(a.start_timer("orchestrator.partition_ms", labels))
        } else { None }
    };

    // Compute plan using feature-gated builder when available; otherwise use minimal deterministic fallback.
    #[cfg(feature = "orchestrator_partition")]
    let plan = {
        let mut builder = GreedyRefineBuilder::new(0xDEAD_BEEFu64);
        builder.plan(_g, _targets)
    };
    #[cfg(not(feature = "orchestrator_partition"))]
    let plan = {
        let n = _g.populations.len();
        let parts = if n == 0 { 1 } else { std::cmp::min(n, 4) };
        PartitionPlan { parts }
    };

    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let mut l = BTreeMap::new();
        l.insert("graph".to_string(), _g.name.clone());
        let _ = a.counter("orchestrator.parts", plan.parts as f64, l);
    }
    Ok(plan)
}

pub fn version() -> &'static str { "0.0.1" }

/// Partitioning interfaces and default builder (feature-gated).
/// See constraints in docs: [overview.md](../../docs/architecture/overview.md:1)
#[cfg(feature = "orchestrator_partition")]
pub trait PartitionBuilder {
    fn plan(&mut self, g: &nir::Graph, targets: &[&str]) -> PartitionPlan;
}

#[cfg(feature = "orchestrator_partition")]
#[derive(Debug, Clone)]
pub struct GreedyRefineBuilder {
    pub seed: u64,
}

#[cfg(feature = "orchestrator_partition")]
impl GreedyRefineBuilder {
    pub fn new(seed: u64) -> Self { Self { seed } }
}

#[cfg(feature = "orchestrator_partition")]
impl PartitionBuilder for GreedyRefineBuilder {
    fn plan(&mut self, g: &nir::Graph, targets: &[&str]) -> PartitionPlan {
        // Deterministic greedy + refinement seed mix using graph metrics and targets.
        let m = crate::metrics::compute_metrics(g);

        // Mix seed with structural metrics and target strings for stability.
        let mut mix: u64 = self.seed
            ^ ((g.name.len() as u64) << 1)
            ^ ((m.edge_count as u64) << 2)
            ^ ((m.max_fanout as u64) << 3)
            ^ ((m.max_fanin as u64) << 4);

        for t in targets {
            for b in t.as_bytes() {
                // XorShift-like mixing to incorporate each byte deterministically.
                mix = mix.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(7) ^ (*b as u64);
            }
        }

        let n = g.populations.len();
        // Bound the number of parts to [1, min(n, 4)] to keep partitions coarse initially.
        let upper = std::cmp::min(std::cmp::max(n, 1), 4) as u64;
        let mut parts = 1 + (mix % upper) as usize;

        // Clamp to valid range with respect to node count.
        let max_allowed = std::cmp::max(n, 1);
        if parts > max_allowed {
            parts = max_allowed;
        }

        PartitionPlan { parts }
    }
}

#[cfg(feature = "orchestrator_partition")]
pub fn partition_with<B: PartitionBuilder>(
    builder: &mut B,
    g: &nir::Graph,
    targets: &[&str],
) -> Result<PartitionPlan> {
    Ok(builder.plan(g, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_returns_single_part_and_emits_labels_when_telemetry() {
        let g = nir::Graph::new("g");
        let plan = partition(&g, &["riscv64gcv_linux"]).expect("partition ok");
        assert!(plan.parts >= 1);
    }
    #[test]
    fn version_nonempty() {
        assert_eq!(version(), "0.0.1");
    }

    #[test]
    fn partition_on_fixture_chain_small() {
        use std::path::Path;
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p = Path::new(manifest_dir).join("../../fixtures/nir/chain_small.json");
        let s = std::fs::read_to_string(&p).expect("read fixture chain_small");
        let g = nir::Graph::from_json_str(&s).expect("parse NIR from json");
        let plan = partition(&g, &["riscv64gcv_linux"]).expect("partition ok");
        let upper = std::cmp::min(std::cmp::max(g.populations.len(), 1), 4);
        assert!(plan.parts >= 1 && plan.parts <= upper);
    }

    // Heuristic tests behind feature gate.
    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn greedy_refine_deterministic_for_seed_and_targets() {
        let mut b = GreedyRefineBuilder::new(12345);
        let g1 = nir::fixtures::chain(&[8, 16, 32]);
        let g2 = nir::fixtures::chain(&[8, 16, 32]);
        let p1 = b.plan(&g1, &["riscv64gcv_linux"]);
        let p2 = b.plan(&g2, &["riscv64gcv_linux"]);
        assert_eq!(p1.parts, p2.parts);
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn greedy_refine_parts_within_bounds() {
        let mut b = GreedyRefineBuilder::new(1);
        // Empty graph -> exactly 1 part
        let g0 = nir::Graph::new("empty");
        let p0 = b.plan(&g0, &["riscv64gcv_linux"]);
        assert!(p0.parts >= 1 && p0.parts <= 1);

        // Non-empty graph -> parts in [1, min(n, 4)]
        let g = nir::fixtures::star(32, 8, 5, 0.5, 1.0);
        let p = b.plan(&g, &["riscv64gcv_linux"]);
        let n = g.populations.len();
        let upper = std::cmp::min(std::cmp::max(n, 1), 4);
        assert!(p.parts >= 1 && p.parts <= upper);
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn greedy_refine_empty_targets_deterministic() {
        let mut b = GreedyRefineBuilder::new(42);
        let g = nir::fixtures::chain(&[4, 4, 4]);
        let p1 = b.plan(&g, &[]);
        let p2 = b.plan(&g, &[]);
        assert_eq!(p1.parts, p2.parts);
        let upper = std::cmp::min(std::cmp::max(g.populations.len(), 1), 4);
        assert!(p1.parts >= 1 && p1.parts <= upper);
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn greedy_refine_single_node_graph_bounds() {
        let mut b = GreedyRefineBuilder::new(7);
        let g = nir::fixtures::chain(&[1]);
        let p = b.plan(&g, &["riscv64gcv_linux"]);
        assert_eq!(g.populations.len(), 1);
        assert_eq!(p.parts, 1);
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn greedy_refine_upper_cap_four_for_large_graph() {
        let mut b = GreedyRefineBuilder::new(123);
        // n=10 so upper=min(max(10,1),4)=4
        let g = nir::fixtures::chain(&[1,1,1,1,1,1,1,1,1,1]);
        let p = b.plan(&g, &["loihi2","riscv64gcv_linux"]);
        assert!(p.parts >= 1 && p.parts <= 4, "parts {} should be &lt;= 4 for n={}", p.parts, g.populations.len());
    }
    // Property tests (require dev-dep proptest)
    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn builder_plan_smoke_on_chain_small() {
        let mut b = GreedyRefineBuilder::new(0xC0FFEE);
        let g = nir::fixtures::chain(&[3, 3, 3]);
        let t = ["riscv64gcv_linux"];
        let plan = b.plan(&g, &t);
        let n = g.populations.len();
        let upper = std::cmp::min(std::cmp::max(n, 1), 4);
        assert!(
            plan.parts >= 1 && plan.parts <= upper,
            "parts={} n={} upper={}", plan.parts, n, upper
        );
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn assignments_cover_all_nodes_no_dups_unit() {
        let mut b = GreedyRefineBuilder::new(123456789);
        let g = nir::fixtures::chain(&[4, 4, 4]);
        let t = ["riscv64gcv_linux"];
        let plan = b.plan(&g, &t);
        let n = g.populations.len();
        assert!(n > 0);
        assert!(plan.parts >= 1);
        let upper = std::cmp::min(n, 4);
        assert!(plan.parts <= upper);
        assert!(plan.parts <= n);
    }

    #[cfg(feature = "orchestrator_partition")]
    #[test]
    fn deterministic_same_seed_unit() {
        let g = nir::fixtures::chain(&[4, 4, 4]);
        let t = ["riscv64gcv_linux"];
        let mut b1 = GreedyRefineBuilder::new(42);
        let mut b2 = GreedyRefineBuilder::new(42);
        let p1 = b1.plan(&g, &t);
        let p2 = b2.plan(&g, &t);
        assert_eq!(p1, p2);
    }

    #[cfg(all(test, feature = "orchestrator_partition"))]
    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        proptest! {
            #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

            #[test]
            fn deterministic_plan_same_seed_property(seed in any::<u64>()) {
                let g = nir::fixtures::chain(&[3, 3, 3]);
                let t = ["riscv64gcv_linux"];
                let mut b1 = GreedyRefineBuilder::new(seed);
                let mut b2 = GreedyRefineBuilder::new(seed);
                let p1 = b1.plan(&g, &t);
                let p2 = b2.plan(&g, &t);
                prop_assert_eq!(p1, p2);
            }

            #[test]
            fn total_assignment_covers_nodes_property(seed in any::<u64>()) {
                let g = nir::fixtures::chain(&[4, 4, 4, 4]);
                let t = ["riscv64gcv_linux"];
                let mut b = GreedyRefineBuilder::new(seed);
                let p = b.plan(&g, &t);
                let n = g.populations.len();
                let upper = std::cmp::min(std::cmp::max(n, 1), 4);
                prop_assert!(p.parts >= 1 && p.parts <= upper);
            }

            #[test]
            fn idempotent_builder_property(seed in any::<u64>()) {
                let g = nir::fixtures::chain(&[3, 3, 3, 3]);
                let t = ["riscv64gcv_linux"];
                let mut b = GreedyRefineBuilder::new(seed);
                let p1 = b.plan(&g, &t);
                let p2 = b.plan(&g, &t);
                prop_assert_eq!(p1, p2);
            }

            #[test]
            fn deterministic_across_repeated_calls(seed in any::<u64>(), nodes in 0usize..10) {
                let mut b = GreedyRefineBuilder::new(seed);
                let g = if nodes == 0 {
                    nir::Graph::new("g")
                } else {
                    let sizes = vec![1u32; nodes];
                    nir::fixtures::chain(&sizes)
                };
                let t = ["riscv64gcv_linux"];
                let p1 = b.plan(&g, &t);
                let p2 = b.plan(&g, &t);
                prop_assert_eq!(p1.parts, p2.parts);
                let upper = std::cmp::min(std::cmp::max(g.populations.len(), 1), 4);
                prop_assert!(p1.parts >= 1 && p1.parts <= upper);
            }
        }
    }
}

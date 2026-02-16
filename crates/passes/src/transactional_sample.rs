//! Sample pass demonstrating epoch/transaction semantics and proof-carrying layout safety.
//!
//! This pass uses transactions for reversible optimizations and verifies proofs for layout invariants.

use crate::nir;
use crate::proofs::{Proof, Invariant, ProofSystem};
use anyhow::Result;

/// Sample pass that demonstrates transactions and proofs.
pub struct TransactionalSamplePass;

impl crate::Pass for TransactionalSamplePass {
    fn name(&self) -> &str { "transactional-sample" }

    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        // Begin transaction
        g.begin_transaction();

        // Perform some mutation that can be rolled back
        if g.populations.len() > 0 {
            // Example: Add a dummy population
            let dummy_pop = nir::Population {
                name: "dummy".into(),
                size: 1,
                model: "lif".into(),
                params: serde_json::json!({}),
            };
            g.populations.push(dummy_pop);
        }

        // Embed some proofs
        let proof = Proof::new(
            "sample_layout".to_string(),
            vec![
                Invariant::Bounds { min: 0, max: 100 },
                Invariant::TypeMatch("f32".into()),
                Invariant::OwnershipExclusive,
                Invariant::Sparsity { density: 0.1 },
            ]
        );
        ProofSystem::embed_proofs(&mut g, vec![proof]);

        // Verify proofs
        ProofSystem::verify_graph(&g)?;

        // If all good, commit
        g.commit_transaction().map_err(|e| anyhow::anyhow!("Commit failed: {}", e))?;

        Ok(g)
    }
}
//! TransactionManager for orchestrating reversible pass pipelines.
//!
//! Wraps pass execution with transaction semantics for rollback and concurrency.

use crate::nir;
use nc_passes as passes;
use anyhow::Result;

/// Manages transactions for a graph during pass execution.
/// Supports nesting and deterministic commit/rollback.
pub struct TransactionManager<'a> {
    graph: &'a mut nir::Graph,
}

impl<'a> TransactionManager<'a> {
    pub fn new(graph: &'a mut nir::Graph) -> Self {
        Self { graph }
    }

    /// Execute a pass within a transaction.
    /// Begins transaction, runs pass, and commits if successful, else rolls back.
    pub fn execute_pass(&mut self, pass: &dyn passes::Pass) -> Result<()> {
        self.graph.begin_transaction();
        match pass.run(self.graph.clone()) {
            Ok(new_graph) => {
                *self.graph = new_graph;
                self.graph.commit_transaction().map_err(|e| anyhow::anyhow!("Commit failed: {}", e))
            }
            Err(e) => {
                self.graph.rollback_transaction().map_err(|e| anyhow::anyhow!("Rollback failed: {}", e))?;
                Err(e)
            }
        }
    }

    /// Execute multiple passes sequentially, each in its own transaction.
    pub fn execute_pipeline(&mut self, passes: &[Box<dyn passes::Pass>]) -> Result<()> {
        for pass in passes {
            self.execute_pass(pass.as_ref())?;
        }
        Ok(())
    }

    /// Begin a nested transaction manually.
    pub fn begin_nested(&mut self) -> Result<()> {
        self.graph.begin_transaction();
        Ok(())
    }

    /// Commit the current nested transaction.
    pub fn commit_nested(&mut self) -> Result<()> {
        self.graph.commit_transaction().map_err(|e| anyhow::anyhow!("Commit failed: {}", e))
    }

    /// Rollback the current nested transaction.
    pub fn rollback_nested(&mut self) -> Result<()> {
        self.graph.rollback_transaction().map_err(|e| anyhow::anyhow!("Rollback failed: {}", e))
    }
}
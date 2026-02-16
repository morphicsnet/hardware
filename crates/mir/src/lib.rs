use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use uuid::Uuid;

/// Morphogenesis Intermediate Representation (MIR)
///
/// Implements System 1/2 morphogenesis with three IR levels:
/// - IR-A: Semantic hypergraph (typed manifolds, cognitive structure)
/// - IR-B: Event-driven execution (state machines, routing)
/// - IR-C: Physical configuration (hardware deployment image)

pub mod semantic;
pub mod event_driven;
pub mod physical;
pub mod edit_api;
pub mod validation;

pub use semantic::*;
pub use event_driven::*;
pub use physical::*;
pub use edit_api::*;

/// Compilation target for dual-system morphogenesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTarget {
    /// System 1: Fast inference fabric (event-driven)
    System1Fabric,
    /// System 2: Morphogenesis controller (edit → recompile cycle)
    System2Controller,
}

/// Top-level MIR graph containing all three IR levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirGraph {
    pub id: Uuid,
    pub name: String,
    pub version: String,

    /// IR-A: Semantic hypergraph with manifold constraints
    pub semantic: SemanticGraph,

    /// IR-B: Event-driven execution (compiled from semantic)
    pub event_driven: Option<EventGraph>,

    /// IR-C: Physical configuration (compiled from event-driven)
    pub physical: Option<ConfigImage>,

    /// Version tag for compatibility checking
    pub nir_version: String,
}

impl MirGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            version: "0.1.0".to_string(),
            semantic: SemanticGraph::new(),
            event_driven: None,
            physical: None,
            nir_version: "1.0".to_string(),
        }
    }

    /// Validate semantic constraints before compilation
    pub fn validate(&self) -> Result<()> {
        validation::validate_semantic_graph(&self.semantic)?;
        Ok(())
    }

    /// Compile semantic → event-driven IR
    pub fn compile_to_event_driven(&mut self) -> Result<()> {
        let event_graph = event_driven::compile_semantic_to_event(&self.semantic)?;
        self.event_driven = Some(event_graph);
        Ok(())
    }

    /// Compile event-driven → physical config
    pub fn compile_to_physical(&mut self, target: &str) -> Result<()> {
        let event_graph = self.event_driven.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Event-driven IR not compiled"))?;

        let config_image = physical::compile_event_to_physical(event_graph, target)?;
        self.physical = Some(config_image);
        Ok(())
    }

    /// Apply a morphogenesis edit to the semantic graph
    pub fn apply_edit(&mut self, edit: &GraphEdit) -> Result<()> {
        edit.apply_to(&mut self.semantic)?;

        // Invalidate downstream compilations
        self.event_driven = None;
        self.physical = None;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mir_graph_creation() {
        let graph = MirGraph::new("test_graph");
        assert_eq!(graph.name, "test_graph");
        assert!(graph.semantic.manifolds.is_empty());
    }

    #[test]
    fn mir_graph_validation() {
        let graph = MirGraph::new("test_graph");
        assert!(graph.validate().is_ok());
    }
}
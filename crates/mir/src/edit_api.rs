use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::semantic::*;

/// System 2 Edit API: Graph Morphogenesis Operations
///
/// These edits represent the "epigenesis" actions that System 2 performs:
/// GROW/PRUNE/FREEZE/PROMOTE operations that reshape the cognitive architecture.

/// Template ID for operator instantiation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpTemplateId(pub Uuid);

/// Graph edit operation - the atomic unit of morphogenesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEdit {
    /// Add a new hyperedge using a registered operator template
    GrowHyperEdge {
        template: OpTemplateId,
        inputs: Vec<NodeId>,
        output: NodeId,
    },

    /// Remove an operator and its connections
    PruneOp {
        op: OpId,
    },

    /// Freeze a subgraph to prevent further modification
    FreezeSubgraph {
        ops: Vec<OpId>,
    },

    /// Promote a pattern to a new operator template (meta-learning)
    PromotePattern {
        motif: Vec<OpId>,
        new_template: OpTemplateId,
    },
}

impl GraphEdit {
    /// Apply the edit to a semantic graph
    pub fn apply_to(&self, graph: &mut SemanticGraph) -> anyhow::Result<()> {
        match self {
            GraphEdit::GrowHyperEdge { template, inputs, output } => {
                self.apply_grow(graph, *template, inputs, *output)
            }
            GraphEdit::PruneOp { op } => {
                self.apply_prune(graph, *op)
            }
            GraphEdit::FreezeSubgraph { ops } => {
                self.apply_freeze(graph, ops)
            }
            GraphEdit::PromotePattern { motif, new_template } => {
                self.apply_promote(graph, motif, *new_template)
            }
        }
    }

    fn apply_grow(&self, graph: &mut SemanticGraph, template: OpTemplateId,
                  inputs: &[NodeId], output: NodeId) -> anyhow::Result<()> {
        // Validate that inputs and output exist
        for &input_id in inputs {
            if !graph.has_node(input_id) {
                anyhow::bail!("Input node {} does not exist", input_id.0);
            }
        }
        if !graph.has_node(output) {
            anyhow::bail!("Output node {} does not exist", output.0);
        }

        // TODO: Instantiate operator from template
        // For now, create a generic operator
        let op_name = format!("grown_op_{}", Uuid::new_v4().simple());
        let operator = self.create_operator_from_template(template, &op_name, inputs, output)?;
        graph.add_operator_generic(operator);

        Ok(())
    }

    fn apply_prune(&self, graph: &mut SemanticGraph, op_id: OpId) -> anyhow::Result<()> {
        // Find and remove the operator
        let op_index = graph.operators.iter().position(|op| match op {
            OperatorType::FuseVA(op) => op.id == op_id,
            OperatorType::Attention(op) => op.id == op_id,
            OperatorType::MotorControl(op) => op.id == op_id,
        });

        if let Some(index) = op_index {
            graph.operators.remove(index);
            Ok(())
        } else {
            anyhow::bail!("Operator {} not found", op_id.0);
        }
    }

    fn apply_freeze(&self, graph: &mut SemanticGraph, ops: &[OpId]) -> anyhow::Result<()> {
        // Mark operators as frozen (conceptually immutable)
        // Implementation would add frozen flag to operators
        for &op_id in ops {
            if !graph.has_operator(op_id) {
                anyhow::bail!("Operator {} not found for freezing", op_id.0);
            }
        }
        // TODO: Implement freezing logic
        Ok(())
    }

    fn apply_promote(&self, graph: &mut SemanticGraph, motif: &[OpId],
                     new_template: OpTemplateId) -> anyhow::Result<()> {
        // Validate motif exists
        for &op_id in motif {
            if !graph.has_operator(op_id) {
                anyhow::bail!("Operator {} in motif not found", op_id.0);
            }
        }

        // TODO: Extract pattern and create new template
        // This is meta-learning: learning new operators from successful patterns
        Ok(())
    }

    fn create_operator_from_template(&self, _template: OpTemplateId, name: &str,
                                    inputs: &[NodeId], output: NodeId) -> anyhow::Result<OperatorType> {
        // TODO: Look up template and create appropriate operator type
        // For now, assume it's a FuseVA operator
        let op = HyperEdge::<FuseVA>::new(
            name.to_string(),
            inputs.get(0).copied().unwrap_or(NodeId(Uuid::nil())),
            inputs.get(1).copied().unwrap_or(NodeId(Uuid::nil())),
            output
        );
        Ok(OperatorType::FuseVA(op))
    }
}

/// Extension methods for SemanticGraph
impl SemanticGraph {
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.node_ids().contains(&node_id)
    }

    pub fn has_operator(&self, op_id: OpId) -> bool {
        self.operator_ids().contains(&op_id)
    }

    pub fn add_operator_generic(&mut self, op: OperatorType) {
        self.operators.push(op);
    }
}

/// Edit script - sequence of graph edits for atomic morphogenesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditScript {
    pub id: Uuid,
    pub name: String,
    pub edits: Vec<GraphEdit>,
    pub preconditions: Vec<String>, // Validation conditions
    pub postconditions: Vec<String>, // Expected outcomes
}

impl EditScript {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            edits: Vec::new(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
        }
    }

    pub fn add_edit(&mut self, edit: GraphEdit) -> &mut Self {
        self.edits.push(edit);
        self
    }

    pub fn add_precondition(&mut self, condition: impl Into<String>) -> &mut Self {
        self.preconditions.push(condition.into());
        self
    }

    pub fn add_postcondition(&mut self, condition: impl Into<String>) -> &mut Self {
        self.postconditions.push(condition.into());
        self
    }

    /// Apply the entire script atomically
    pub fn apply_to(&self, graph: &mut SemanticGraph) -> anyhow::Result<()> {
        // Check preconditions
        for precondition in &self.preconditions {
            if !self.evaluate_condition(graph, precondition)? {
                anyhow::bail!("Precondition failed: {}", precondition);
            }
        }

        // Apply all edits
        for edit in &self.edits {
            edit.apply_to(graph)?;
        }

        // Check postconditions
        for postcondition in &self.postconditions {
            if !self.evaluate_condition(graph, postcondition)? {
                anyhow::bail!("Postcondition failed: {}", postcondition);
            }
        }

        Ok(())
    }

    fn evaluate_condition(&self, _graph: &SemanticGraph, _condition: &str) -> anyhow::Result<bool> {
        // TODO: Implement condition evaluation
        // This would parse simple expressions like "node_count > 5"
        Ok(true)
    }
}

/// Morphogenesis strategy - high-level patterns for graph evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MorphogenesisStrategy {
    /// Grow capacity in response to learning bottlenecks
    ScaleUp { target_domains: Vec<String> },

    /// Specialize for current task distribution
    Specialize { task_patterns: Vec<String> },

    /// Compress redundant structures
    Compress { compression_targets: Vec<String> },

    /// Explore new architectural patterns
    Explore { exploration_budget: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_edit_creation() {
        let node_id = NodeId(Uuid::new_v4());
        let edit = GraphEdit::PruneOp { op: OpId(Uuid::new_v4()) };
        assert!(matches!(edit, GraphEdit::PruneOp { .. }));
    }

    #[test]
    fn edit_script_creation() {
        let mut script = EditScript::new("test_script");
        script.add_edit(GraphEdit::PruneOp { op: OpId(Uuid::new_v4()) });
        assert_eq!(script.edits.len(), 1);
    }

    #[test]
    fn semantic_graph_edit_application() {
        let mut graph = SemanticGraph::new();
        let node = ConceptNode::<Visual>::new("test", 128);
        graph.add_concept(node.clone());

        // Pruning non-existent operator should fail
        let edit = GraphEdit::PruneOp { op: OpId(Uuid::new_v4()) };
        assert!(edit.apply_to(&mut graph).is_err());
    }
}
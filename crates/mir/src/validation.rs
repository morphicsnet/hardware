use crate::semantic::*;

/// Validation for semantic graph constraints and type safety
/// Implements the "semantic firewall" concept to prevent illegal connections

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Node {0} not found")]
    NodeNotFound(String),

    #[error("Operator {0} not found")]
    OperatorNotFound(String),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Manifold incompatibility: {0}")]
    ManifoldError(String),
}

/// Validate entire semantic graph
pub fn validate_semantic_graph(graph: &SemanticGraph) -> anyhow::Result<()> {
    // Validate all nodes exist
    validate_node_references(graph)?;

    // Validate operator signatures
    validate_operator_signatures(graph)?;

    // Validate manifold constraints
    validate_manifold_constraints(graph)?;

    // Validate custom constraints
    validate_custom_constraints(graph)?;

    Ok(())
}

/// Validate that all referenced nodes exist
fn validate_node_references(graph: &SemanticGraph) -> anyhow::Result<()> {
    let existing_nodes: std::collections::HashSet<_> = graph.node_ids().into_iter().collect();

    for operator in &graph.operators {
        let (in1, in2, out) = match operator {
            OperatorType::FuseVA(op) => (op.in1, op.in2, op.out),
            OperatorType::Attention(op) => (op.in1, op.in2, op.out),
            OperatorType::MotorControl(op) => (op.in1, op.in2, op.out),
        };

        if !existing_nodes.contains(&in1) {
            anyhow::bail!(ValidationError::NodeNotFound(format!("Input 1: {}", in1.0)));
        }
        if !existing_nodes.contains(&in2) {
            anyhow::bail!(ValidationError::NodeNotFound(format!("Input 2: {}", in2.0)));
        }
        if !existing_nodes.contains(&out) {
            anyhow::bail!(ValidationError::NodeNotFound(format!("Output: {}", out.0)));
        }
    }

    Ok(())
}

/// Validate operator type signatures
fn validate_operator_signatures(graph: &SemanticGraph) -> anyhow::Result<()> {
    for operator in &graph.operators {
        match operator {
            OperatorType::FuseVA(op) => validate_fusion_signature(op, graph)?,
            OperatorType::Attention(op) => validate_attention_signature(op, graph)?,
            OperatorType::MotorControl(op) => validate_motor_signature(op, graph)?,
        }
    }
    Ok(())
}

/// Validate multimodal fusion operator
fn validate_fusion_signature(op: &HyperEdge<FuseVA>, graph: &SemanticGraph) -> anyhow::Result<()> {
    // Get manifold types of input nodes
    let in1_manifold = get_node_manifold(graph, op.in1)?;
    let in2_manifold = get_node_manifold(graph, op.in2)?;
    let out_manifold = get_node_manifold(graph, op.out)?;

    // Fusion requires Visual + Audio → Visual
    if in1_manifold != Visual::ID && in1_manifold != Audio::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Fusion input 1 must be Visual or Audio, got manifold {}", in1_manifold)
        ));
    }
    if in2_manifold != Visual::ID && in2_manifold != Audio::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Fusion input 2 must be Visual or Audio, got manifold {}", in2_manifold)
        ));
    }
    if out_manifold != Visual::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Fusion output must be Visual, got manifold {}", out_manifold)
        ));
    }

    Ok(())
}

/// Validate attention operator
fn validate_attention_signature(op: &HyperEdge<Attention>, graph: &SemanticGraph) -> anyhow::Result<()> {
    // Attention requires Language + Language → Language
    let in1_manifold = get_node_manifold(graph, op.in1)?;
    let in2_manifold = get_node_manifold(graph, op.in2)?;
    let out_manifold = get_node_manifold(graph, op.out)?;

    if in1_manifold != Language::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Attention input 1 must be Language, got manifold {}", in1_manifold)
        ));
    }
    if in2_manifold != Language::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Attention input 2 must be Language, got manifold {}", in2_manifold)
        ));
    }
    if out_manifold != Language::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Attention output must be Language, got manifold {}", out_manifold)
        ));
    }

    Ok(())
}

/// Validate motor control operator
fn validate_motor_signature(op: &HyperEdge<MotorControl>, graph: &SemanticGraph) -> anyhow::Result<()> {
    // Motor control requires Visual + Motor → Motor
    let in1_manifold = get_node_manifold(graph, op.in1)?;
    let in2_manifold = get_node_manifold(graph, op.in2)?;
    let out_manifold = get_node_manifold(graph, op.out)?;

    if in1_manifold != Visual::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Motor control input 1 must be Visual, got manifold {}", in1_manifold)
        ));
    }
    if in2_manifold != Motor::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Motor control input 2 must be Motor, got manifold {}", in2_manifold)
        ));
    }
    if out_manifold != Motor::ID {
        anyhow::bail!(ValidationError::TypeMismatch(
            format!("Motor control output must be Motor, got manifold {}", out_manifold)
        ));
    }

    Ok(())
}

/// Get manifold ID for a node
fn get_node_manifold(graph: &SemanticGraph, node_id: NodeId) -> anyhow::Result<u32> {
    for concept in &graph.concepts {
        let (concept_id, manifold_id) = match concept {
            ConceptNodeType::Visual(n) => (n.id, Visual::ID),
            ConceptNodeType::Audio(n) => (n.id, Audio::ID),
            ConceptNodeType::Motor(n) => (n.id, Motor::ID),
            ConceptNodeType::Language(n) => (n.id, Language::ID),
        };

        if concept_id == node_id {
            return Ok(manifold_id);
        }
    }

    anyhow::bail!(ValidationError::NodeNotFound(format!("Node {} not found in concepts", node_id.0)));
}

/// Validate manifold-specific constraints
fn validate_manifold_constraints(_graph: &SemanticGraph) -> anyhow::Result<()> {
    // TODO: Add manifold-specific validation rules
    // For example: embedding dimensions must be compatible
    // Or: certain manifolds cannot connect directly
    Ok(())
}

/// Validate custom constraints from the graph
fn validate_custom_constraints(graph: &SemanticGraph) -> anyhow::Result<()> {
    for constraint in &graph.constraints {
        match constraint {
            TypeConstraint::NodeExists { id, manifold_id } => {
                let actual_manifold = get_node_manifold(graph, *id)?;
                if actual_manifold != *manifold_id {
                    anyhow::bail!(ValidationError::ConstraintViolation(
                        format!("Node {} manifold mismatch: expected {}, got {}",
                               id.0, manifold_id, actual_manifold)
                    ));
                }
            }
            TypeConstraint::EdgeCompatible { .. } => {
                // TODO: Implement edge compatibility validation
            }
            TypeConstraint::Axiom { description, .. } => {
                // TODO: Implement axiom evaluation
                anyhow::bail!(ValidationError::ConstraintViolation(
                    format!("Axiom validation not implemented: {}", description)
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_empty_graph() {
        let graph = SemanticGraph::new();
        assert!(validate_semantic_graph(&graph).is_ok());
    }

    #[test]
    fn validate_graph_with_valid_fusion() {
        let mut graph = SemanticGraph::new();

        let visual_node = ConceptNode::<Visual>::new("image", 512);
        let audio_node = ConceptNode::<Audio>::new("sound", 128);
        let output_node = ConceptNode::<Visual>::new("fused", 640);

        graph.add_concept(visual_node.clone());
        graph.add_concept(audio_node.clone());
        graph.add_concept(output_node.clone());

        let op = HyperEdge::<FuseVA>::new("fuse_1", visual_node.id, audio_node.id, output_node.id);
        graph.add_operator(op);

        assert!(validate_semantic_graph(&graph).is_ok());
    }

    #[test]
    fn validate_graph_with_invalid_connection() {
        let mut graph = SemanticGraph::new();

        let motor_node = ConceptNode::<Motor>::new("motor", 64);
        let audio_node = ConceptNode::<Audio>::new("sound", 128);
        let output_node = ConceptNode::<Visual>::new("fused", 640);

        graph.add_concept(motor_node.clone());
        graph.add_concept(audio_node.clone());
        graph.add_concept(output_node.clone());

        // Invalid: Motor + Audio → Visual (fusion requires Visual/Audio inputs)
        let op = HyperEdge::<FuseVA>::new("invalid_fuse", motor_node.id, audio_node.id, output_node.id);
        graph.add_operator(op);

        assert!(validate_semantic_graph(&graph).is_err());
    }

    #[test]
    fn validate_missing_node() {
        let mut graph = SemanticGraph::new();

        let visual_node = ConceptNode::<Visual>::new("image", 512);
        let missing_node = NodeId(uuid::Uuid::new_v4());

        graph.add_concept(visual_node.clone());

        // Reference to non-existent node
        let op = HyperEdge::<FuseVA>::new("bad_op", visual_node.id, missing_node, visual_node.id);
        graph.add_operator(op);

        assert!(validate_semantic_graph(&graph).is_err());
    }
}
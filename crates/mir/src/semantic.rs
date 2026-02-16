use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use uuid::Uuid;

/// IR-A: Semantic Hypergraph IR with Manifold Constraints
///
/// This is the source of truth for cognitive structure, where we encode
/// "concepts on manifolds" and "typed hyperedges" with semantic firewalls.

/// Manifold type trait - each manifold has a unique ID for type safety
pub trait ManifoldType {
    const ID: u32;
    const NAME: &'static str;
}

/// Example manifold types - extend as needed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Visual;
impl ManifoldType for Visual {
    const ID: u32 = 1;
    const NAME: &'static str = "visual";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Audio;
impl ManifoldType for Audio {
    const ID: u32 = 2;
    const NAME: &'static str = "audio";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Motor;
impl ManifoldType for Motor {
    const ID: u32 = 3;
    const NAME: &'static str = "motor";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Language;
impl ManifoldType for Language {
    const ID: u32 = 4;
    const NAME: &'static str = "language";
}

/// Node ID type for uniquely identifying concepts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

/// Operator ID type for uniquely identifying hyperedges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId(pub Uuid);

/// A concept node living on a specific manifold M
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode<M: ManifoldType> {
    pub id: NodeId,
    pub name: String,
    pub embedding_dim: usize,
    _manifold: PhantomData<M>,
}

impl<M: ManifoldType> ConceptNode<M> {
    pub fn new(name: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            id: NodeId(Uuid::new_v4()),
            name: name.into(),
            embedding_dim,
            _manifold: PhantomData,
        }
    }
}

/// Fixed-arity operator type signature: (In1, In2) -> Out
pub trait HyperOpSig {
    type In1: ManifoldType;
    type In2: ManifoldType;
    type Out: ManifoldType;
    const ARITY: usize = 2;
    const NAME: &'static str;
}

/// Example multimodal fusion operator: (Visual, Audio) -> Visual
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuseVA;
impl HyperOpSig for FuseVA {
    type In1 = Visual;
    type In2 = Audio;
    type Out = Visual;
    const NAME: &'static str = "fuse_visual_audio";
}

/// Attention operator: (Language, Language) -> Language
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Attention;
impl HyperOpSig for Attention {
    type In1 = Language;
    type In2 = Language;
    type Out = Language;
    const NAME: &'static str = "attention";
}

/// Motor control operator: (Visual, Motor) -> Motor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MotorControl;
impl HyperOpSig for MotorControl {
    type In1 = Visual;
    type In2 = Motor;
    type Out = Motor;
    const NAME: &'static str = "motor_control";
}

/// An instantiated hyperedge in the semantic graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge<S: HyperOpSig> {
    pub id: OpId,
    pub name: String,
    pub in1: NodeId,
    pub in2: NodeId,
    pub out: NodeId,
    pub parameters: HashMap<String, f32>,
    _sig: PhantomData<S>,
}

impl<S: HyperOpSig> HyperEdge<S> {
    pub fn new(name: impl Into<String>, in1: NodeId, in2: NodeId, out: NodeId) -> Self {
        Self {
            id: OpId(Uuid::new_v4()),
            name: name.into(),
            in1,
            in2,
            out,
            parameters: HashMap::new(),
            _sig: PhantomData,
        }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: f32) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }
}

/// Type constraint for semantic firewall validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Node must exist and be of specific manifold type
    NodeExists { id: NodeId, manifold_id: u32 },
    /// Edge inputs/outputs must be compatible
    EdgeCompatible { edge_id: OpId, in1_manifold: u32, in2_manifold: u32, out_manifold: u32 },
    /// Custom axiom/constraint
    Axiom { description: String, expression: String },
}

/// The semantic hypergraph - IR-A
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGraph {
    pub manifolds: HashMap<u32, String>, // manifold_id -> name mapping
    pub concepts: Vec<ConceptNodeType>, // Heterogenous concept nodes
    pub operators: Vec<OperatorType>,   // Heterogenous operators
    pub constraints: Vec<TypeConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptNodeType {
    Visual(ConceptNode<Visual>),
    Audio(ConceptNode<Audio>),
    Motor(ConceptNode<Motor>),
    Language(ConceptNode<Language>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperatorType {
    FuseVA(HyperEdge<FuseVA>),
    Attention(HyperEdge<Attention>),
    MotorControl(HyperEdge<MotorControl>),
}

impl SemanticGraph {
    pub fn new() -> Self {
        let mut manifolds = HashMap::new();
        manifolds.insert(Visual::ID, Visual::NAME.to_string());
        manifolds.insert(Audio::ID, Audio::NAME.to_string());
        manifolds.insert(Motor::ID, Motor::NAME.to_string());
        manifolds.insert(Language::ID, Language::NAME.to_string());

        Self {
            manifolds,
            concepts: Vec::new(),
            operators: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a concept node with type safety
    pub fn add_concept<M: ManifoldType>(&mut self, node: ConceptNode<M>) {
        let concept_type = match M::ID {
            1 => ConceptNodeType::Visual(node),
            2 => ConceptNodeType::Audio(node),
            3 => ConceptNodeType::Motor(node),
            4 => ConceptNodeType::Language(node),
            _ => panic!("Unknown manifold type"),
        };
        self.concepts.push(concept_type);
    }

    /// Add an operator with type safety
    pub fn add_operator<S: HyperOpSig>(&mut self, op: HyperEdge<S>) {
        let op_type = match S::NAME {
            "fuse_visual_audio" => OperatorType::FuseVA(op),
            "attention" => OperatorType::Attention(op),
            "motor_control" => OperatorType::MotorControl(op),
            _ => panic!("Unknown operator type"),
        };
        self.operators.push(op_type);
    }

    /// Get all node IDs in the graph
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.concepts.iter().filter_map(|c| match c {
            ConceptNodeType::Visual(n) => Some(n.id),
            ConceptNodeType::Audio(n) => Some(n.id),
            ConceptNodeType::Motor(n) => Some(n.id),
            ConceptNodeType::Language(n) => Some(n.id),
        }).collect()
    }

    /// Get all operator IDs in the graph
    pub fn operator_ids(&self) -> Vec<OpId> {
        self.operators.iter().filter_map(|o| match o {
            OperatorType::FuseVA(op) => Some(op.id),
            OperatorType::Attention(op) => Some(op.id),
            OperatorType::MotorControl(op) => Some(op.id),
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_graph_creation() {
        let graph = SemanticGraph::new();
        assert_eq!(graph.manifolds.len(), 4);
        assert!(graph.concepts.is_empty());
        assert!(graph.operators.is_empty());
    }

    #[test]
    fn add_visual_concept() {
        let mut graph = SemanticGraph::new();
        let node = ConceptNode::<Visual>::new("cat", 512);
        graph.add_concept(node);
        assert_eq!(graph.concepts.len(), 1);
    }

    #[test]
    fn add_fusion_operator() {
        let mut graph = SemanticGraph::new();
        let visual_node = ConceptNode::<Visual>::new("image", 512);
        let audio_node = ConceptNode::<Audio>::new("sound", 128);
        let output_node = ConceptNode::<Visual>::new("fused", 640);

        graph.add_concept(visual_node.clone());
        graph.add_concept(audio_node.clone());
        graph.add_concept(output_node.clone());

        let op = HyperEdge::<FuseVA>::new("fuse_1", visual_node.id, audio_node.id, output_node.id);
        graph.add_operator(op);

        assert_eq!(graph.operators.len(), 1);
    }
}
//! Physics Intermediate Representation (PhysIR) for energy-based computation.
//!
//! PhysIR represents the problem as a "Mass-Spring-Damper" system where nodes are physical
//! variables and edges are energy couplings.

use serde::{Deserialize, Serialize};

/// Node in PhysIR: represents a physical state variable (voltage, spin, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysNode {
    pub id: String,
    pub mass: f64,              // Inertia (for continuous dynamics)
    pub damping: f64,           // Dissipation coefficient
    pub voltage_bounds: (f64, f64), // Min/max voltage or spin values
    pub initial_value: f64,     // Starting state
}

/// Edge in PhysIR: represents energy coupling between variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysEdge {
    pub source: String,
    pub target: String,
    pub coupling_strength: f64, // J_ij in Hamiltonian H = ∑ J_ij x_i x_j
    pub energy_penalty: f64,    // Additional penalty term
}

/// Dynamics type for the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysDynamics {
    GradientDescent { learning_rate: f64 }, // \dot{x} = -η ∇H
    SymplecticEuler { time_step: f64 },     // Hamiltonian preservation
    Langevin { temperature: f64 },          // Stochastic relaxation
}

/// Physical Intermediate Representation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysIR {
    pub nodes: Vec<PhysNode>,
    pub edges: Vec<PhysEdge>,
    pub dynamics: PhysDynamics,
    pub hamiltonian: Vec<f64>,  // Precomputed quadratic terms
    pub gradients: Vec<Vec<f64>>, // ∂H/∂x_i for each variable
}

impl PhysIR {
    pub fn new(dynamics: PhysDynamics) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            dynamics,
            hamiltonian: Vec::new(),
            gradients: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: PhysNode) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: PhysEdge) {
        self.edges.push(edge);
    }

    /// Compute Hamiltonian matrix from edges
    pub fn compute_hamiltonian(&mut self) {
        let n = self.nodes.len();
        self.hamiltonian = vec![0.0; n * n];
        for edge in &self.edges {
            let i = self.nodes.iter().position(|n| n.id == edge.source).unwrap();
            let j = self.nodes.iter().position(|n| n.id == edge.target).unwrap();
            let idx = i * n + j;
            self.hamiltonian[idx] = edge.coupling_strength;
        }
    }

    /// Compute gradients for gradient descent
    pub fn compute_gradients(&mut self) {
        let n = self.nodes.len();
        self.gradients = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                // ∂H/∂x_i = 2 ∑_j J_ij x_j (for quadratic H)
                let sum: f64 = (0..n).map(|k| 2.0 * self.hamiltonian[i * n + k] * self.nodes[k].initial_value).sum();
                self.gradients[i][j] = sum;
            }
        }
    }
}
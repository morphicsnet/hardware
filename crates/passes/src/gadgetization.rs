//! Gadgetization Pass for reducing higher-order interactions.
//!
//! Implements gadget library to replace 3-body terms with auxiliary variables and quadratic couplings.

use crate::nir;
use crate::nir::phys_ir::{PhysIR, PhysNode, PhysEdge};
use anyhow::Result;

/// Pass for gadgetization: higher-order to pairwise interactions.
pub struct GadgetizationPass;

impl crate::Pass for GadgetizationPass {
    fn name(&self) -> &str { "gadgetization" }

    fn run(&mut self, mut g: nir::Graph) -> Result<nir::Graph> {
        if let Some(phys_ir_val) = g.attributes.get_mut("phys_ir") {
            let mut phys_ir: PhysIR = serde_json::from_value(phys_ir_val.clone())?;
            self.gadgetize(&mut phys_ir);
            *phys_ir_val = serde_json::json!(phys_ir);
        }
        Ok(g)
    }
}

impl GadgetizationPass {
    fn gadgetize(&self, phys_ir: &mut PhysIR) {
        // Find 3-body terms (simplified: assume encoded as edges with high coupling)
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        for edge in &phys_ir.edges {
            if edge.coupling_strength > 10.0 { // Arbitrary threshold for "higher-order"
                // Introduce auxiliary variable
                let aux_id = format!("aux_{}_{}", edge.source, edge.target);
                let aux_node = PhysNode {
                    id: aux_id.clone(),
                    mass: 1.0,
                    damping: 0.1,
                    voltage_bounds: (-1.0, 1.0),
                    initial_value: 0.0,
                };
                new_nodes.push(aux_node);

                // Replace 3-body with quadratic terms
                let quad_edge1 = PhysEdge {
                    source: edge.source.clone(),
                    target: aux_id.clone(),
                    coupling_strength: edge.coupling_strength / 2.0,
                    energy_penalty: 0.0,
                };
                let quad_edge2 = PhysEdge {
                    source: aux_id,
                    target: edge.target.clone(),
                    coupling_strength: edge.coupling_strength / 2.0,
                    energy_penalty: 0.0,
                };
                new_edges.push(quad_edge1);
                new_edges.push(quad_edge2);
            } else {
                new_edges.push(edge.clone());
            }
        }

        phys_ir.nodes.extend(new_nodes);
        phys_ir.edges = new_edges;
        phys_ir.compute_hamiltonian();
        phys_ir.compute_gradients();
    }
}
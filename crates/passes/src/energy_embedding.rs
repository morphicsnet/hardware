//! Energy Embedding Pass for mapping PhysIR to hardware.
//!
//! Embeds PhysIR onto energy-based hardware like memristive crossbars.

use crate::nir;
use crate::hal;
use crate::nir::phys_ir::PhysIR;
use anyhow::Result;

/// Pass for embedding PhysIR onto energy hardware.
pub struct EnergyEmbeddingPass;

impl crate::Pass for EnergyEmbeddingPass {
    fn name(&self) -> &str { "energy-embedding" }

    fn run(&mut self, mut g: nir::Graph) -> Result<nir::Graph> {
        if let Some(phys_ir_val) = g.attributes.get("phys_ir") {
            let phys_ir: PhysIR = serde_json::from_value(phys_ir_val.clone())?;
            let embedding = self.embed(&phys_ir, &g)?;
            g.attributes.insert("hardware_embedding".to_string(), serde_json::json!(embedding));
        }
        Ok(g)
    }
}

impl EnergyEmbeddingPass {
    fn embed(&self, phys_ir: &PhysIR, g: &nir::Graph) -> Result<serde_json::Value> {
        // Simple embedding: map nodes to hardware indices
        let mut node_mapping = std::collections::HashMap::new();
        for (i, node) in phys_ir.nodes.iter().enumerate() {
            node_mapping.insert(node.id.clone(), i);
        }

        // Generate conductance matrix G_ij
        let n = phys_ir.nodes.len();
        let mut g_matrix = vec![vec![0.0; n]; n];
        for edge in &phys_ir.edges {
            let i = *node_mapping.get(&edge.source).unwrap();
            let j = *node_mapping.get(&edge.target).unwrap();
            g_matrix[i][j] = edge.coupling_strength;
        }

        Ok(serde_json::json!({
            "node_mapping": node_mapping,
            "conductance_matrix": g_matrix,
            "dynamics": phys_ir.dynamics
        }))
    }
}
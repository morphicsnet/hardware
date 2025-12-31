//! Common utilities for frontend implementations

use nc_nir::{Graph, Population, Connection, Probe};
use serde_json::Value;
use std::collections::HashMap;

/// Common utilities for NIR graph construction from various frameworks
pub struct GraphBuilder {
    graph: Graph,
    population_map: HashMap<String, usize>, // name -> index
}

impl GraphBuilder {
    /// Create a new graph builder with the given name
    pub fn new(name: &str) -> Self {
        Self {
            graph: Graph::new(name),
            population_map: HashMap::new(),
        }
    }

    /// Add a population to the graph
    pub fn add_population(&mut self, name: String, size: u32, model: String, params: Value) -> &mut Self {
        let index = self.graph.populations.len();
        self.population_map.insert(name.clone(), index);

        self.graph.populations.push(Population {
            name,
            size,
            model,
            params,
        });
        self
    }

    /// Add a connection between populations
    pub fn add_connection(&mut self, pre: &str, post: &str, weight: f32, delay_ms: f32, plasticity: Option<Value>) -> Result<&mut Self, String> {
        if !self.population_map.contains_key(pre) {
            return Err(format!("Pre population '{}' not found", pre));
        }
        if !self.population_map.contains_key(post) {
            return Err(format!("Post population '{}' not found", post));
        }

        let plasticity_rule = plasticity.map(|p| nc_nir::PlasticityRule {
            kind: nc_nir::PlasticityKind::Custom, // Default to custom, could be extended
            params: p,
        });

        self.graph.connections.push(Connection {
            pre: pre.to_string(),
            post: post.to_string(),
            weight,
            delay_ms,
            plasticity: plasticity_rule,
        });
        Ok(self)
    }

    /// Add a probe to monitor a population
    pub fn add_probe(&mut self, target: &str, kind: &str) -> Result<&mut Self, String> {
        if !self.population_map.contains_key(target) {
            return Err(format!("Probe target '{}' not found", target));
        }

        self.graph.probes.push(Probe {
            target: target.to_string(),
            kind: kind.to_string(),
        });
        Ok(self)
    }

    /// Set graph dialect
    pub fn set_dialect(&mut self, dialect: nc_nir::Dialect) -> &mut Self {
        self.graph.dialect = Some(dialect);
        self
    }

    /// Add custom attributes to the graph
    pub fn add_attribute(&mut self, key: String, value: Value) -> &mut Self {
        self.graph.attributes.insert(key, value);
        self
    }

    /// Build and return the final graph
    pub fn build(mut self) -> Graph {
        self.graph.ensure_version_tag();
        self.graph
    }

    /// Get a reference to the current graph state
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

/// Neuron model parameter mapping utilities
pub mod neuron_models {
    use serde_json::json;

    /// Standard Leaky Integrate-and-Fire neuron parameters
    pub fn lif_neuron(tau_m: f32, v_th: f32, v_reset: f32) -> serde_json::Value {
        json!({
            "tau_m": tau_m,
            "v_th": v_th,
            "v_reset": v_reset
        })
    }

    /// Adaptive Leaky Integrate-and-Fire neuron
    pub fn alif_neuron(tau_m: f32, v_th: f32, v_reset: f32, tau_adapt: f32) -> serde_json::Value {
        json!({
            "tau_m": tau_m,
            "v_th": v_th,
            "v_reset": v_reset,
            "tau_adapt": tau_adapt,
            "adaptation": true
        })
    }

    /// Source neuron (constant input)
    pub fn source_neuron() -> serde_json::Value {
        json!({})
    }

    /// Poisson spike source
    pub fn poisson_source(rate: f32) -> serde_json::Value {
        json!({
            "rate": rate,
            "distribution": "poisson"
        })
    }
}

/// Connection weight utilities
pub mod connections {
    /// Create a simple connection specification
    pub fn simple(weight: f32, delay_ms: f32) -> (f32, f32, Option<serde_json::Value>) {
        (weight, delay_ms, None)
    }

    /// Create a plastic connection (STDP)
    pub fn stdp(weight: f32, delay_ms: f32, tau_plus: f32, tau_minus: f32) -> (f32, f32, Option<serde_json::Value>) {
        let plasticity = serde_json::json!({
            "learning_rule": "STDP",
            "tau_plus": tau_plus,
            "tau_minus": tau_minus
        });
        (weight, delay_ms, Some(plasticity))
    }
}

/// Validation utilities for constructed graphs
pub mod validation {
    use nc_nir::Graph;

    /// Validate that a constructed graph is well-formed
    pub fn validate_construction(graph: &Graph) -> Result<(), String> {
        // Check for empty graph
        if graph.populations.is_empty() {
            return Err("Graph has no populations".to_string());
        }

        // Check for isolated populations (no connections)
        let connected_pops: std::collections::HashSet<_> = graph.connections.iter()
            .flat_map(|c| vec![c.pre.clone(), c.post.clone()])
            .collect();

        let isolated_pops: Vec<_> = graph.populations.iter()
            .filter(|p| !connected_pops.contains(&p.name))
            .map(|p| &p.name)
            .collect();

        if !isolated_pops.is_empty() {
            eprintln!("Warning: Isolated populations detected: {:?}", isolated_pops);
        }

        // Validate the graph structure
        graph.validate().map_err(|e| format!("Graph validation failed: {}", e))?;

        Ok(())
    }
}
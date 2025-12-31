use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use nc_passes::Pass;

/// Quantize weight to Loihi precision (8-bit by default)
fn quantize_weight(w: f32, bits: u32) -> f32 {
    // Loihi uses symmetric quantization around zero
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let max_val = (levels / 2) as f32;
    let scale = max_val / 1.0; // Map [-1, 1] to full range

    let w_clamped = w.clamp(-1.0, 1.0);
    (w_clamped * scale).round() / scale
}

/// Convert quantized weight to Loihi synaptic format
fn weight_to_synaptic_format(weight_q: f32, bits: u32) -> (u8, u8) {
    // Loihi uses 8-bit weights with sign-magnitude encoding
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let max_val = (levels / 2) as f32;

    // Map [-1, 1] to [0, 255] for 8-bit
    let index = ((weight_q + 1.0) * max_val).round() as u8;
    let sign = if weight_q < 0.0 { 1u8 } else { 0u8 };
    let magnitude = index & 0x7F; // 7 bits for magnitude

    (sign, magnitude)
}

/// Loihi-specific passes for Loihi 2 hardware optimizations
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// Loihi core mapping pass - assigns neurons to cores and manages compartments
    pub struct LoihiCoreMappingPass;
    impl Pass for LoihiCoreMappingPass {
        fn name(&self) -> &str { "loihi-core-mapping" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract Loihi capabilities
            let caps = g.attributes.get("caps_loihi");
            let neurons_per_core = caps.and_then(|v| v.get("neurons_per_core")).and_then(|x| x.as_u64()).unwrap_or(1024);
            let compartments_per_neuron = 1; // Loihi uses compartment-based neurons
            let cores_per_chip = caps.and_then(|v| v.get("cores_per_chip")).and_then(|x| x.as_u64()).unwrap_or(128);

            // Calculate total capacity
            let total_cores = cores_per_chip;
            let total_neurons_capacity = total_cores * neurons_per_core;

            let total_neurons: usize = g.populations.iter().map(|p| p.size as usize).sum();

            // Simple core assignment strategy
            let mut core_assignments = Vec::new();
            let mut current_core = 0usize;

            for pop in &g.populations {
                let cores_needed = ((pop.size as f64) / (neurons_per_core as f64)).ceil() as usize;

                for i in 0..cores_needed {
                    let core_id = (current_core + i) % cores_per_chip as usize;
                    core_assignments.push(serde_json::json!({
                        "population": pop.name,
                        "core_id": core_id,
                        "chip_id": 0, // Single chip for now
                        "neuron_start": (i * neurons_per_core as usize) % neurons_per_core as usize,
                        "neurons_assigned": pop.size.min(neurons_per_core as u32)
                    }));
                }

                current_core = (current_core + cores_needed) % cores_per_chip as usize;
            }

            let meta = serde_json::json!({
                "total_neurons": total_neurons,
                "total_capacity": total_neurons_capacity,
                "cores_used": core_assignments.len(),
                "chips_used": 1,
                "neurons_per_core": neurons_per_core,
                "compartments_per_neuron": compartments_per_neuron,
                "core_assignments": core_assignments
            });
            g.attributes.insert("loihi_core_mapping".to_string(), meta);
            Ok(g)
        }
    }

    /// Loihi synapse programming pass - converts weights to Loihi synaptic format
    pub struct LoihiSynapseProgrammingPass;
    impl Pass for LoihiSynapseProgrammingPass {
        fn name(&self) -> &str { "loihi-synapse-programming" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            let bits = g.attributes.get("caps_loihi")
                .and_then(|v| v.get("weight_bits"))
                .and_then(|x| x.as_u64())
                .unwrap_or(8) as u32;

            let mut programmed_synapses = Vec::new();
            for conn in &g.connections {
                let weight_q = quantize_weight(conn.weight, bits);
                let (sign, magnitude) = weight_to_synaptic_format(weight_q, bits);

                programmed_synapses.push(serde_json::json!({
                    "pre_population": conn.pre,
                    "post_population": conn.post,
                    "sign": sign,
                    "magnitude": magnitude,
                    "weight": weight_q,
                    "delay": conn.delay_ms,
                    "plasticity": conn.plasticity
                }));
            }

            let meta = serde_json::json!({
                "weight_bits": bits,
                "total_synapses": programmed_synapses.len(),
                "programmed_synapses": programmed_synapses
            });
            g.attributes.insert("loihi_synapse_programming".to_string(), meta);
            Ok(g)
        }
    }

    /// Loihi learning rule configuration pass
    pub struct LoihiLearningRulePass;
    impl Pass for LoihiLearningRulePass {
        fn name(&self) -> &str { "loihi-learning-rule" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Configure learning rules for plastic synapses
            let plastic_connections: Vec<_> = g.connections.iter()
                .filter(|c| c.plasticity.is_some())
                .collect();

            let mut learning_configs = Vec::new();
            for conn in &plastic_connections {
                if let Some(plasticity) = &conn.plasticity {
                    learning_configs.push(serde_json::json!({
                        "connection": format!("{}->{}", conn.pre, conn.post),
                        "learning_rule": plasticity.params.get("learning_rule").and_then(|v| v.as_str()).unwrap_or("none"),
                        "learning_rate": plasticity.params.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.01),
                        "tau_plus": plasticity.params.get("tau_plus").and_then(|v| v.as_f64()).unwrap_or(20.0),
                        "tau_minus": plasticity.params.get("tau_minus").and_then(|v| v.as_f64()).unwrap_or(20.0)
                    }));
                }
            }

            let meta = serde_json::json!({
                "plastic_connections": plastic_connections.len(),
                "learning_configs": learning_configs
            });
            g.attributes.insert("loihi_learning_rule".to_string(), meta);
            Ok(g)
        }
    }
}

/// Compile NIR to Loihi hardware artifacts
pub fn compile(graph: &nc_nir::Graph, manifest: &nc_hal::TargetManifest) -> Result<String> {
    // Validate input IR and target manifest
    graph.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    nc_hal::validate_manifest(manifest)?;

    // Optional telemetry profiling
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| nc_telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let labels = nc_telemetry::labels::backend(&graph.name, "loihi", Some(&manifest.name));
            Some(a.start_timer("backend.compile_ms", labels))
        } else {
            None
        }
    };

    // Create output directory
    let out_dir = std::path::PathBuf::from(format!("target/{}-{}", manifest.name, graph.name));
    if !out_dir.exists() {
        fs::create_dir_all(&out_dir).context("create output directory")?;
    }

    // Extract Loihi capabilities
    let neurons_per_core = manifest.capabilities.as_ref()
        .and_then(|c| c.max_neurons_per_core)
        .unwrap_or(1024) as usize;
    let compartments_per_neuron = 1; // Loihi uses compartment-based neurons
    let weight_bits = manifest.capabilities.as_ref()
        .and_then(|c| c.weight_precisions.as_ref())
        .and_then(|v| v.iter().max().copied())
        .unwrap_or(8) as usize;

    // Run Loihi-specific pass pipeline
    let g = run_loihi_pipeline(graph, neurons_per_core, compartments_per_neuron, weight_bits)?;

    // Generate hardware artifacts
    generate_loihi_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = nc_telemetry::labels::backend(&graph.name, "loihi", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    Ok(format!("artifact:{}", out_dir.display()))
}

/// Run Loihi-specific compilation pipeline
fn run_loihi_pipeline(
    graph: &nc_nir::Graph,
    neurons_per_core: usize,
    compartments_per_neuron: usize,
    weight_bits: usize
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Attach Loihi capabilities
    let caps = serde_json::json!({
        "neurons_per_core": neurons_per_core,
        "compartments_per_neuron": compartments_per_neuron,
        "cores_per_chip": 128,
        "weight_bits": weight_bits
    });
    g.attributes.insert("caps_loihi".to_string(), caps);

    // Run Loihi passes
    g = passes::LoihiCoreMappingPass.run(g)?;
    g = passes::LoihiSynapseProgrammingPass.run(g)?;
    g = passes::LoihiLearningRulePass.run(g)?;

    Ok(g)
}

/// Generate Loihi hardware artifacts
fn generate_loihi_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Generate core mapping and synapse configuration
    let core_mapping = graph.attributes.get("loihi_core_mapping")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let synapse_programming = graph.attributes.get("loihi_synapse_programming")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let learning_rules = graph.attributes.get("loihi_learning_rule")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // Loihi configuration
    let loihi_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "platform": "Loihi2",
        "core_mapping": core_mapping,
        "synapse_programming": synapse_programming,
        "learning_rules": learning_rules
    });

    fs::write(out_dir.join("loihi_config.json"), serde_json::to_string_pretty(&loihi_config)?)?;

    // Generate README
    let readme = format!(
        "Loihi Hardware Configuration for '{}'\n\
        ======================================\n\
        Generated: {}\n\
        Target: {}\n\
        Platform: Loihi 2\n\
        Populations: {}\n\
        Connections: {}\n\
        \nHardware Mapping:\n\
        - Neurons per core: {}\n\
        - Compartments per neuron: {}\n\
        - Cores per chip: 128\n\
        - Weight precision: {} bits\n",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        graph.populations.len(),
        graph.connections.len(),
        manifest.capabilities.as_ref().and_then(|c| c.max_neurons_per_core).unwrap_or(1024),
        1, // Loihi uses compartment-based neurons
        manifest.capabilities.as_ref().and_then(|c| c.weight_precisions.as_ref()).and_then(|v| v.iter().max().copied()).unwrap_or(8)
    );

    fs::write(out_dir.join("README.txt"), readme)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compile_smoke() {
        let g = nc_nir::Graph::new("g");
        let m = nc_hal::parse_target_manifest_str(r#"
            name = "loihi2"
            vendor = "Intel"
            family = "Loihi"
            version = "2"
            [capabilities]
            weight_precisions = [8]
            max_neurons_per_core = 1
            max_synapses_per_core = 1
            time_resolution_ns = 1
        "#).unwrap();
        let _ = compile(&g, &m).unwrap();
    }
}

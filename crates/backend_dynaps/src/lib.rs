use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use nc_passes::Pass;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;

/// Quantize weight for Dynap-SE mixed-signal precision
fn quantize_weight(w: f32, bits: u32) -> f32 {
    // Dynap-SE uses mixed-signal weights, typically 4-8 bit
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let max_val = (levels / 2) as f32;
    let scale = max_val / 1.0;

    let w_clamped = w.clamp(-1.0, 1.0);
    (w_clamped * scale).round() / scale
}

/// Dynap-SE-specific passes for mixed-signal neuromorphic processing
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// Dynap-SE chip mapping pass - assigns neurons to mixed-signal cores
    pub struct DynapsChipMappingPass;
    impl Pass for DynapsChipMappingPass {
        fn name(&self) -> &str { "dynaps-chip-mapping" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract Dynap-SE capabilities
            let caps = g.attributes.get("caps_dynaps");
            let neurons_per_core = caps.and_then(|v| v.get("neurons_per_core")).and_then(|x| x.as_u64()).unwrap_or(256);
            let cores_per_chip = caps.and_then(|v| v.get("cores_per_chip")).and_then(|x| x.as_u64()).unwrap_or(4);

            // Simple chip assignment - distribute populations across available cores
            let mut chip_mappings = Vec::new();
            let mut current_core = 0usize;

            for (_i, pop) in g.populations.iter().enumerate() {
                let cores_needed = ((pop.size as f64) / (neurons_per_core as f64)).ceil() as usize;

                for j in 0..cores_needed {
                    let core_id = (current_core + j) % cores_per_chip as usize;
                    chip_mappings.push(serde_json::json!({
                        "population": pop.name,
                        "core_id": core_id,
                        "chip_id": 0, // Single chip for now
                        "neuron_start": (j * neurons_per_core as usize) % neurons_per_core as usize,
                        "neurons_assigned": pop.size.min(neurons_per_core as u32)
                    }));
                }

                current_core = (current_core + cores_needed) % cores_per_chip as usize;
            }

            let meta = serde_json::json!({
                "total_neurons": g.populations.iter().map(|p| p.size as usize).sum::<usize>(),
                "cores_used": chip_mappings.len(),
                "chips_used": 1,
                "neurons_per_core": neurons_per_core,
                "cores_per_chip": cores_per_chip,
                "chip_mappings": chip_mappings
            });
            g.attributes.insert("dynaps_chip_mapping".to_string(), meta);
            Ok(g)
        }
    }

    /// Dynap-SE synapse programming pass - configures mixed-signal synapses
    pub struct DynapsSynapseProgrammingPass;
    impl Pass for DynapsSynapseProgrammingPass {
        fn name(&self) -> &str { "dynaps-synapse-programming" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            let bits = g.attributes.get("caps_dynaps")
                .and_then(|v| v.get("weight_bits"))
                .and_then(|x| x.as_u64())
                .unwrap_or(8) as u32;

            let mut programmed_synapses = Vec::new();
            for conn in &g.connections {
                let weight_q = quantize_weight(conn.weight, bits);

                programmed_synapses.push(serde_json::json!({
                    "pre_population": conn.pre,
                    "post_population": conn.post,
                    "weight": weight_q,
                    "synapse_type": "mixed_signal", // Dynap-SE uses mixed-signal synapses
                    "delay": conn.delay_ms,
                    "plasticity": conn.plasticity
                }));
            }

            let meta = serde_json::json!({
                "weight_bits": bits,
                "total_synapses": programmed_synapses.len(),
                "synapse_type": "mixed_signal",
                "programmed_synapses": programmed_synapses
            });
            g.attributes.insert("dynaps_synapse_programming".to_string(), meta);
            Ok(g)
        }
    }

    /// Dynap-SE learning configuration pass - sets up on-chip learning
    pub struct DynapsLearningConfigPass;
    impl Pass for DynapsLearningConfigPass {
        fn name(&self) -> &str { "dynaps-learning-config" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Configure on-chip learning for plastic synapses
            let plastic_connections: Vec<_> = g.connections.iter()
                .filter(|c| c.plasticity.is_some())
                .collect();

            let mut learning_configs = Vec::new();
            for conn in &plastic_connections {
                if let Some(plasticity) = &conn.plasticity {
                    learning_configs.push(serde_json::json!({
                        "connection": format!("{}->{}", conn.pre, conn.post),
                        "learning_rule": "stdp", // Dynap-SE supports STDP
                        "learning_rate": plasticity.params.get("learning_rate")
                            .and_then(|v| v.as_f64()).unwrap_or(0.01),
                        "tau_plus": plasticity.params.get("tau_plus")
                            .and_then(|v| v.as_f64()).unwrap_or(20.0),
                        "tau_minus": plasticity.params.get("tau_minus")
                            .and_then(|v| v.as_f64()).unwrap_or(20.0),
                        "on_chip": true // Dynap-SE has on-chip learning
                    }));
                }
            }

            let meta = serde_json::json!({
                "plastic_connections": plastic_connections.len(),
                "on_chip_learning": true,
                "learning_configs": learning_configs
            });
            g.attributes.insert("dynaps_learning_config".to_string(), meta);
            Ok(g)
        }
    }
}

/// Compile NIR to Dynap-SE hardware artifacts
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
            let labels = nc_telemetry::labels::backend(&graph.name, "dynaps", Some(&manifest.name));
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

    // Extract Dynap-SE capabilities
    let neurons_per_core = manifest.capabilities.as_ref()
        .and_then(|c| c.max_neurons_per_core)
        .unwrap_or(256) as usize;
    let cores_per_chip = 4; // Dynap-SE has 4 cores per chip
    let weight_bits = manifest.capabilities.as_ref()
        .and_then(|c| c.weight_precisions.as_ref())
        .and_then(|v| v.iter().max().copied())
        .unwrap_or(8) as usize;

    // Run Dynap-SE-specific pass pipeline
    let g = run_dynaps_pipeline(graph, neurons_per_core, cores_per_chip, weight_bits)?;

    // Generate hardware artifacts
    generate_dynaps_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = nc_telemetry::labels::backend(&graph.name, "dynaps", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    // Return serialized graph for testing
    Ok(serde_json::to_string_pretty(&g)?)
}

/// Run Dynap-SE-specific compilation pipeline
fn run_dynaps_pipeline(
    graph: &nc_nir::Graph,
    neurons_per_core: usize,
    cores_per_chip: usize,
    weight_bits: usize
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Attach Dynap-SE capabilities
    let caps = serde_json::json!({
        "neurons_per_core": neurons_per_core,
        "cores_per_chip": cores_per_chip,
        "weight_bits": weight_bits
    });
    g.attributes.insert("caps_dynaps".to_string(), caps);

    // Run Dynap-SE passes
    g = passes::DynapsChipMappingPass.run(g)?;
    g = passes::DynapsSynapseProgrammingPass.run(g)?;
    g = passes::DynapsLearningConfigPass.run(g)?;

    Ok(g)
}

/// Generate Dynap-SE hardware artifacts
fn generate_dynaps_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Generate chip configuration and synapse programming
    let chip_mapping = graph.attributes.get("dynaps_chip_mapping")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let synapse_programming = graph.attributes.get("dynaps_synapse_programming")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let learning_config = graph.attributes.get("dynaps_learning_config")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // Dynap-SE configuration
    let dynaps_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "platform": "Dynap-SE",
        "chip_mapping": chip_mapping,
        "synapse_programming": synapse_programming,
        "learning_config": learning_config
    });

    fs::write(out_dir.join("dynaps_config.json"), serde_json::to_string_pretty(&dynaps_config)?)?;

    // Generate README
    let readme = format!(
        "Dynap-SE Hardware Configuration for '{}'\n\
        ==========================================\n\
        Generated: {}\n\
        Target: {}\n\
        Platform: Dynap-SE\n\
        Populations: {}\n\
        Connections: {}\n\
        \nMixed-Signal Configuration:\n\
        - Neurons per core: {}\n\
        - Cores per chip: 4\n\
        - Weight precision: {} bits\n\
        - On-chip learning: {}\n",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        graph.populations.len(),
        graph.connections.len(),
        manifest.capabilities.as_ref().and_then(|c| c.max_neurons_per_core).unwrap_or(256),
        manifest.capabilities.as_ref().and_then(|c| c.weight_precisions.as_ref()).and_then(|v| v.iter().max().copied()).unwrap_or(8),
        manifest.capabilities.as_ref().and_then(|c| c.on_chip_learning).unwrap_or(false)
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
            name = "dynaps"
            vendor = "Generic"
            family = "Dynap-SE"
            version = "1"
            [capabilities]
            weight_precisions = [4,8]
            max_neurons_per_core = 1
            max_synapses_per_core = 1
            time_resolution_ns = 1
        "#).unwrap();
        let out = compile(&g, &m).expect("compile ok");
        assert!(out.contains("\"connections\""));
    }
}

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use nc_passes::Pass;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;

/// Quantize weight for Akida precision (typically 8-bit)
fn quantize_weight(w: f32, bits: u32) -> f32 {
    // Akida uses unsigned quantization for weights
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let max_val = (levels - 1) as f32;

    let w_clamped = w.clamp(0.0, 1.0); // Akida weights are typically unsigned
    (w_clamped * max_val).round() / max_val
}

/// Akida-specific passes for event-based neuromorphic processing
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// Akida layer mapping pass - assigns neural populations to Akida layers
    pub struct AkidaLayerMappingPass;
    impl Pass for AkidaLayerMappingPass {
        fn name(&self) -> &str { "akida-layer-mapping" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract Akida capabilities
            let caps = g.attributes.get("caps_akida");
            let max_layers = caps.and_then(|v| v.get("max_layers")).and_then(|x| x.as_u64()).unwrap_or(8);
            let neurons_per_layer = caps.and_then(|v| v.get("neurons_per_layer")).and_then(|x| x.as_u64()).unwrap_or(65536);

            // Simple layer assignment - each population gets its own layer
            let mut layer_mappings = Vec::new();
            for (i, pop) in g.populations.iter().enumerate() {
                if i >= max_layers as usize {
                    return Err(anyhow::anyhow!("Too many populations for Akida layer limit"));
                }

                layer_mappings.push(serde_json::json!({
                    "population": pop.name,
                    "layer_id": i,
                    "layer_type": "spiking_conv2d", // Default to convolutional
                    "neurons_assigned": pop.size.min(neurons_per_layer as u32)
                }));
            }

            let meta = serde_json::json!({
                "total_layers": layer_mappings.len(),
                "max_layers": max_layers,
                "neurons_per_layer": neurons_per_layer,
                "layer_mappings": layer_mappings
            });
            g.attributes.insert("akida_layer_mapping".to_string(), meta);
            Ok(g)
        }
    }

    /// Akida weight programming pass - converts weights for Akida's event-based processing
    pub struct AkidaWeightProgrammingPass;
    impl Pass for AkidaWeightProgrammingPass {
        fn name(&self) -> &str { "akida-weight-programming" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            let bits = g.attributes.get("caps_akida")
                .and_then(|v| v.get("weight_bits"))
                .and_then(|x| x.as_u64())
                .unwrap_or(8) as u32;

            let mut programmed_weights = Vec::new();
            for conn in &g.connections {
                let weight_q = quantize_weight(conn.weight.abs(), bits); // Akida uses absolute weights

                programmed_weights.push(serde_json::json!({
                    "pre_population": conn.pre,
                    "post_population": conn.post,
                    "weight": weight_q,
                    "threshold": 1.0, // Default spiking threshold
                    "delay": conn.delay_ms,
                    "plasticity": conn.plasticity
                }));
            }

            let meta = serde_json::json!({
                "weight_bits": bits,
                "total_weights": programmed_weights.len(),
                "programmed_weights": programmed_weights
            });
            g.attributes.insert("akida_weight_programming".to_string(), meta);
            Ok(g)
        }
    }

    /// Akida event routing pass - configures event-based communication
    pub struct AkidaEventRoutingPass;
    impl Pass for AkidaEventRoutingPass {
        fn name(&self) -> &str { "akida-event-routing" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Configure event routing between layers
            let layer_mappings = g.attributes.get("akida_layer_mapping")
                .and_then(|v| v.get("layer_mappings"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let mut event_routes = Vec::new();
            for conn in &g.connections {
                // Find source and target layers
                let source_layer = layer_mappings.iter()
                    .find(|m| m.get("population").and_then(|p| p.as_str()) == Some(&conn.pre));
                let target_layer = layer_mappings.iter()
                    .find(|m| m.get("population").and_then(|p| p.as_str()) == Some(&conn.post));

                if let (Some(src), Some(tgt)) = (source_layer, target_layer) {
                    let src_id = src.get("layer_id").and_then(|id| id.as_u64()).unwrap_or(0);
                    let tgt_id = tgt.get("layer_id").and_then(|id| id.as_u64()).unwrap_or(0);

                    event_routes.push(serde_json::json!({
                        "source_layer": src_id,
                        "target_layer": tgt_id,
                        "connection": format!("{}->{}", conn.pre, conn.post),
                        "event_mode": "spike"
                    }));
                }
            }

            let meta = serde_json::json!({
                "event_routes": event_routes,
                "routing_mode": "direct_layer"
            });
            g.attributes.insert("akida_event_routing".to_string(), meta);
            Ok(g)
        }
    }
}

/// Compile NIR to Akida hardware artifacts
pub fn compile(graph: &nc_nir::Graph, manifest: &nc_hal::TargetManifest) -> Result<String> {
    // Validate input IR and target manifest
    graph.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    nc_hal::validate_manifest(manifest)?;

    // Optional telemetry profiling
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let mut labels = BTreeMap::new();
            labels.insert("backend".to_string(), "akida".to_string());
            labels.insert("target".to_string(), manifest.name.clone());
            labels.insert("graph".to_string(), graph.name.clone());
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

    // Extract Akida capabilities
    let max_layers = manifest.capabilities.as_ref()
        .and_then(|c| c.max_neurons_per_core)
        .unwrap_or(8) as usize; // Akida has limited layers
    let neurons_per_layer = manifest.capabilities.as_ref()
        .and_then(|c| c.max_synapses_per_core)
        .unwrap_or(65536) as usize;
    let weight_bits = manifest.capabilities.as_ref()
        .and_then(|c| c.weight_precisions.as_ref())
        .and_then(|v| v.iter().max().copied())
        .unwrap_or(8) as usize;

    // Run Akida-specific pass pipeline
    let g = run_akida_pipeline(graph, max_layers, neurons_per_layer, weight_bits)?;

    // Generate hardware artifacts
    generate_akida_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = telemetry::labels::backend(&graph.name, "akida", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    Ok(format!("artifact:{}", out_dir.display()))
}

/// Run Akida-specific compilation pipeline
fn run_akida_pipeline(
    graph: &nc_nir::Graph,
    max_layers: usize,
    neurons_per_layer: usize,
    weight_bits: usize
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Attach Akida capabilities
    let caps = serde_json::json!({
        "max_layers": max_layers,
        "neurons_per_layer": neurons_per_layer,
        "weight_bits": weight_bits
    });
    g.attributes.insert("caps_akida".to_string(), caps);

    // Run Akida passes
    g = passes::AkidaLayerMappingPass.run(g)?;
    g = passes::AkidaWeightProgrammingPass.run(g)?;
    g = passes::AkidaEventRoutingPass.run(g)?;

    Ok(g)
}

/// Generate Akida hardware artifacts
fn generate_akida_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Generate layer configuration and event routing
    let layer_mapping = graph.attributes.get("akida_layer_mapping")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let weight_programming = graph.attributes.get("akida_weight_programming")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let event_routing = graph.attributes.get("akida_event_routing")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // Akida configuration
    let akida_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "platform": "Akida",
        "layer_mapping": layer_mapping,
        "weight_programming": weight_programming,
        "event_routing": event_routing
    });

    fs::write(out_dir.join("akida_config.json"), serde_json::to_string_pretty(&akida_config)?)?;

    // Generate README
    let readme = format!(
        "Akida Hardware Configuration for '{}'\n\
        =====================================\n\
        Generated: {}\n\
        Target: {}\n\
        Platform: Akida\n\
        Populations: {}\n\
        Connections: {}\n\
        \nLayer Configuration:\n\
        - Max layers: {}\n\
        - Neurons per layer: {}\n\
        - Weight precision: {} bits\n\
        - Event-based processing: Enabled\n",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        graph.populations.len(),
        graph.connections.len(),
        manifest.capabilities.as_ref().and_then(|c| c.max_neurons_per_core).unwrap_or(8),
        manifest.capabilities.as_ref().and_then(|c| c.max_synapses_per_core).unwrap_or(65536),
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
            name = "akida"
            vendor = "BrainChip"
            family = "Akida"
            version = "1"
            [capabilities]
            weight_precisions = [8]
            max_neurons_per_core = 1
            max_synapses_per_core = 1
            time_resolution_ns = 1
        "#).unwrap();
        let out = compile(&g, &m).expect("compile ok");
        assert!(out.starts_with("artifact:"));
    }
}

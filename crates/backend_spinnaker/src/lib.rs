use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;
use nc_passes::Pass;
use serde_json::json;

/// Quantize weight to specified precision for SpiNNaker
fn quantize_weight(w: f32, bits: u32) -> f32 {
    // Uniform symmetric quantization onto [-1,1] with 2^bits levels
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let l_minus_1 = (levels.saturating_sub(1)) as f32;
    let l_minus_1 = if l_minus_1 <= 0.0 { 1.0 } else { l_minus_1 };
    let w_clamped = w.clamp(-1.0, 1.0);
    let step = 2.0 / l_minus_1;
    ((w_clamped + 1.0) / step).round() * step - 1.0
}

/// SpiNNaker-specific passes for AER protocol and multicore routing
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// SpiNNaker core allocation pass - assigns neurons to SpiNNaker chips and cores
    pub struct SnCoreAllocationPass;
    impl Pass for SnCoreAllocationPass {
        fn name(&self) -> &str { "sn-core-allocation" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract SpiNNaker capabilities
            let caps = g.attributes.get("caps_spinnaker");
            let neurons_per_core = caps.and_then(|v| v.get("neurons_per_core")).and_then(|x| x.as_u64()).unwrap_or(1000);
            let cores_per_chip = caps.and_then(|v| v.get("cores_per_chip")).and_then(|x| x.as_u64()).unwrap_or(18);
            let chips_available = caps.and_then(|v| v.get("chips_available")).and_then(|x| x.as_u64()).unwrap_or(48);

            // Calculate total capacity
            let total_cores = chips_available * cores_per_chip;
            let total_neurons_capacity = total_cores * neurons_per_core;

            let total_neurons: usize = g.populations.iter().map(|p| p.size as usize).sum();

            // Simple allocation strategy: distribute populations across cores
            let mut core_allocations = Vec::new();
            let mut current_core = 0usize;
            let mut current_chip = 0usize;

            for pop in &g.populations {
                let cores_needed = ((pop.size as f64) / (neurons_per_core as f64)).ceil() as usize;
                let allocated_cores = (0..cores_needed).map(|i| {
                    let chip_id = current_chip;
                    let core_id = current_core + i;
                    if core_id >= cores_per_chip as usize {
                        // Move to next chip
                        current_chip += 1;
                        current_core = 0;
                    }
                    json!({
                        "chip_id": chip_id,
                        "core_id": core_id,
                        "population": pop.name,
                        "allocated_neurons": pop.size
                    })
                }).collect::<Vec<_>>();

                core_allocations.extend(allocated_cores);
                current_core += cores_needed;
            }

            let meta = serde_json::json!({
                "total_neurons": total_neurons,
                "total_capacity": total_neurons_capacity,
                "cores_allocated": core_allocations.len(),
                "chips_used": current_chip + 1,
                "core_allocations": core_allocations
            });
            g.attributes.insert("sn_core_allocation".to_string(), meta);
            Ok(g)
        }
    }

    /// SpiNNaker AER routing pass - generates Address-Event Representation routing tables
    pub struct SnAerRoutingPass;
    impl Pass for SnAerRoutingPass {
        fn name(&self) -> &str { "sn-aer-routing" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Generate AER routing tables based on core allocations
            let core_allocations = g.attributes.get("sn_core_allocation")
                .and_then(|v| v.get("core_allocations"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let mut routing_tables = Vec::new();
            let mut aer_keys = std::collections::HashMap::new();

            // Assign AER keys to each population (neuron ID ranges)
            let mut current_key = 0u64;
            for alloc in &core_allocations {
                if let (Some(chip_id), Some(core_id), Some(pop_name), Some(neuron_count)) = (
                    alloc.get("chip_id").and_then(|v| v.as_u64()),
                    alloc.get("core_id").and_then(|v| v.as_u64()),
                    alloc.get("population").and_then(|v| v.as_str()),
                    alloc.get("allocated_neurons").and_then(|v| v.as_u64())
                ) {
                    let key_range = (current_key..current_key + neuron_count).collect::<Vec<_>>();
                    aer_keys.insert(pop_name.to_string(), key_range.clone());

                    routing_tables.push(serde_json::json!({
                        "chip_id": chip_id,
                        "core_id": core_id,
                        "population": pop_name,
                        "aer_key_range": format!("{:#010x}..{:#010x}", current_key, current_key + neuron_count),
                        "neuron_count": neuron_count
                    }));

                    current_key += neuron_count;
                }
            }

            // Generate inter-chip routing for connections
            let mut inter_chip_routes = Vec::new();
            for conn in &g.connections {
                let pre_keys = aer_keys.get(&conn.pre);
                let post_keys = aer_keys.get(&conn.post);

                if let (Some(pre_range), Some(post_range)) = (pre_keys, post_keys) {
                    // Simplified routing: direct AER packet forwarding
                    inter_chip_routes.push(serde_json::json!({
                        "source_population": conn.pre,
                        "target_population": conn.post,
                        "source_keys": pre_range,
                        "target_keys": post_range,
                        "route_type": "direct_aer"
                    }));
                }
            }

            let meta = serde_json::json!({
                "routing_tables": routing_tables,
                "inter_chip_routes": inter_chip_routes,
                "aer_key_space": format!("{:#010x}", current_key)
            });
            g.attributes.insert("sn_aer_routing".to_string(), meta);
            Ok(g)
        }
    }

    /// SpiNNaker synapse programming pass
    pub struct SnSynapseProgrammingPass;
    impl Pass for SnSynapseProgrammingPass {
        fn name(&self) -> &str { "sn-synapse-programming" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            let bits = g.attributes.get("caps_spinnaker")
                .and_then(|v| v.get("weight_bits"))
                .and_then(|x| x.as_u64())
                .unwrap_or(16) as u32;

            let mut synapse_configs = Vec::new();
            for conn in &g.connections {
                // SpiNNaker uses 16-bit weights by default
                let weight_q = if bits <= 8 {
                    quantize_weight(conn.weight, bits)
                } else {
                    // For 16-bit, use the weight directly (SpiNNaker supports floating point synapses)
                    conn.weight
                };

                synapse_configs.push(serde_json::json!({
                    "pre_population": conn.pre,
                    "post_population": conn.post,
                    "weight": weight_q,
                    "delay": conn.delay_ms,
                    "plasticity": conn.plasticity
                }));
            }

            let meta = serde_json::json!({
                "weight_bits": bits,
                "total_synapses": synapse_configs.len(),
                "synapse_configs": synapse_configs
            });
            g.attributes.insert("sn_synapse_programming".to_string(), meta);
            Ok(g)
        }
    }
}

/// Compile NIR to SpiNNaker artifacts
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
            let labels = telemetry::labels::backend(&graph.name, "spinnaker", Some(&manifest.name));
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

    // Extract SpiNNaker capabilities
    let neurons_per_core = manifest.capabilities.as_ref()
        .and_then(|c| c.max_neurons_per_core)
        .unwrap_or(1000) as usize;
    let cores_per_chip = manifest.capabilities.as_ref()
        .and_then(|c| c.max_synapses_per_core)
        .map(|v| v / 1000) // Estimate cores per chip
        .unwrap_or(18) as usize;
    let weight_bits = manifest.capabilities.as_ref()
        .and_then(|c| c.weight_precisions.as_ref())
        .and_then(|v| v.iter().max().copied())
        .unwrap_or(16) as usize;

    // Run SpiNNaker-specific pass pipeline
    let g = run_spinnaker_pipeline(graph, neurons_per_core, cores_per_chip, weight_bits)?;

    // Generate hardware artifacts
    generate_spinnaker_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = telemetry::labels::backend(&graph.name, "spinnaker", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    Ok(format!("artifact:{}", out_dir.display()))
}

/// Run SpiNNaker-specific compilation pipeline
fn run_spinnaker_pipeline(
    graph: &nc_nir::Graph,
    neurons_per_core: usize,
    cores_per_chip: usize,
    weight_bits: usize
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Attach SpiNNaker capabilities
    let caps = serde_json::json!({
        "neurons_per_core": neurons_per_core,
        "cores_per_chip": cores_per_chip,
        "chips_available": 48, // SpiNNaker 48-chip board
        "weight_bits": weight_bits
    });
    g.attributes.insert("caps_spinnaker".to_string(), caps);

    // Run SpiNNaker passes
    g = passes::SnCoreAllocationPass.run(g)?;
    g = passes::SnAerRoutingPass.run(g)?;
    g = passes::SnSynapseProgrammingPass.run(g)?;

    Ok(g)
}

/// Generate SpiNNaker hardware artifacts
fn generate_spinnaker_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Generate core allocation and routing configuration
    let core_allocation = graph.attributes.get("sn_core_allocation")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let aer_routing = graph.attributes.get("sn_aer_routing")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let synapse_programming = graph.attributes.get("sn_synapse_programming")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // SpiNNaker configuration
    let spinnaker_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "platform": "SpiNNaker2",
        "core_allocation": core_allocation,
        "aer_routing": aer_routing,
        "synapse_programming": synapse_programming
    });

    fs::write(out_dir.join("spinnaker_config.json"), serde_json::to_string_pretty(&spinnaker_config)?)?;

    // Generate README
    let readme = format!(
        "SpiNNaker Hardware Configuration for '{}'\n\
        ==========================================\n\
        Generated: {}\n\
        Target: {}\n\
        Platform: SpiNNaker 2\n\
        Populations: {}\n\
        Connections: {}\n\
        \nSpiNNaker Mapping:\n\
        - Neurons per core: {}\n\
        - Cores per chip: {}\n\
        - AER routing: Enabled\n\
        - Synapse precision: {} bits\n",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        graph.populations.len(),
        graph.connections.len(),
        manifest.capabilities.as_ref().and_then(|c| c.max_neurons_per_core).unwrap_or(1000),
        18, // SpiNNaker 2 cores per chip
        manifest.capabilities.as_ref().and_then(|c| c.weight_precisions.as_ref()).and_then(|v| v.iter().max().copied()).unwrap_or(16)
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
            name = "spinnaker"
            vendor = "Generic"
            family = "SpiNNaker"
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

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use nc_passes::Pass;

/// Quantize weight to TrueNorth precision (4-bit by default)
fn quantize_weight(w: f32, bits: u32) -> f32 {
    // Map [-1, 1] to (L) levels with step = 2/(L-1)
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let l_minus_1 = (levels.saturating_sub(1)) as f32;
    let l_minus_1 = if l_minus_1 <= 0.0 { 1.0 } else { l_minus_1 };
    let w_clamped = w.clamp(-1.0, 1.0);
    let step = 2.0 / l_minus_1;
    ((w_clamped + 1.0) / step).round() * step - 1.0
}

/// Convert quantized weight to TrueNorth core format (axon index + weight value)
fn weight_to_core_format(weight_q: f32, bits: u32) -> (u8, u8) {
    // TrueNorth uses 4-bit weights encoded as axon index and weight value
    let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
    let l_minus_1 = (levels.saturating_sub(1)) as f32;

    // Map [-1, 1] to [0, levels-1]
    let index = ((weight_q + 1.0) * l_minus_1 / 2.0).round() as u8;
    let axon_idx = (index / 16) as u8;  // 16 weights per axon
    let weight_val = (index % 16) as u8;

    (axon_idx, weight_val)
}

/// TrueNorth-specific passes for hardware-specific optimizations
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// TrueNorth core mapping pass - assigns neurons to cores and axons
    pub struct TnCoreMappingPass;
    impl Pass for TnCoreMappingPass {
        fn name(&self) -> &str { "tn-core-mapping" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract TrueNorth capabilities
            let caps = g.attributes.get("caps_truenorth");
            let neurons_per_core = caps.and_then(|v| v.get("neurons_per_core")).and_then(|x| x.as_u64()).unwrap_or(256);
            let axons_per_core = caps.and_then(|v| v.get("axons_per_core")).and_then(|x| x.as_u64()).unwrap_or(256);

            // Simple core assignment (first-fit)
            let total_neurons: usize = g.populations.iter().map(|p| p.size as usize).sum();
            let cores_needed = ((total_neurons as f64) / (neurons_per_core as f64)).ceil() as usize;

            let mut core_assignments = Vec::new();
            for (i, pop) in g.populations.iter().enumerate() {
                let core_id = i % cores_needed;
                core_assignments.push(serde_json::json!({
                    "population": pop.name,
                    "core_id": core_id,
                    "axon_start": core_id * axons_per_core as usize,
                    "neuron_start": (i * pop.size as usize) % neurons_per_core as usize
                }));
            }

            let meta = serde_json::json!({
                "cores_used": cores_needed,
                "neurons_per_core": neurons_per_core,
                "axons_per_core": axons_per_core,
                "core_assignments": core_assignments
            });
            g.attributes.insert("tn_core_mapping".to_string(), meta);
            Ok(g)
        }
    }

    /// TrueNorth weight programming pass - converts weights to core format
    pub struct TnWeightProgrammingPass;
    impl Pass for TnWeightProgrammingPass {
        fn name(&self) -> &str { "tn-weight-programming" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            let bits = g.attributes.get("caps_truenorth")
                .and_then(|v| v.get("weight_bits"))
                .and_then(|x| x.as_u64())
                .unwrap_or(4) as u32;

            let mut programmed_weights = Vec::new();
            for conn in &g.connections {
                let weight_q = quantize_weight(conn.weight, bits);
                let (axon_idx, weight_val) = weight_to_core_format(weight_q, bits);

                programmed_weights.push(serde_json::json!({
                    "pre_population": conn.pre,
                    "post_population": conn.post,
                    "axon_index": axon_idx,
                    "weight_value": weight_val,
                    "original_weight": conn.weight,
                    "quantized_weight": weight_q
                }));
            }

            let meta = serde_json::json!({
                "weight_bits": bits,
                "programmed_weights": programmed_weights
            });
            g.attributes.insert("tn_weight_programming".to_string(), meta);
            Ok(g)
        }
    }

    /// TrueNorth crossbar configuration pass
    pub struct TnCrossbarConfigPass;
    impl Pass for TnCrossbarConfigPass {
        fn name(&self) -> &str { "tn-crossbar-config" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Configure crossbar routing based on core assignments
            let core_assignments = g.attributes.get("tn_core_mapping")
                .and_then(|v| v.get("core_assignments"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let mut crossbar_config = Vec::new();
            for assignment in core_assignments {
                if let (Some(pop_name), Some(core_id)) = (
                    assignment.get("population").and_then(|v| v.as_str()),
                    assignment.get("core_id").and_then(|v| v.as_u64())
                ) {
                    crossbar_config.push(serde_json::json!({
                        "population": pop_name,
                        "core_id": core_id,
                        "crossbar_enabled": true,
                        "routing_mode": "direct"
                    }));
                }
            }

            let meta = serde_json::json!({
                "crossbar_config": crossbar_config,
                "routing_algorithm": "direct_mapping"
            });
            g.attributes.insert("tn_crossbar_config".to_string(), meta);
            Ok(g)
        }
    }
}

/// Compile NIR to TrueNorth hardware artifacts
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
            let labels = nc_telemetry::labels::backend(&graph.name, "truenorth", Some(&manifest.name));
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

    // Extract TrueNorth capabilities
    let neurons_per_core = manifest.capabilities.as_ref()
        .and_then(|c| c.max_neurons_per_core)
        .unwrap_or(256) as usize;
    let axons_per_core = manifest.capabilities.as_ref()
        .and_then(|c| c.max_synapses_per_core)
        .map(|v| v / 256) // Approximate axons per core
        .unwrap_or(256) as usize;
    let weight_bits = manifest.capabilities.as_ref()
        .and_then(|c| c.weight_precisions.as_ref())
        .and_then(|v| v.iter().max().copied())
        .unwrap_or(4) as usize;

    // Run TrueNorth-specific pass pipeline
    let g = run_truenorth_pipeline(graph, neurons_per_core, axons_per_core, weight_bits)?;

    // Generate hardware artifacts
    generate_truenorth_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = nc_telemetry::labels::backend(&graph.name, "truenorth", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    Ok(format!("artifact:{}", out_dir.display()))
}

/// Run TrueNorth-specific compilation pipeline
fn run_truenorth_pipeline(
    graph: &nc_nir::Graph,
    neurons_per_core: usize,
    axons_per_core: usize,
    weight_bits: usize
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Attach TrueNorth capabilities
    let caps = serde_json::json!({
        "neurons_per_core": neurons_per_core,
        "axons_per_core": axons_per_core,
        "weight_bits": weight_bits
    });
    g.attributes.insert("caps_truenorth".to_string(), caps);

    // Run TrueNorth passes
    g = passes::TnCoreMappingPass.run(g)?;
    g = passes::TnWeightProgrammingPass.run(g)?;
    g = passes::TnCrossbarConfigPass.run(g)?;

    Ok(g)
}

/// Generate TrueNorth hardware artifacts
fn generate_truenorth_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Generate core programming files
    let core_assignments = graph.attributes.get("tn_core_mapping")
        .and_then(|v| v.get("core_assignments"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let programmed_weights = graph.attributes.get("tn_weight_programming")
        .and_then(|v| v.get("programmed_weights"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Core configuration
    let core_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "cores": core_assignments,
        "weights": programmed_weights
    });

    fs::write(out_dir.join("truenorth_config.json"), serde_json::to_string_pretty(&core_config)?)?;

    // Generate README
    let readme = format!(
        "TrueNorth Hardware Configuration for '{}'\n\
        ==========================================\n\
        Generated: {}\n\
        Target: {}\n\
        Cores Used: {}\n\
        Total Populations: {}\n\
        Total Connections: {}\n\
        \nHardware Mapping:\n\
        - Neurons per core: {}\n\
        - Axons per core: {}\n\
        - Weight precision: {} bits\n",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        core_assignments.len(),
        graph.populations.len(),
        graph.connections.len(),
        manifest.capabilities.as_ref().and_then(|c| c.max_neurons_per_core).unwrap_or(256),
        manifest.capabilities.as_ref().and_then(|c| c.max_synapses_per_core).map(|v| v / 256).unwrap_or(256),
        manifest.capabilities.as_ref().and_then(|c| c.weight_precisions.as_ref()).and_then(|v| v.iter().max().copied()).unwrap_or(4)
    );

    fs::write(out_dir.join("README.txt"), readme)?;

    Ok(())
}

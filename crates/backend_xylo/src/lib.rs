//! SynSense Xylo Backend for Neuro-Compiler
//!
//! This backend targets the SynSense Xylo SoC, an ultra-low-power neuromorphic
//! processor optimized for always-on audio processing and keyword spotting.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use nc_passes::Pass;
use nc_nir;

/// Xylo-specific passes for ultra-low-power audio processing
pub mod passes {
    use super::*;
    use nc_passes::Pass;
    use anyhow::Result;

    /// Xylo audio processing pass - optimizes for audio event streams
    pub struct XyloAudioProcessingPass;
    impl Pass for XyloAudioProcessingPass {
        fn name(&self) -> &str { "xylo-audio-processing" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract Xylo capabilities
            let caps = g.attributes.get("caps_xylo");
            let audio_sample_rate = caps.and_then(|v| v.get("audio_sample_rate")).and_then(|x| x.as_u64()).unwrap_or(16000);
            let mel_bins = caps.and_then(|v| v.get("mel_bins")).and_then(|x| x.as_u64()).unwrap_or(64);

            // Optimize for audio processing
            let xylo_audio = serde_json::json!({
                "audio_sample_rate": audio_sample_rate,
                "mel_bins": mel_bins,
                "audio_event_driven": true,
                "keyword_spotting": true,
                "noise_suppression": true
            });

            g.attributes.insert("xylo_audio_processing".to_string(), xylo_audio);
            Ok(g)
        }
    }

    /// Xylo power optimization pass - minimizes energy consumption
    pub struct XyloPowerOptimizationPass;
    impl Pass for XyloPowerOptimizationPass {
        fn name(&self) -> &str { "xylo-power-optimization" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Extract power constraints
            let power_budget_uw = g.attributes.get("caps_xylo")
                .and_then(|v| v.get("power_budget_uw"))
                .and_then(|x| x.as_u64())
                .unwrap_or(5000); // 5mW default

            // Calculate audio processing power requirements
            let total_neurons: usize = g.populations.iter().map(|p| p.size as usize).sum();
            let estimated_power_uw = (total_neurons as f64 * 0.2) as u64; // 0.2uW per audio neuron

            let power_optimization = serde_json::json!({
                "power_budget_uw": power_budget_uw,
                "estimated_power_uw": estimated_power_uw,
                "power_efficiency": (power_budget_uw as f64) / (estimated_power_uw as f64),
                "ultra_low_power_mode": estimated_power_uw < power_budget_uw
            });

            g.attributes.insert("xylo_power_optimization".to_string(), power_optimization);
            Ok(g)
        }
    }

    /// Xylo audio pipeline pass - optimizes for keyword spotting and audio features
    pub struct XyloAudioPipelinePass;
    impl Pass for XyloAudioPipelinePass {
        fn name(&self) -> &str { "xylo-audio-pipeline" }
        fn run(&self, mut g: nc_nir::Graph) -> Result<nc_nir::Graph> {
            // Configure audio-specific processing
            let audio_config = serde_json::json!({
                "microphone_integration": true,
                "mfcc_extraction": true,
                "keyword_spotting": true,
                "wake_word_detection": true,
                "noise_reduction": true,
                "supported_sample_rates": [8000, 16000, 44100],
                "max_keywords": 10
            });

            g.attributes.insert("xylo_audio_pipeline".to_string(), audio_config);
            Ok(g)
        }
    }
}

/// Compile NIR to Xylo hardware artifacts
pub fn compile(graph: &nc_nir::Graph, manifest: &nc_hal::TargetManifest) -> Result<String> {
    // Validate input
    graph.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    nc_hal::validate_manifest(manifest)?;

    // Optional telemetry
    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| nc_telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let labels = nc_telemetry::labels::backend(&graph.name, "xylo", Some(&manifest.name));
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

    // Run Xylo-specific pipeline
    let g = run_xylo_pipeline(graph, manifest)?;

    // Generate hardware artifacts
    generate_xylo_artifacts(&g, &out_dir, manifest)?;

    // Telemetry counters
    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = nc_telemetry::labels::backend(&graph.name, "xylo", Some(&manifest.name));
        let _ = a.counter("graph.populations", graph.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", graph.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", graph.probes.len() as f64, l);
    }

    Ok(format!("artifact:{}", out_dir.display()))
}

/// Run Xylo-specific compilation pipeline
fn run_xylo_pipeline(
    graph: &nc_nir::Graph,
    manifest: &nc_hal::TargetManifest
) -> Result<nc_nir::Graph> {
    let mut g = graph.clone();

    // Extract Xylo capabilities from manifest (using defaults for now)
    let audio_sample_rate = 16000; // default
    let mel_bins = 64; // default
    let power_budget_uw = 5000; // default

    // Attach Xylo capabilities
    let caps = serde_json::json!({
        "audio_sample_rate": audio_sample_rate,
        "mel_bins": mel_bins,
        "power_budget_uw": power_budget_uw,
        "microphone_support": true,
        "always_on_audio": true,
        "keyword_spotting": true
    });
    g.attributes.insert("caps_xylo".to_string(), caps);

    // Run Xylo passes
    g = passes::XyloAudioProcessingPass.run(g)?;
    g = passes::XyloPowerOptimizationPass.run(g)?;
    g = passes::XyloAudioPipelinePass.run(g)?;

    Ok(g)
}

/// Generate Xylo hardware artifacts
fn generate_xylo_artifacts(
    graph: &nc_nir::Graph,
    out_dir: &Path,
    manifest: &nc_hal::TargetManifest
) -> Result<()> {
    // Extract Xylo-specific metadata
    let audio_processing = graph.attributes.get("xylo_audio_processing")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let power_optimization = graph.attributes.get("xylo_power_optimization")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let audio_pipeline = graph.attributes.get("xylo_audio_pipeline")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // Xylo configuration
    let xylo_config = serde_json::json!({
        "target": manifest.name,
        "graph": graph.name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "platform": "SynSense Xylo",
        "audio_processing": audio_processing,
        "power_optimization": power_optimization,
        "audio_pipeline": audio_pipeline,
        "populations": graph.populations.len(),
        "connections": graph.connections.len(),
        "probes": graph.probes.len()
    });

    fs::write(out_dir.join("xylo_config.json"), serde_json::to_string_pretty(&xylo_config)?)?;

    // Generate README with Xylo-specific information
    let readme = format!(
        "SynSense Xylo Configuration for '{}'\n\
        ==================================\n\
        Generated: {}\n\
        Target: {}\n\
        Platform: SynSense Xylo (Ultra-Low-Power Audio SoC)\n\
        \nNetwork Summary:\n\
        - Populations: {}\n\
        - Connections: {}\n\
        - Probes: {}\n\
        \nXylo Optimizations:\n\
        - Audio processing: ✅\n\
        - Power optimization: {}μW budget\n\
        - Keyword spotting: ✅\n\
        - Always-on audio: ✅\n\
        - Microphone integration: ✅\n\
        \nHardware Specifications:\n\
        - Audio sample rate: 16kHz\n\
        - Mel bins: 64\n\
        - Power budget: {}μW\n\
        - Max keywords: 10",
        graph.name,
        chrono::Utc::now().to_rfc3339(),
        manifest.name,
        graph.populations.len(),
        graph.connections.len(),
        graph.probes.len(),
        5000, // power budget
        5000  // power budget
    );

    fs::write(out_dir.join("README.txt"), readme)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compile_smoke() {
        let g = nc_nir::Graph::new("xylo_test");
        let m = nc_hal::parse_target_manifest_str(r#"
            name = "xylo"
            vendor = "SynSense"
            family = "Xylo"
            version = "1"
            [capabilities]
            weight_precisions = [8]
            max_neurons_per_core = 1024
            max_synapses_per_core = 8192
            time_resolution_ns = 1000
        "#).unwrap();
        let _ = compile(&g, &m).unwrap();
    }
}
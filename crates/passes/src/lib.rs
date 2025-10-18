use anyhow::{bail, Result};
pub use nc_nir as nir;
use nc_hal as hal;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
#[cfg(feature = "telemetry")]
use std::collections::BTreeMap;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;
use nc_orchestrator as orchestrator;

pub mod lower_to_kernels;
pub mod memory_layout_and_quant;
pub mod kernel_fusion_and_scheduling;
pub mod validation;
pub mod eir_validate;

#[derive(Debug, Error)]
pub enum PassError {
    #[error("mapping violation: {0}")]
    Mapping(&'static str),
}

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, g: nir::Graph) -> Result<nir::Graph>;
}

pub struct NoOpPass;
impl Pass for NoOpPass {
    fn name(&self) -> &str { "no-op" }
    fn run(&self, g: nir::Graph) -> Result<nir::Graph> { Ok(g) }
}

pub struct ValidatePass;
impl Pass for ValidatePass {
    fn name(&self) -> &str { "validate" }
    fn run(&self, g: nir::Graph) -> Result<nir::Graph> {
        g.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(g)
    }
}

pub struct QuantizeWeightsPass {
    pub bits: u32,
}

impl QuantizeWeightsPass {
    fn quantize(w: f32, bits: u32) -> f32 {
        // Uniform symmetric quantization onto [-1,1] with 2^bits levels
        let levels: u32 = if bits >= 31 { u32::MAX } else { 1u32 << bits };
        let l_minus_1 = (levels.saturating_sub(1)) as f32;
        let l_minus_1 = if l_minus_1 <= 0.0 { 1.0 } else { l_minus_1 };
        let w_clamped = w.clamp(-1.0, 1.0);
        let step = 2.0 / l_minus_1;
        ((w_clamped + 1.0) / step).round() * step - 1.0
    }
}

impl Pass for QuantizeWeightsPass {
    fn name(&self) -> &str { "quantize" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        for c in &mut g.connections {
            c.weight = Self::quantize(c.weight, self.bits);
        }
        Ok(g)
    }
}

fn extract_caps_from_graph(g: &nir::Graph) -> Option<hal::Capabilities> {
    if let Some(p) = g.attributes.get("hal_manifest_path").and_then(|v| v.as_str()) {
        if let Ok(m) = hal::parse_target_manifest_path(p) {
            return m.capabilities.clone();
        }
    }
    None
}

pub struct PartitionPass;
impl Pass for PartitionPass {
    fn name(&self) -> &str { "partition" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let mut strategy = "naive";
        let mut parts: usize = 1;
        let mut assignment: Vec<(String, usize)> = Vec::new();
        let mut violations: Vec<serde_json::Value> = Vec::new();

        // Orchestrator plan (serialized for handoff)
        let targets_vec: Vec<String> = g.attributes
            .get("hal_manifest_path")
            .and_then(|v| v.as_str())
            .and_then(|p| Path::new(p).file_stem().and_then(|s| s.to_str()))
            .map(|s| vec![s.to_string()])
            .unwrap_or_else(|| Vec::new());
        let target_slices: Vec<&str> = targets_vec.iter().map(|s| s.as_str()).collect();
        if let Ok(plan) = orchestrator::partition(&g, &target_slices) {
            g.attributes.insert("orchestrator_plan".to_string(), serde_json::json!({
                "parts": plan.parts,
                "targets": target_slices,
            }));
        }

        if let Some(caps) = extract_caps_from_graph(&g) {
            strategy = "cap-aware";
            let max_neurons = caps.max_neurons_per_core.unwrap_or(0) as usize;
            let max_syn = caps.max_synapses_per_core.unwrap_or(0) as usize;

            let total_neurons: usize = g.populations.iter().map(|p| p.size as usize).sum();
            let total_synapses: usize = g.connections.len();

            let parts_by_neurons = if max_neurons > 0 { total_neurons.div_ceil(max_neurons) } else { 1 };
            let parts_by_syn = if max_syn > 0 { total_synapses.div_ceil(max_syn) } else { 1 };
            parts = parts_by_neurons.max(parts_by_syn).max(1);

            // Greedy size-balanced assignment
            let mut buckets: Vec<usize> = vec![0; parts];
            let mut pops: Vec<(String, usize)> = g.populations.iter().map(|p| (p.name.clone(), p.size as usize)).collect();
            pops.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
            for (name, sz) in pops {
                let idx = buckets
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if max_neurons > 0 && sz > max_neurons {
                    violations.push(serde_json::json!({
                        "code": "POP_EXCEEDS_MAX_NEURONS_PER_CORE",
                        "population": name,
                        "size": sz,
                        "max": max_neurons
                    }));
                }
                buckets[idx] += sz;
                assignment.push((name, idx));
            }
        } else {
            // Naive: single part, trivial assignment (use initial default of 1)
            assignment = g.populations.iter().map(|p| (p.name.clone(), 0usize)).collect();
        }

        let assignment_json: Vec<serde_json::Value> = assignment
            .iter()
            .map(|(pop, part)| serde_json::json!({ "population": pop, "part": part }))
            .collect();

        let meta = serde_json::json!({
            "parts": parts as u32,
            "strategy": strategy,
            "assignment": assignment_json,
            "violations": violations,
        });
        g.attributes.insert("partition".to_string(), meta);
        Ok(g)
    }
}

pub struct PlacementPass;
impl Pass for PlacementPass {
    fn name(&self) -> &str { "placement" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        // Derive partition assignment
        let parts = g.attributes.get("partition").and_then(|v| v.get("parts")).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut pop_to_part: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Some(assign) = g.attributes.get("partition").and_then(|v| v.get("assignment")).and_then(|v| v.as_array()) {
            for a in assign {
                if let (Some(pop), Some(part)) = (a.get("population").and_then(|x| x.as_str()), a.get("part").and_then(|x| x.as_u64())) {
                    pop_to_part.insert(pop.to_string(), part as usize);
                }
            }
        } else {
            for p in &g.populations {
                pop_to_part.insert(p.name.clone(), 0usize);
            }
        }

        // Count resources per part
        let mut neurons_per_part = vec![0usize; parts];
        for p in &g.populations {
            let part = *pop_to_part.get(&p.name).unwrap_or(&0usize);
            neurons_per_part[part] += p.size as usize;
        }
        let mut syn_per_part = vec![0usize; parts];
        for c in &g.connections {
            let pre_part = *pop_to_part.get(&c.pre).unwrap_or(&0usize);
            let post_part = *pop_to_part.get(&c.post).unwrap_or(&0usize);
            if pre_part == post_part {
                syn_per_part[pre_part] += 1;
            }
        }

        // Target-aware memory model (fallback to coarse defaults if unspecified)
        let caps = extract_caps_from_graph(&g);
        let neuron_mem_kib: f64 = caps.as_ref().and_then(|c| c.neuron_mem_kib_per).unwrap_or(0.01); // ~10B/neuron
        let syn_mem_kib: f64 = caps.as_ref().and_then(|c| c.syn_mem_kib_per).unwrap_or(0.001);      // ~1B/synapse
        let core_mem_cap: Option<f64> = caps.as_ref().and_then(|c| c.core_memory_kib).map(|v| v as f64);
        let max_fan_in = caps.as_ref().and_then(|c| c.max_fan_in).map(|v| v as usize);
        let max_fan_out = caps.as_ref().and_then(|c| c.max_fan_out).map(|v| v as usize);

        let mut violations: Vec<serde_json::Value> = Vec::new();
        for part in 0..parts {
            let mem: f64 = (neurons_per_part[part] as f64) * neuron_mem_kib
                + (syn_per_part[part] as f64) * syn_mem_kib;
            if let Some(cap) = core_mem_cap {
                if mem > cap {
                    violations.push(serde_json::json!({
                        "code": "CORE_MEMORY_EXCEEDED",
                        "part": part,
                        "estimate_kib": mem,
                        "cap_kib": cap
                    }));
                }
            }
        }

        // Fan-in/out checks per population
        let mut fan_in: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut fan_out: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for c in &g.connections {
            *fan_out.entry(c.pre.clone()).or_insert(0) += 1;
            *fan_in.entry(c.post.clone()).or_insert(0) += 1;
        }
        for p in &g.populations {
            if let Some(cap) = max_fan_in {
                if let Some(v) = fan_in.get(&p.name) {
                    if *v > cap {
                        violations.push(serde_json::json!({
                            "code": "MAX_FAN_IN_EXCEEDED",
                            "population": p.name,
                            "fan_in": v,
                            "cap": cap
                        }));
                    }
                }
            }
            if let Some(cap) = max_fan_out {
                if let Some(v) = fan_out.get(&p.name) {
                    if *v > cap {
                        violations.push(serde_json::json!({
                            "code": "MAX_FAN_OUT_EXCEEDED",
                            "population": p.name,
                            "fan_out": v,
                            "cap": cap
                        }));
                    }
                }
            }
        }

        let status = if violations.is_empty() { "ok" } else { "violations" };
        let meta = serde_json::json!({
            "status": status,
            "parts": parts,
            "neurons_per_part": neurons_per_part,
            "synapses_per_part": syn_per_part,
            "violations": violations
        });
        g.attributes.insert("placement".to_string(), meta);
        Ok(g)
    }
}

pub struct RoutingPass;
impl Pass for RoutingPass {
    fn name(&self) -> &str { "routing" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        // Load partition assignment
        let parts = g.attributes
            .get("partition")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let mut pop_to_part: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Some(assign) = g.attributes
            .get("partition")
            .and_then(|v| v.get("assignment"))
            .and_then(|v| v.as_array())
        {
            for a in assign {
                if let (Some(pop), Some(part)) = (
                    a.get("population").and_then(|x| x.as_str()),
                    a.get("part").and_then(|x| x.as_u64()),
                ) {
                    pop_to_part.insert(pop.to_string(), part as usize);
                }
            }
        } else {
            for p in &g.populations {
                pop_to_part.insert(p.name.clone(), 0usize);
            }
        }

        // Count inter-part edges
        let mut matrix = vec![vec![0usize; parts]; parts];
        let mut cross_edges = 0usize;
        for c in &g.connections {
            let i = *pop_to_part.get(&c.pre).unwrap_or(&0usize);
            let j = *pop_to_part.get(&c.post).unwrap_or(&0usize);
            if i != j {
                matrix[i][j] += 1;
                cross_edges += 1;
            }
        }

        // Bandwidth estimate against HAL cap using per-event size and default rate
        let caps = extract_caps_from_graph(&g);
        let cap_bw = caps.as_ref().and_then(|c| c.interconnect_bandwidth_mbps).map(|v| v as f64);
        let bytes_per_event = caps.as_ref().and_then(|c| c.bytes_per_event).unwrap_or(4) as f64;
        let event_rate_hz = caps.as_ref().and_then(|c| c.default_spike_rate_hz).unwrap_or(100.0);
        // Estimate: each cross-part edge contributes event_rate_hz spikes/s of size bytes_per_event
        let est_bw_mbps = (cross_edges as f64) * event_rate_hz * bytes_per_event * 8.0 / 1_000_000.0;
        let status = match cap_bw {
            Some(cap) if est_bw_mbps > cap => "congested",
            _ => "ok",
        };

        let meta = serde_json::json!({
            "status": status,
            "cross_edges": cross_edges,
            "estimated_bandwidth_mbps": est_bw_mbps,
            "matrix": matrix,
        });
        g.attributes.insert("routing".to_string(), meta);
        Ok(g)
    }
}

pub struct TimingPass;
impl Pass for TimingPass {
    fn name(&self) -> &str { "timing" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        // Use HAL time resolution to translate per-edge delay to discrete ticks
        let caps = extract_caps_from_graph(&g);
        let time_res_ns: u64 = caps.as_ref().and_then(|c| c.time_resolution_ns).unwrap_or(1_000_000); // default 1ms
        let mut ticks: Vec<u64> = Vec::new();
        for c in &g.connections {
            let ns = (c.delay_ms.max(0.0) as f64) * 1_000_000.0;
            let t = (ns / (time_res_ns as f64)).ceil() as u64;
            ticks.push(t);
        }
        let max_ticks = ticks.iter().copied().max().unwrap_or(0);
        let min_ticks = ticks.iter().copied().min().unwrap_or(0);
        let avg_ticks = if ticks.is_empty() { 0.0 } else { (ticks.iter().copied().sum::<u64>() as f64) / (ticks.len() as f64) };
        let meta = serde_json::json!({
            "time_resolution_ns": time_res_ns,
            "max_delay_ticks": max_ticks,
            "min_delay_ticks": min_ticks,
            "avg_delay_ticks": avg_ticks
        });
        g.attributes.insert("timing".to_string(), meta);
        Ok(g)
    }
}

pub struct ResourceCheckPass;
impl Pass for ResourceCheckPass {
    fn name(&self) -> &str { "resource-check" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let caps = extract_caps_from_graph(&g);

        // Partition context
        let parts = g.attributes.get("partition").and_then(|v| v.get("parts")).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut pop_to_part: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Some(assign) = g.attributes.get("partition").and_then(|v| v.get("assignment")).and_then(|v| v.as_array()) {
            for a in assign {
                if let (Some(pop), Some(part)) = (a.get("population").and_then(|x| x.as_str()), a.get("part").and_then(|x| x.as_u64())) {
                    pop_to_part.insert(pop.to_string(), part as usize);
                }
            }
        } else {
            for p in &g.populations { pop_to_part.insert(p.name.clone(), 0usize); }
        }

        // Per-part resources
        let mut neurons_per_part = vec![0usize; parts];
        let mut syn_per_part = vec![0usize; parts];
        for p in &g.populations {
            let part = *pop_to_part.get(&p.name).unwrap_or(&0usize);
            neurons_per_part[part] += p.size as usize;
        }
        for c in &g.connections {
            let i = *pop_to_part.get(&c.pre).unwrap_or(&0usize);
            let j = *pop_to_part.get(&c.post).unwrap_or(&0usize);
            if i == j { syn_per_part[i] += 1; }
        }

        // Fan in/out
        let mut fan_in: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut fan_out: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for c in &g.connections {
            *fan_out.entry(c.pre.clone()).or_insert(0) += 1;
            *fan_in.entry(c.post.clone()).or_insert(0) += 1;
        }

        // Violations against HAL caps
        let mut violations: Vec<serde_json::Value> = Vec::new();
        if let Some(c) = caps {
            if let Some(maxn) = c.max_neurons_per_core {
                for (i, n) in neurons_per_part.iter().enumerate() {
                    if (*n as u32) > maxn {
                        violations.push(serde_json::json!({
                            "code": "MAX_NEURONS_PER_CORE_EXCEEDED",
                            "part": i,
                            "neurons": n,
                            "cap": maxn
                        }));
                    }
                }
            }
            if let Some(maxs) = c.max_synapses_per_core {
                for (i, s) in syn_per_part.iter().enumerate() {
                    if (*s as u32) > maxs {
                        violations.push(serde_json::json!({
                            "code": "MAX_SYNAPSES_PER_CORE_EXCEEDED",
                            "part": i,
                            "synapses": s,
                            "cap": maxs
                        }));
                    }
                }
            }
            if let Some(cap) = c.max_fan_in {
                for (pop, v) in &fan_in {
                    if (*v as u32) > cap {
                        violations.push(serde_json::json!({
                            "code": "MAX_FAN_IN_EXCEEDED",
                            "population": pop,
                            "fan_in": v,
                            "cap": cap
                        }));
                    }
                }
            }
            if let Some(cap) = c.max_fan_out {
                for (pop, v) in &fan_out {
                    if (*v as u32) > cap {
                        violations.push(serde_json::json!({
                            "code": "MAX_FAN_OUT_EXCEEDED",
                            "population": pop,
                            "fan_out": v,
                            "cap": cap
                        }));
                    }
                }
            }
        }

        let legal = violations.is_empty();
        let meta = serde_json::json!({
            "legal": legal,
            "neurons_per_part": neurons_per_part,
            "synapses_per_part": syn_per_part,
            "fan_in": fan_in.iter().map(|(k,v)| serde_json::json!({"population": k, "fan_in": v})).collect::<Vec<_>>(),
            "fan_out": fan_out.iter().map(|(k,v)| serde_json::json!({"population": k, "fan_out": v})).collect::<Vec<_>>(),
            "violations": violations
        });
        g.attributes.insert("resource_check".to_string(), meta);
        Ok(g)
    }
}

/* RISC-V specific pass stubs: LowerToKernels, MemoryLayoutAndQuant, KernelFusionAndScheduling,
   VectorizeKernels, BareMetalTuning, ControlPlaneDriverGen. These are backend-agnostic stubs that
   annotate the graph for downstream RISC-V codegen without requiring hardware routing. */

pub struct RvLowerToKernelsPass;
impl Pass for RvLowerToKernelsPass {
    fn name(&self) -> &str { "rv-lower" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let kernel_count = g.populations.len().max(1);
        let mode = if g.attributes.contains_key("timing") { "tick" } else { "event" };
        let meta = serde_json::json!({
            "status": "ok",
            "mode": mode,
            "kernel_count": kernel_count,
            "notes": "lowered SNN ops into CPU kernels"
        });
        g.attributes.insert("rv_kernels".to_string(), meta);
        Ok(g)
    }
}

pub struct RvMemoryLayoutAndQuantPass;
impl Pass for RvMemoryLayoutAndQuantPass {
    fn name(&self) -> &str { "rv-layout" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        // Heuristics: detect manifest type from attached path (if present)
        let manifest_path = g.attributes
            .get("hal_manifest_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let is_rv64gcv = manifest_path.ends_with("riscv64gcv_linux.toml");
        let is_rv32_bare = manifest_path.ends_with("riscv32imac_bare.toml");
        let vector_available = is_rv64gcv;

        // Pull available weight precisions to set a default quantization choice
        let caps = extract_caps_from_graph(&g);
        let default_bits = caps
            .as_ref()
            .and_then(|c| c.weight_precisions.as_ref())
            .and_then(|v| v.iter().copied().max())
            .unwrap_or(16);

        let vector_bytes = if vector_available { 64 } else { 16 };
        let align_bytes = if vector_available { 64 } else { 16 };
        let meta = serde_json::json!({
            "status": "ok",
            "vector_available": vector_available,
            "vector_bytes": vector_bytes,
            "align_bytes": align_bytes,
            "quant_bits_default": default_bits,
            "profile": if is_rv32_bare { "rv32-bare" } else { "rv64-linux" },
        });
        g.attributes.insert("rv_layout".to_string(), meta);
        Ok(g)
    }
}

pub struct RvKernelFusionAndSchedulingPass;
impl Pass for RvKernelFusionAndSchedulingPass {
    fn name(&self) -> &str { "rv-schedule" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let fused = vec!["integrate", "threshold"];
        let threads: u32 = 1; // M1: single-threaded baseline
        let meta = serde_json::json!({
            "status": "ok",
            "fused_stages": fused,
            "threads": threads,
            "notes": "baseline single-thread schedule"
        });
        g.attributes.insert("rv_schedule".to_string(), meta);
        Ok(g)
    }
}

pub struct RvVectorizeKernelsPass;
impl Pass for RvVectorizeKernelsPass {
    fn name(&self) -> &str { "rv-vectorize" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let layout = g.attributes.get("rv_layout").cloned().unwrap_or(serde_json::json!({}));
        let vector_available = layout.get("vector_available").and_then(|v| v.as_bool()).unwrap_or(false);
        let vlen = layout.get("vector_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
        let meta = serde_json::json!({
            "status": "ok",
            "enabled": vector_available,
            "vlen_bytes": vlen,
            "notes": "RVV intrinsic mapping deferred to backend"
        });
        g.attributes.insert("rv_vectorize".to_string(), meta);
        Ok(g)
    }
}

pub struct RvBareMetalTuningPass;
impl Pass for RvBareMetalTuningPass {
    fn name(&self) -> &str { "rv-baremetal-tuning" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let manifest_path = g.attributes
            .get("hal_manifest_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let size_optimized = manifest_path.ends_with("riscv32imac_bare.toml");
        let meta = serde_json::json!({
            "status": "ok",
            "size_optimized": size_optimized,
            "use_compressed": true,
            "notes": "optimize for code size on RV32 bare metal"
        });
        g.attributes.insert("rv_bare_tuning".to_string(), meta);
        Ok(g)
    }
}

pub struct RvControlPlaneDriverGenPass;
impl Pass for RvControlPlaneDriverGenPass {
    fn name(&self) -> &str { "rv-control-plane-driver" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let manifest_path = g.attributes
            .get("hal_manifest_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let targeted = manifest_path.ends_with("riscv64gc_ctrl.toml");
        let meta = serde_json::json!({
            "status": if targeted { "ok" } else { "skipped" },
            "mmio": { "requires_fence_io": true, "aligned_access": true },
            "dma": { "supported": targeted, "alignment": 64 },
            "notes": "generate control-plane configuration for accelerator"
        });
        g.attributes.insert("rv_ctrl_plane".to_string(), meta);
        Ok(g)
    }
}

pub enum DumpFormat {
    Json,
    Yaml,
    #[cfg(feature = "bin")]
    Bin,
}

pub struct PipelineConfig {
    pub passes: Vec<String>,
    pub dump_dir: Option<PathBuf>,
    pub dump_formats: Vec<DumpFormat>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            passes: vec!["noop".into()],
            dump_dir: None,
            dump_formats: vec![DumpFormat::Json],
        }
    }
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self { Self { passes: Vec::new() } }
    pub fn add_pass<P: Pass + 'static>(&mut self, p: P) { self.passes.push(Box::new(p)); }

    pub fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        for p in &self.passes {
            g = p.run(g)?;
        }
        Ok(g)
    }

    pub fn run_with_config(&self, mut g: nir::Graph, cfg: &PipelineConfig) -> Result<nir::Graph> {
        #[cfg(feature = "telemetry")]
        let app = std::env::var("NC_PROFILE_JSONL")
            .ok()
            .and_then(|p| telemetry::profiling::Appender::open(p).ok());

        for (idx, p) in self.passes.iter().enumerate() {
            #[cfg(feature = "telemetry")]
            let _timer = {
                if let Some(a) = app.as_ref() {
                    let labels = telemetry::labels::pass(&g.name, p.name());
                    Some(a.start_timer("passes.pass_ms", labels))
                } else {
                    None
                }
            };

            g = p.run(g)?;
            if let Some(dir) = &cfg.dump_dir {
                dump_graph(&g, dir, idx, p.name(), &cfg.dump_formats)?;
            }

            #[cfg(feature = "telemetry")]
            if let Some(a) = &app {
                let l = telemetry::labels::pass(&g.name, p.name());
                let _ = a.counter("graph.populations", g.populations.len() as f64, l.clone());
                let _ = a.counter("graph.connections", g.connections.len() as f64, l.clone());
                let _ = a.counter("graph.probes", g.probes.len() as f64, l);
            }
        }
        Ok(g)
    }
}

impl Default for PassManager {
    fn default() -> Self { Self::new() }
}

fn dump_graph(g: &nir::Graph, dir: &Path, idx: usize, pass: &str, fmts: &[DumpFormat]) -> Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    let base = format!("{:02}_{}", idx, pass.replace('/', "_"));
    for f in fmts {
        match f {
            DumpFormat::Json => {
                let s = g.to_json_string().map_err(|e| anyhow::anyhow!(e))?;
                fs::write(dir.join(format!("{base}.json")), s)?;
            }
            DumpFormat::Yaml => {
                let s = g.to_yaml_string().map_err(|e| anyhow::anyhow!(e))?;
                fs::write(dir.join(format!("{base}.yaml")), s)?;
            }
            #[cfg(feature = "bin")]
            DumpFormat::Bin => {
                let b = g.to_bytes().map_err(|e| anyhow::anyhow!(e))?;
                fs::write(dir.join(format!("{base}.bin")), b)?;
            }
        }
    }
    Ok(())
}

/// Build a pipeline by pass names (string identifiers)
pub fn build_pipeline(pm: &mut PassManager, names: &[String]) -> Result<()> {
    for n in names {
        match n.as_str() {
            "noop" | "no-op" => pm.add_pass(NoOpPass),
            "validate" => pm.add_pass(ValidatePass),
            "quantize4" => pm.add_pass(QuantizeWeightsPass { bits: 4 }),
            "quantize8" => pm.add_pass(QuantizeWeightsPass { bits: 8 }),
            "quantize16" => pm.add_pass(QuantizeWeightsPass { bits: 16 }),
            "partition" => pm.add_pass(PartitionPass),
            "placement" => pm.add_pass(PlacementPass),
            "routing" => pm.add_pass(RoutingPass),
            "timing" => pm.add_pass(TimingPass),
            "resource-check" | "resource_check" => pm.add_pass(ResourceCheckPass),
            other => bail!("unknown pass '{other}'"),
        }
    }
    Ok(())
}

/// Register all generic NIR passes (for generic::Registry) including "lower_to_kernels".
pub fn register_generic_nir_passes(reg: &mut generic::Registry<nir::Graph>) {
    lower_to_kernels::register(reg);
    memory_layout_and_quant::register(reg);
    kernel_fusion_and_scheduling::register(reg);
    validation::register(reg);
}

/// Convenience: create a default registry pre-registered with built-in passes.
pub fn default_generic_nir_registry() -> generic::Registry<nir::Graph> {
    let mut r = generic::Registry::<nir::Graph>::new();
    register_generic_nir_passes(&mut r);
    r
}

/// Build a non-invasive pipeline that runs EIR validation first, followed by the
/// current default generic NIR passes. This does not alter any default pipelines
/// used elsewhere; it simply provides an additive constructor for consumers.
pub fn pipeline_with_eir_validation() -> generic::Pipeline<nir::Graph> {
    // Build a local registry and register both the EIR validator and the existing passes.
    let mut reg = generic::Registry::<nir::Graph>::new();
    eir_validate::register(&mut reg);
    lower_to_kernels::register(&mut reg);
    memory_layout_and_quant::register(&mut reg);
    kernel_fusion_and_scheduling::register(&mut reg);
    validation::register(&mut reg);

    // Compose the descriptor: Validate EIR first, then the existing default set.
    let passes = vec![
        generic::PassSpec { name: "eir_validate".into(), config: None },
        generic::PassSpec { name: "lower_to_kernels".into(), config: None },
        generic::PassSpec { name: "memory_layout_and_quant".into(), config: None },
        generic::PassSpec { name: "kernel_fusion_and_scheduling".into(), config: None },
        generic::PassSpec { name: "validation".into(), config: None },
    ];
    let desc = generic::PipelineDescriptor { passes };

    // Build the concrete pipeline (panic on programmer error to keep signature simple).
    reg.build_pipeline(&desc).expect("pipeline_with_eir_validation build")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn run_noop_pipeline() {
        let g = nir::Graph::new("t");
        let mut pm = PassManager::new();
        pm.add_pass(NoOpPass);
        let out = pm.run(g).unwrap();
        assert_eq!(out.name, "t");
    }

    #[test]
    fn run_validate_pipeline() {
        let g = nir::Graph::new("t2");
        let mut pm = PassManager::new();
        pm.add_pass(ValidatePass);
        let out = pm.run(g).unwrap();
        assert_eq!(out.name, "t2");
    }

    #[test]
    fn run_quantize_pipeline() {
        let mut g = nir::Graph::new("tq");
        g.populations.push(nir::Population { name: "a".into(), size: 1, model: "LIF".into(), params: serde_json::json!({}) });
        g.populations.push(nir::Population { name: "b".into(), size: 1, model: "LIF".into(), params: serde_json::json!({}) });
        g.connections.push(nir::Connection { pre: "a".into(), post: "b".into(), weight: 0.1234, delay_ms: 0.0, plasticity: None });
        let mut pm = PassManager::new();
        pm.add_pass(ValidatePass);
        pm.add_pass(QuantizeWeightsPass { bits: 8 });
        let out = pm.run(g).unwrap();
        assert_eq!(out.name, "tq");
        assert!(out.connections[0].weight.is_finite());
        assert!(out.connections[0].weight >= -1.0 && out.connections[0].weight <= 1.0);
    }
}

//------------------------------------------------------------------------------
// Generic, lightweight pass API and pipeline descriptor (compile-only for now)
//------------------------------------------------------------------------------
// These types are intentionally generic over the module type M and do not couple
// to NIR. They live under the `generic` module to avoid any disruption to the
// existing NIR-specific Pass trait and PassManager that other crates use today.
pub mod generic {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::fmt;

    // Outcome and error -----------------------------------------------------------------

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PassOutcome {
        Changed,
        Unchanged,
    }

    #[derive(Debug)]
    pub enum PassError {
        InvalidInput(String),
        InvariantViolation(String),
        Unsupported(String),
        Internal(String),
        Io(std::io::Error),
    }

    impl fmt::Display for PassError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                PassError::InvalidInput(s) => write!(f, "invalid input: {s}"),
                PassError::InvariantViolation(s) => write!(f, "invariant violation: {s}"),
                PassError::Unsupported(s) => write!(f, "unsupported: {s}"),
                PassError::Internal(s) => write!(f, "internal error: {s}"),
                PassError::Io(e) => write!(f, "io error: {e}"),
            }
        }
    }

    impl std::error::Error for PassError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                PassError::Io(e) => Some(e),
                _ => None,
            }
        }
    }

    impl From<std::io::Error> for PassError {
        fn from(e: std::io::Error) -> Self {
            PassError::Io(e)
        }
    }

    pub type PassResult = Result<PassOutcome, PassError>;

    // Context ---------------------------------------------------------------------------

    #[derive(Debug, Default, Clone)]
    pub struct PassContext {
        pub config: Option<Value>,
        pub run_id: Option<String>,
    }

    // Generic pass trait ----------------------------------------------------------------

    pub trait Pass<M>: Send + Sync {
        fn name(&self) -> &'static str;
        fn run(&self, module: &mut M, ctx: &mut PassContext) -> PassResult;
    }

    // Pipeline and execution ------------------------------------------------------------

    pub struct Pipeline<M> {
        // Keep per-step config to thread into PassContext on execution
        steps: Vec<(Box<dyn Pass<M>>, Option<Value>)>,
    }

    impl<M> Pipeline<M> {
        pub fn new() -> Self {
            Self { steps: Vec::new() }
        }

        pub fn with_steps(steps: Vec<(Box<dyn Pass<M>>, Option<Value>)>) -> Self {
            Self { steps }
        }

        /// Executes steps in order; if any step returns Changed, overall is Changed.
        /// On error, propagates PassError immediately.
        pub fn run(&self, module: &mut M, ctx: &mut PassContext) -> PassResult {
            let mut changed_any = false;
            for (p, cfg) in &self.steps {
                ctx.config = cfg.clone();
                let out = p.run(module, ctx)?;
                if matches!(out, PassOutcome::Changed) {
                    changed_any = true;
                }
            }
            Ok(if changed_any {
                PassOutcome::Changed
            } else {
                PassOutcome::Unchanged
            })
        }
    }

    // Descriptor + builder + registry ---------------------------------------------------

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct PassSpec {
        pub name: String,
        #[serde(default)]
        pub config: Option<Value>,
    }

    /// PipelineDescriptor holds an ordered list of passes to run.
    ///
    /// Example (JSON descriptor parsed with serde_json), then registry + pipeline:
    /// ```ignore
    /// use nc_passes::generic::{Registry, PipelineDescriptor, PassSpec, PassContext, PassOutcome, Pass};
    ///
    /// #[derive(Default)]
    /// struct MyModule { pub n: usize }
    ///
    /// struct Noop;
    /// impl Pass<MyModule> for Noop {
    ///     fn name(&self) -> &'static str { "noop" }
    ///     fn run(&self, _m: &mut MyModule, _ctx: &mut PassContext) -> Result<PassOutcome, _> {
    ///         Ok(PassOutcome::Unchanged)
    ///     }
    /// }
    ///
    /// struct Inc;
    /// impl Pass<MyModule> for Inc {
    ///     fn name(&self) -> &'static str { "inc" }
    ///     fn run(&self, m: &mut MyModule, ctx: &mut PassContext) -> Result<PassOutcome, _> {
    ///         let delta = ctx.config.as_ref()
    ///             .and_then(|v| v.get("delta"))
    ///             .and_then(|v| v.as_u64())
    ///             .unwrap_or(1) as usize;
    ///         m.n += delta;
    ///         Ok(PassOutcome::Changed)
    ///     }
    /// }
    ///
    /// fn mk_noop(_: Option<&serde_json::Value>) -> Box<dyn Pass<MyModule>> { Box::new(Noop) }
    /// fn mk_inc(_: Option<&serde_json::Value>) -> Box<dyn Pass<MyModule>> { Box::new(Inc) }
    ///
    /// let json = r#"[{"name":"noop"},{"name":"inc","config":{"delta":2}}]"#;
    /// let passes: Vec<PassSpec> = serde_json::from_str(json).unwrap();
    /// let desc = PipelineDescriptor { passes };
    ///
    /// let mut reg = Registry::<MyModule>::new();
    /// reg.register("noop", mk_noop);
    /// reg.register("inc", mk_inc);
    ///
    /// let p = reg.build_pipeline(&desc).unwrap();
    /// let mut module = MyModule::default();
    /// let mut ctx = PassContext::default();
    /// let out = p.run(&mut module, &mut ctx).unwrap();
    /// assert!(matches!(out, PassOutcome::Changed));
    /// ```
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct PipelineDescriptor {
        pub passes: Vec<PassSpec>,
    }

    /// Registry maps a pass name to a constructor taking an optional config.
    ///
    /// It can then build a typed Pipeline from a PipelineDescriptor.
    pub struct Registry<M> {
        ctors: HashMap<&'static str, fn(Option<&Value>) -> Box<dyn Pass<M>>>,
    }

    impl<M> Registry<M> {
        pub fn new() -> Self {
            Self { ctors: HashMap::new() }
        }

        pub fn register(
            &mut self,
            name: &'static str,
            ctor: fn(Option<&Value>) -> Box<dyn Pass<M>>,
        ) {
            self.ctors.insert(name, ctor);
        }

        pub fn build_pipeline(&self, desc: &PipelineDescriptor) -> Result<Pipeline<M>, PassError> {
            let mut steps = Vec::with_capacity(desc.passes.len());
            for spec in &desc.passes {
                let ctor = self
                    .ctors
                    .get(spec.name.as_str())
                    .ok_or_else(|| PassError::Unsupported(format!("unknown pass '{}'", spec.name)))?;
                let pass = ctor(spec.config.as_ref());
                steps.push((pass, spec.config.clone()));
            }
            Ok(Pipeline { steps })
        }
    }

    // Unit tests -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        #[derive(Debug, Default)]
        struct DummyModule {
            pub n: usize,
        }

        struct NoOpPass;
        impl Pass<DummyModule> for NoOpPass {
            fn name(&self) -> &'static str { "noop" }
            fn run(&self, _module: &mut DummyModule, _ctx: &mut PassContext) -> PassResult {
                Ok(PassOutcome::Unchanged)
            }
        }

        struct IncPass;
        impl Pass<DummyModule> for IncPass {
            fn name(&self) -> &'static str { "inc" }
            fn run(&self, module: &mut DummyModule, ctx: &mut PassContext) -> PassResult {
                let delta = ctx.config.as_ref()
                    .and_then(|v| v.get("delta"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                module.n += delta;
                Ok(PassOutcome::Changed)
            }
        }

        fn mk_noop(_: Option<&Value>) -> Box<dyn Pass<DummyModule>> { Box::new(NoOpPass) }
        fn mk_inc(_: Option<&Value>) -> Box<dyn Pass<DummyModule>> { Box::new(IncPass) }

        #[test]
        fn pipeline_builds_and_runs_smoke() {
            let json = r#"[{"name":"noop"},{"name":"inc","config":{"delta":2}},{"name":"noop"}]"#;
            let passes: Vec<PassSpec> = serde_json::from_str(json).unwrap();
            let desc = PipelineDescriptor { passes };

            let mut reg = Registry::<DummyModule>::new();
            reg.register("noop", mk_noop);
            reg.register("inc", mk_inc);

            let pipeline = reg.build_pipeline(&desc).unwrap();

            let mut m = DummyModule { n: 0 };
            let mut ctx = PassContext::default();
            let outcome = pipeline.run(&mut m, &mut ctx).unwrap();

            assert_eq!(m.n, 2);
            assert!(matches!(outcome, PassOutcome::Changed));
        }

        #[test]
        fn unknown_pass_yields_error() {
            let json = r#"[{"name":"missing"}]"#;
            let passes: Vec<PassSpec> = serde_json::from_str(json).unwrap();
            let desc = PipelineDescriptor { passes };

            let reg = Registry::<DummyModule>::new();
            match reg.build_pipeline(&desc) {
                Err(PassError::Unsupported(_)) => {}
                _ => panic!("expected PassError::Unsupported for unknown pass"),
            }
        }

        #[test]
        fn unchanged_when_all_noops() {
            let json = r#"[{"name":"noop"},{"name":"noop"}]"#;
            let passes: Vec<PassSpec> = serde_json::from_str(json).unwrap();
            let desc = PipelineDescriptor { passes };

            let mut reg = Registry::<DummyModule>::new();
            reg.register("noop", mk_noop);

            let pipeline = reg.build_pipeline(&desc).unwrap();

            let mut m = DummyModule { n: 0 };
            let mut ctx = PassContext::default();
            let outcome = pipeline.run(&mut m, &mut ctx).unwrap();

            assert!(matches!(outcome, PassOutcome::Unchanged));
            assert_eq!(m.n, 0);
        }

        #[test]
        fn config_is_threaded_into_pass_context() {
            // Two inc passes with different deltas ensure per-step config is visible via ctx.config.
            let json = r#"[{"name":"inc","config":{"delta":3}},{"name":"inc","config":{"delta":4}}]"#;
            let passes: Vec<PassSpec> = serde_json::from_str(json).unwrap();
            let desc = PipelineDescriptor { passes };

            let mut reg = Registry::<DummyModule>::new();
            reg.register("inc", mk_inc);

            let pipeline = reg.build_pipeline(&desc).unwrap();

            let mut m = DummyModule { n: 0 };
            let mut ctx = PassContext::default();
            let outcome = pipeline.run(&mut m, &mut ctx).unwrap();

            assert!(matches!(outcome, PassOutcome::Changed));
            assert_eq!(m.n, 7);
        }
    }
}

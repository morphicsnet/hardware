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
use serde_json::json;

pub mod lower_to_kernels;
pub mod memory_layout_and_quant;
pub mod kernel_fusion_and_scheduling;
pub mod validation;
pub mod eir_validate;
pub mod proofs;
pub mod transactional_sample;
pub mod logic_to_energy;
pub mod gadgetization;
pub mod energy_embedding;
pub mod engine_selection;
pub mod llvm_optimize;

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

/// TrueNorth core mapping pass - assigns neurons to cores and axons
pub struct TnCoreMappingPass;
impl Pass for TnCoreMappingPass {
    fn name(&self) -> &str { "tn-core-mapping" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
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
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let bits = g.attributes.get("caps_truenorth")
            .and_then(|v| v.get("weight_bits"))
            .and_then(|x| x.as_u64())
            .unwrap_or(4) as u32;

        let mut programmed_weights = Vec::new();
        for conn in &g.connections {
            // Simple quantization for TrueNorth (4-bit)
            let weight_q = QuantizeWeightsPass::quantize(conn.weight, bits);
            let levels = if bits >= 31 { u32::MAX } else { 1u32 << bits };
            let l_minus_1 = (levels.saturating_sub(1)) as f32;
            let index = ((weight_q + 1.0) * l_minus_1 / 2.0).round() as u8;
            let axon_idx = (index / 16) as u8;  // 16 weights per axon
            let weight_val = (index % 16) as u8;

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
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
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

/// SpiNNaker core allocation pass - assigns neurons to SpiNNaker chips and cores
pub struct SnCoreAllocationPass;
impl Pass for SnCoreAllocationPass {
    fn name(&self) -> &str { "sn-core-allocation" }
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
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
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
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
    fn run(&self, mut g: nir::Graph) -> Result<nir::Graph> {
        let bits = g.attributes.get("caps_spinnaker")
            .and_then(|v| v.get("weight_bits"))
            .and_then(|x| x.as_u64())
            .unwrap_or(16) as u32;

        let mut synapse_configs = Vec::new();
        for conn in &g.connections {
            // SpiNNaker uses 16-bit weights by default
            let weight_q = if bits <= 8 {
                QuantizeWeightsPass::quantize(conn.weight, bits)
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
            // TrueNorth passes
            "tn-core-mapping" => pm.add_pass(TnCoreMappingPass),
            "tn-weight-programming" => pm.add_pass(TnWeightProgrammingPass),
            "tn-crossbar-config" => pm.add_pass(TnCrossbarConfigPass),
            // SpiNNaker passes
            "sn-core-allocation" => pm.add_pass(SnCoreAllocationPass),
            "sn-aer-routing" => pm.add_pass(SnAerRoutingPass),
            "sn-synapse-programming" => pm.add_pass(SnSynapseProgrammingPass),
            "engine-selection" => pm.add_pass(crate::engine_selection::EngineSelectionPass),
            "transactional-sample" => pm.add_pass(crate::transactional_sample::TransactionalSamplePass),
            "llvm-optimize" => pm.add_pass(crate::llvm_optimize::LLVMPass::new()),
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

/// -----------------------------------------------------------------------------
/// Dump toggles + metrics helpers (H3-7)
///
/// Configuration keys (string -> string) interpreted by should_dump and metrics:
/// - dump_dir: Required for any dumps to be written. When absent, no dumps.
/// - dump_all: Truthy => enable dumps for all passes regardless of other toggles.
/// - dump_lower, dump_layout, dump_schedule, dump_validation:
///     If any of these specific toggles are present in the config, dumping becomes
///     selective: only passes explicitly set to a truthy value will dump. If the
///     toggle for a queried pass is absent or false, that pass will not dump.
/// - metrics: Truthy => enable lightweight tracing::info! logs with structured
///     fields. When disabled, there is only a single fast boolean check overhead.
///
/// Truthy strings (case-insensitive): "1", "true", "yes".
/// Everything else is treated as false.
///
/// Back-compat behavior:
/// - When dump_dir is set and none of the specific dump_* toggles are present,
///   all dumps are produced (equivalent to legacy behavior).
pub fn parse_bool(s: &str) -> bool {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => true,
        _ => false,
    }
}

pub fn metrics_enabled(config: &std::collections::HashMap<String, String>) -> bool {
    config.get("metrics").map(|v| parse_bool(v)).unwrap_or(false)
}

pub fn should_dump(pass_key: &str, config: &std::collections::HashMap<String, String>) -> bool {
    // 1) dump_dir must be present
    let has_dump_dir = config.get("dump_dir").map(|s| !s.is_empty()).unwrap_or(false);
    if !has_dump_dir {
        return false;
    }
    // 2) dump_all override
    if config.get("dump_all").map(|v| parse_bool(v)).unwrap_or(false) {
        return true;
    }
    // 3) selective toggles if any specific keys are present
    let specific_keys = ["dump_lower", "dump_layout", "dump_schedule", "dump_validation"];
    let any_specific_present = specific_keys.iter().any(|k| config.contains_key(*k));
    if any_specific_present {
        let key = match pass_key {
            "lower" => "dump_lower",
            "layout" => "dump_layout",
            "schedule" => "dump_schedule",
            "validation" => "dump_validation",
            _ => return false,
        };
        return config.get(key).map(|v| parse_bool(v)).unwrap_or(false);
    }
    // 4) Back-compat: dump when dump_dir is set and no specific toggles provided
    true
}

/// Convert an optional serde_json::Value config into a flat String map suitable
/// for toggle evaluation. Non-string scalars are stringified; complex values are
/// ignored. A bare string config is treated as {"dump_dir": "<that string>"}.
pub fn config_map_from_value(
    cfg: Option<&serde_json::Value>,
) -> std::collections::HashMap<String, String> {
    use serde_json::Value;
    let mut out = std::collections::HashMap::new();
    match cfg {
        Some(Value::String(s)) => {
            out.insert("dump_dir".to_string(), s.clone());
        }
        Some(Value::Object(map)) => {
            for (k, v) in map.iter() {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => {
                        if *b { "true".to_string() } else { "false".to_string() }
                    }
                    Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                out.insert(k.clone(), s);
            }
        }
        _ => {}
    }
    out
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

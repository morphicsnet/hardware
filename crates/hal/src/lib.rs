use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::bail;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HalError {
    #[error("invalid manifest field: {field} ({msg})")]
    InvalidField { field: &'static str, msg: &'static str },
}

#[derive(Debug, Clone)]
pub struct TargetDescriptor {
    pub name: &'static str,
    pub vendor: &'static str,
}

/// Names of built-in targets with manifest stubs under the `targets/` directory.
pub fn builtin_targets() -> &'static [&'static str] {
    &[
        "loihi2",
        "truenorth",
        "akida",
        "spinnaker2",
        "neurogrid",
        "dynaps",
        "memxbar",
        "custom_asic",
        "riscv64gcv_linux",
        "riscv32imac_bare",
        "riscv64gc_ctrl",
    ]
}

#[derive(Debug, Clone, Deserialize)]
pub struct Capabilities {
    pub on_chip_learning: Option<bool>,
    pub weight_precisions: Option<Vec<u32>>,
    pub max_neurons_per_core: Option<u32>,
    pub max_synapses_per_core: Option<u32>,
    pub time_resolution_ns: Option<u64>,

    // Expanded descriptor fields (optional; extend manifests incrementally)
    pub supports_sparse: Option<bool>,
    pub neuron_models: Option<Vec<String>>,
    pub max_fan_in: Option<u32>,
    pub max_fan_out: Option<u32>,
    pub core_memory_kib: Option<u32>,
    pub interconnect_bandwidth_mbps: Option<u32>,
    pub analog: Option<bool>,
    pub on_chip_plasticity_rules: Option<Vec<String>>,

    // Resource and traffic modeling (optional; used by mapping passes)
    /// Approximate KiB required per neuron on this target
    pub neuron_mem_kib_per: Option<f64>,
    /// Approximate KiB required per synapse on this target
    pub syn_mem_kib_per: Option<f64>,
    /// Bytes per spike/event transferred across interconnect
    pub bytes_per_event: Option<u32>,
    /// Default spike rate per connection (Hz) used for coarse bandwidth estimates
    pub default_spike_rate_hz: Option<f64>,

    // CPU/RISC-V (optional, backward-compatible)
    /// e.g., "rv64gcv", "rv32imac", "rv64gc"
    pub isa: Option<String>,
    /// e.g., "lp64d", "ilp32"
    pub abi: Option<String>,

    // ISA/ABI/Extension flags
    pub has_a: Option<bool>,
    pub has_c: Option<bool>,
    pub has_f: Option<bool>,
    pub has_d: Option<bool>,
    pub has_b: Option<bool>,
    pub has_p: Option<bool>,

    pub has_vector: Option<bool>,
    pub vlen_bits_max: Option<u32>,
    /// minimum legal vector length (bits)
    pub zvl_bits_min: Option<u32>,
    pub vlen_is_dynamic: Option<bool>,
    pub has_zicntr: Option<bool>,
    pub has_zihpm: Option<bool>,
    /// free-form extension strings like ["zba","zbb","zbs","v"]
    pub extensions: Option<Vec<String>>,

    // Memory model / layout
    /// "little" | "big"
    pub endianness: Option<String>,
    pub cacheline_bytes: Option<u32>,
    pub icache_kib: Option<u32>,
    pub dcache_kib: Option<u32>,
    pub l2_kib: Option<u32>,
    /// for OS profiles
    pub page_size_bytes: Option<u32>,
    /// "medlow" | "medany" | "small"
    pub code_model: Option<String>,

    // MMIO/DMA (already added; keep and validate)
    pub mmio_supported: Option<bool>,
    pub mmio_base_addr: Option<u64>,
    /// Only 32 or 64 when present
    pub mmio_width_bits: Option<u32>,
    pub dma_supported: Option<bool>,
    /// Must be power-of-two when present and dma_supported = true
    pub dma_alignment: Option<u32>,

    // Profile indicator
    /// "linux_user", "bare_metal", "control_plane" (free form)
    pub profile: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetManifest {
    pub name: String,
    pub vendor: String,
    pub family: String,
    pub version: String,
    pub notes: Option<String>,
    pub capabilities: Option<Capabilities>,
}

pub fn parse_target_manifest_str(s: &str) -> Result<TargetManifest, anyhow::Error> {
    let m: TargetManifest = toml::from_str(s)?;
    Ok(m)
}

pub fn parse_target_manifest_path<P: AsRef<Path>>(path: P) -> Result<TargetManifest, anyhow::Error> {
    let data = fs::read_to_string(path)?;
    parse_target_manifest_str(&data)
}

pub fn load_manifests_from_dir<P: AsRef<Path>>(dir: P) -> Result<Vec<(PathBuf, TargetManifest)>, anyhow::Error> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("toml") {
            let m = parse_target_manifest_path(&p)?;
            out.push((p, m));
        }
    }
    Ok(out)
}

/// Validate a target manifest for basic consistency.
pub fn validate_manifest(m: &TargetManifest) -> anyhow::Result<()> {
    if m.name.trim().is_empty() {
        bail!("manifest.name must be non-empty");
    }
    if m.vendor.trim().is_empty() {
        bail!("manifest.vendor must be non-empty");
    }
    if let Some(c) = &m.capabilities {
        // Basic numeric checks
        if let Some(wp) = &c.weight_precisions {
            if wp.contains(&0) {
                bail!("capabilities.weight_precisions entries must be > 0");
            }
        }
        if let Some(v) = c.max_neurons_per_core {
            if v == 0 {
                bail!("capabilities.max_neurons_per_core must be > 0");
            }
        }
        if let Some(v) = c.max_synapses_per_core {
            if v == 0 {
                bail!("capabilities.max_synapses_per_core must be > 0");
            }
        }
        if let Some(v) = c.time_resolution_ns {
            if v == 0 {
                bail!("capabilities.time_resolution_ns must be > 0");
            }
        }
        if let Some(v) = c.max_fan_in {
            if v == 0 {
                bail!("capabilities.max_fan_in must be > 0");
            }
        }
        if let Some(v) = c.max_fan_out {
            if v == 0 {
                bail!("capabilities.max_fan_out must be > 0");
            }
        }
        if let Some(v) = c.core_memory_kib {
            if v == 0 {
                bail!("capabilities.core_memory_kib must be > 0");
            }
        }
        if let Some(v) = c.interconnect_bandwidth_mbps {
            if v == 0 {
                bail!("capabilities.interconnect_bandwidth_mbps must be > 0");
            }
        }
        if let Some(v) = c.neuron_mem_kib_per {
            if v <= 0.0 {
                bail!("capabilities.neuron_mem_kib_per must be > 0");
            }
        }
        if let Some(v) = c.syn_mem_kib_per {
            if v <= 0.0 {
                bail!("capabilities.syn_mem_kib_per must be > 0");
            }
        }
        if let Some(v) = c.bytes_per_event {
            if v == 0 {
                bail!("capabilities.bytes_per_event must be > 0");
            }
        }
        if let Some(v) = c.default_spike_rate_hz {
            if v <= 0.0 {
                bail!("capabilities.default_spike_rate_hz must be > 0");
            }
        }

        // Vector constraints
        if matches!(c.has_vector, Some(true)) {
            match c.vlen_bits_max {
                Some(v) if v > 0 => {}
                _ => bail!("capabilities.vlen_bits_max must be > 0 when has_vector = true"),
            }
        }
        if let (Some(zvl), Some(vmax)) = (c.zvl_bits_min, c.vlen_bits_max) {
            if zvl == 0 {
                bail!("capabilities.zvl_bits_min must be > 0 when provided");
            }
            if zvl > vmax {
                bail!("capabilities.zvl_bits_min must be <= vlen_bits_max when both present");
            }
            if zvl % 8 != 0 || vmax % 8 != 0 {
                bail!("capabilities.zvl_bits_min and vlen_bits_max must be multiples of 8 when both present");
            }
        }

        // MMIO/DMA constraints
        if matches!(c.mmio_supported, Some(true)) {
            match c.mmio_base_addr {
                Some(v) if v > 0 => {}
                _ => bail!("capabilities.mmio_base_addr must be > 0 when mmio_supported = true"),
            }
            match c.mmio_width_bits {
                Some(32) | Some(64) => {}
                _ => bail!("capabilities.mmio_width_bits must be 32 or 64 when mmio_supported = true"),
            }
        }
        if matches!(c.dma_supported, Some(true)) {
            match c.dma_alignment {
                Some(v) if v > 0 && v.is_power_of_two() => {}
                _ => bail!("capabilities.dma_alignment must be power-of-two > 0 when dma_supported = true"),
            }
        }

        // Memory/layout constraints (best-effort)
        if let Some(e) = c.endianness.as_deref() {
            let e = e.to_lowercase();
            if e != "little" && e != "big" {
                bail!("capabilities.endianness must be 'little' or 'big'");
            }
        }
        if let Some(v) = c.cacheline_bytes {
            if v == 0 || !v.is_power_of_two() {
                bail!("capabilities.cacheline_bytes must be a power-of-two > 0");
            }
        }
        if let Some(v) = c.page_size_bytes {
            if v == 0 || !v.is_power_of_two() {
                bail!("capabilities.page_size_bytes must be a power-of-two > 0");
            }
        }
        if let Some(cm) = c.code_model.as_deref() {
            match cm {
                "medlow" | "medany" | "small" => {}
                _ => bail!("capabilities.code_model must be one of: medlow|medany|small"),
            }
        }

        // For RISC-V manifests (family contains "risc" or isa present)
        let is_riscv_family = m.family.to_lowercase().contains("risc");
        if is_riscv_family || c.isa.is_some() {
            if let Some(isa) = &c.isa {
                let isa_lc = isa.to_lowercase();
                if isa_lc.starts_with("rv32") {
                    if let Some(abi) = &c.abi {
                        if !abi.to_lowercase().starts_with("ilp32") {
                            bail!("capabilities.abi should start_with 'ilp32' for 32-bit RISC-V isa");
                        }
                    }
                }
                if isa_lc.starts_with("rv64") {
                    if let Some(abi) = &c.abi {
                        if !abi.to_lowercase().starts_with("lp64") {
                            bail!("capabilities.abi should start_with 'lp64' for 64-bit RISC-V isa");
                        }
                    }
                }
                if matches!(c.has_vector, Some(true)) && !isa_lc.contains('v') {
                    bail!("capabilities.has_vector = true but isa does not contain 'v'");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_str() {
        let s = r#"
            name = "loihi2"
            vendor = "Intel"
            family = "Loihi"
            version = "2"
            notes = "Sample"
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        assert_eq!(m.name, "loihi2");
        assert_eq!(m.vendor, "Intel");
    }
}


#[cfg(test)]
mod tests_builtin {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn builtin_targets_nonempty() {
        assert!(!builtin_targets().is_empty());
    }

    #[test]
    fn validate_all_targets_dir() {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ws_root = crate_dir.parent().and_then(|p| p.parent()).expect("workspace root");
        let dir = ws_root.join("targets");
        let manis = super::load_manifests_from_dir(&dir).expect("load manifests");
        assert!(!manis.is_empty(), "no target manifests found");
        for (_p, m) in manis {
            super::validate_manifest(&m).expect("manifest valid");
        }
    }
}

#[cfg(test)]
mod tests_validate {
    use super::*;
    #[test]
    fn validate_manifest_ok() {
        let s = r#"
            name = "x"
            vendor = "v"
            family = "F"
            version = "1"
            [capabilities]
            on_chip_learning = true
            weight_precisions = [8, 16]
            max_neurons_per_core = 1024
            max_synapses_per_core = 65536
            time_resolution_ns = 1000
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        validate_manifest(&m).unwrap();
    }

    #[test]
    fn validate_manifest_bad_weight_precision() {
        let s = r#"
            name = "x"
            vendor = "v"
            family = "F"
            version = "1"
            [capabilities]
            weight_precisions = [0]
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        assert!(validate_manifest(&m).is_err());
    }

    #[test]
    fn validate_manifest_riscv_extended_ok() {
        let s = r#"
            name = "riscv64gcv_linux"
            vendor = "Generic"
            family = "RISC-V"
            version = "1"
            [capabilities]
            isa = "rv64gcv"
            abi = "lp64d"
            has_vector = true
            vlen_bits_max = 512
            zvl_bits_min = 128
            vlen_is_dynamic = true
            has_zicntr = true
            has_zihpm = true
            endianness = "little"
            cacheline_bytes = 64
            page_size_bytes = 4096
            code_model = "medany"
            mmio_supported = false
            dma_supported = false
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        validate_manifest(&m).unwrap();
    }

    #[test]
    fn validate_manifest_riscv_invalid_mmio_width() {
        let s = r#"
            name = "rv"
            vendor = "g"
            family = "RISC-V"
            version = "1"
            [capabilities]
            mmio_supported = true
            mmio_base_addr = 1024
            mmio_width_bits = 16
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        assert!(validate_manifest(&m).is_err(), "expected invalid mmio_width_bits");
    }

    #[test]
    fn validate_manifest_riscv_invalid_endianness() {
        let s = r#"
            name = "rv"
            vendor = "g"
            family = "RISC-V"
            version = "1"
            [capabilities]
            endianness = "weird"
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        assert!(validate_manifest(&m).is_err(), "expected invalid endianness");
    }
    /// Backend adapter registry (opt-in).
    ///
    /// This module is only compiled when the feature "backend-cpu-ref-sim" is enabled.
    /// It exposes a simple discovery and dispatch API that allows callers to run a
    /// backend by name without linking UEC or importing its code.
    ///
    /// Notes:
    /// - The adapter crate shells out to an external `uec` CLI; no licensing risk.
    /// - Unknown names result in a friendly error.
    /// - Options and RunReport are serialized as JSON to avoid leaking adapter types.
    #[cfg(feature = "backend-cpu-ref-sim")]
    pub mod backend_registry {
        use super::*;
        use std::path::Path;
    
        /// Return the list of available backend adapter names.
        pub fn available_backends() -> &'static [&'static str] {
            &["cpu-ref-sim"]
        }
    
        /// Run a backend by name.
        ///
        /// - name: "cpu-ref-sim"
        /// - plan: NIR graph (should include EIR graph attributes under "eir" or "eir_graph_attrs")
        /// - out_trace: destination path for JSONL trace
        /// - opts_json: optional JSON object matching nc-backend-cpu-ref-sim::BackendRunOpts
        ///
        /// On success returns a JSON value of nc-backend-cpu-ref-sim::RunReport.
        pub fn run_backend_by_name(
            name: &str,
            plan: &nc_nir::Graph,
            out_trace: &Path,
            opts_json: Option<&serde_json::Value>,
        ) -> anyhow::Result<serde_json::Value> {
            match name {
                "cpu-ref-sim" => {
                    let opts: nc_backend_cpu_ref_sim::BackendRunOpts = match opts_json {
                        Some(v) => serde_json::from_value(v.clone())?,
                        None => Default::default(),
                    };
                    let be = nc_backend_cpu_ref_sim::CpuRefSimBackend::default();
                    let report = be.run(plan, out_trace, &opts)?;
                    Ok(serde_json::to_value(report)?)
                }
                other => {
                    bail!("unknown backend adapter '{other}'. Available: {:?}", available_backends());
                }
            }
        }
    }

    #[test]
    fn validate_manifest_riscv_zvl_vs_vlen() {
        let s = r#"
            name = "rv"
            vendor = "g"
            family = "RISC-V"
            version = "1"
            [capabilities]
            has_vector = true
            vlen_bits_max = 128
            zvl_bits_min = 256
        "#;
        let m = parse_target_manifest_str(s).unwrap();
        assert!(validate_manifest(&m).is_err(), "expected zvl_bits_min > vlen_bits_max to fail");
    }
}

#[cfg(feature = "backend-cpu-ref-sim")]
pub mod backend_registry {
    use anyhow::bail;
    use std::path::Path;

    /// Return the list of available backend adapter names.
    pub fn available_backends() -> &'static [&'static str] {
        &["cpu-ref-sim"]
    }

    /// Run a backend by name.
    ///
    /// - name: "cpu-ref-sim"
    /// - plan: NIR graph (should include EIR attributes)
    /// - out_trace: destination path for JSONL trace
    /// - opts_json: optional JSON object matching nc-backend-cpu-ref-sim::BackendRunOpts
    ///
    /// On success returns a JSON value of nc-backend-cpu-ref-sim::RunReport.
    pub fn run_backend_by_name(
        name: &str,
        plan: &nc_nir::Graph,
        out_trace: &Path,
        opts_json: Option<&serde_json::Value>,
    ) -> anyhow::Result<serde_json::Value> {
        match name {
            "cpu-ref-sim" => {
                let opts: nc_backend_cpu_ref_sim::BackendRunOpts = match opts_json {
                    Some(v) => serde_json::from_value(v.clone())?,
                    None => Default::default(),
                };
                let be = nc_backend_cpu_ref_sim::CpuRefSimBackend::default();
                let report = be.run(plan, out_trace, &opts)?;
                Ok(serde_json::to_value(report)?)
            }
            other => {
                bail!(
                    "unknown backend adapter '{}'. Available: {:?}",
                    other,
                    available_backends()
                );
            }
        }
    }
}

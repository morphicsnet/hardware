use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::event_driven::*;

/// IR-C: Physical Configuration Image
///
/// Compiled from IR-B event-driven graph, this represents the hardware-deployable
/// configuration image with placements, routing tables, and parameter blobs.

/// Physical placement of a processing locus on hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalPlacement {
    pub locus_id: Uuid,
    pub hardware_id: String,      // "chip0.core5" or similar
    pub coordinates: (u32, u32),  // (x, y) on chip
    pub memory_offset: u32,       // Memory allocation offset
    pub power_domain: Option<String>, // Power management group
}

/// NoC routing table entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTableEntry {
    pub source_hardware_id: String,
    pub target_hardware_id: String,
    pub route_path: Vec<(u32, u32)>, // Sequence of (x,y) hops
    pub priority: u8,              // Routing priority (0-255)
    pub qos_class: String,         // Quality of service class
}

/// Parameter blob for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBlob {
    pub table_id: Uuid,
    pub hardware_address: u64,     // Physical memory address
    pub size_bytes: usize,
    pub checksum: u32,             // For integrity verification
    pub data: Vec<u8>,             // Serialized parameter data
}

/// Reconfiguration plan for hot-swapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconfigPlan {
    pub steps: Vec<ReconfigStep>,
    pub estimated_duration_ms: f32,
    pub rollback_plan: Vec<ReconfigStep>, // For failure recovery
}

/// Step in reconfiguration sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconfigStep {
    PauseSubgraph { locus_ids: Vec<Uuid> },
    UpdateParameters { blobs: Vec<ParameterBlob> },
    UpdateRouting { entries: Vec<RoutingTableEntry> },
    UpdatePlacements { placements: Vec<PhysicalPlacement> },
    ResumeSubgraph { locus_ids: Vec<Uuid> },
    ValidateConnectivity,
}

/// The hardware configuration image - IR-C
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigImage {
    pub id: Uuid,
    pub target_hardware: String,           // "loihi2", "spinnaker2", etc.
    pub version: String,
    pub placements: Vec<PhysicalPlacement>,
    pub routing_tables: Vec<RoutingTableEntry>,
    pub parameter_blobs: Vec<ParameterBlob>,
    pub reconfiguration_plan: ReconfigPlan,
    pub metadata: HashMap<String, String>, // Target-specific metadata
}

impl ConfigImage {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            target_hardware: target.into(),
            version: "1.0.0".to_string(),
            placements: Vec::new(),
            routing_tables: Vec::new(),
            parameter_blobs: Vec::new(),
            reconfiguration_plan: ReconfigPlan {
                steps: Vec::new(),
                estimated_duration_ms: 0.0,
                rollback_plan: Vec::new(),
            },
            metadata: HashMap::new(),
        }
    }

    /// Serialize to binary format for deployment
    pub fn to_binary(&self) -> anyhow::Result<Vec<u8>> {
        // Custom binary serialization for hardware loading
        // This would be target-specific (different formats for Loihi vs SpiNNaker)
        let json = serde_json::to_string(self)?;
        Ok(json.into_bytes())
    }

    /// Load from binary format
    pub fn from_binary(data: &[u8]) -> anyhow::Result<Self> {
        let json = std::str::from_utf8(data)?;
        Ok(serde_json::from_str(json)?)
    }

    /// Validate configuration integrity
    pub fn validate(&self) -> anyhow::Result<()> {
        // Check that all references are valid
        // Verify checksums
        // Validate routing consistency
        Ok(())
    }
}

/// Compiler from IR-B event-driven to IR-C physical config
pub fn compile_event_to_physical(event_graph: &EventGraph, target: &str) -> anyhow::Result<ConfigImage> {
    let mut config = ConfigImage::new(target);

    // 1. Place processing loci on hardware
    config.placements = place_loci_on_hardware(&event_graph.loci, target)?;

    // 2. Generate routing tables
    config.routing_tables = generate_routing_tables(&event_graph.routes, &config.placements)?;

    // 3. Package parameter blobs
    config.parameter_blobs = package_parameters(&event_graph.parameters)?;

    // 4. Create reconfiguration plan
    config.reconfiguration_plan = create_reconfig_plan(event_graph, &config)?;

    // 5. Add target-specific metadata
    add_target_metadata(&mut config, target)?;

    Ok(config)
}

fn place_loci_on_hardware(loci: &[ProcessingLocus], target: &str) -> anyhow::Result<Vec<PhysicalPlacement>> {
    // Hardware-specific placement algorithms
    match target {
        "loihi2" => place_on_loihi(loci),
        "spinnaker2" => place_on_spinnaker(loci),
        "truenorth" => place_on_truenorth(loci),
        _ => place_generic(loci),
    }
}

fn place_on_loihi(loci: &[ProcessingLocus]) -> anyhow::Result<Vec<PhysicalPlacement>> {
    // Loihi-specific placement: cores, compartments, synapses
    let mut placements = Vec::new();

    for (i, locus) in loci.iter().enumerate() {
        placements.push(PhysicalPlacement {
            locus_id: locus.id,
            hardware_id: format!("loihi.chip0.core{}", i % 128), // 128 cores per chip
            coordinates: ((i % 16) as u32, (i / 16) as u32),     // 16x8 core grid
            memory_offset: (i * 4096) as u32,                   // 4KB per locus
            power_domain: Some("default".to_string()),
        });
    }

    Ok(placements)
}

fn place_on_spinnaker(loci: &[ProcessingLocus]) -> anyhow::Result<Vec<PhysicalPlacement>> {
    // SpiNNaker-specific placement: ARM cores
    let mut placements = Vec::new();

    for (i, locus) in loci.iter().enumerate() {
        placements.push(PhysicalPlacement {
            locus_id: locus.id,
            hardware_id: format!("spinnaker.chip{}.core{}", i / 18, i % 18), // 18 cores per chip
            coordinates: ((i % 6) as u32, (i / 6) as u32),     // 6x3 core grid
            memory_offset: (i * 8192) as u32,                   // 8KB per locus
            power_domain: Some("default".to_string()),
        });
    }

    Ok(placements)
}

fn place_on_truenorth(loci: &[ProcessingLocus]) -> anyhow::Result<Vec<PhysicalPlacement>> {
    // TrueNorth-specific placement: neurosynaptic cores
    let mut placements = Vec::new();

    for (i, locus) in loci.iter().enumerate() {
        placements.push(PhysicalPlacement {
            locus_id: locus.id,
            hardware_id: format!("truenorth.core{}", i % 4096), // 4096 cores total
            coordinates: ((i % 64) as u32, (i / 64) as u32),    // 64x64 core grid
            memory_offset: (i * 1024) as u32,                   // 1KB per locus
            power_domain: None,
        });
    }

    Ok(placements)
}

fn place_generic(loci: &[ProcessingLocus]) -> anyhow::Result<Vec<PhysicalPlacement>> {
    // Generic placement for simulation or unknown targets
    let mut placements = Vec::new();

    for (i, locus) in loci.iter().enumerate() {
        placements.push(PhysicalPlacement {
            locus_id: locus.id,
            hardware_id: format!("generic.locus{}", i),
            coordinates: (i as u32, 0),
            memory_offset: (i * 4096) as u32,
            power_domain: None,
        });
    }

    Ok(placements)
}

fn generate_routing_tables(routes: &[RoutingRule], placements: &[PhysicalPlacement])
    -> anyhow::Result<Vec<RoutingTableEntry>>
{
    // Build hardware ID lookup
    let mut locus_to_hardware = HashMap::new();
    for placement in placements {
        locus_to_hardware.insert(placement.locus_id, &placement.hardware_id);
    }

    let mut routing_entries = Vec::new();

    for route in routes {
        let source_hw = locus_to_hardware.get(&route.source_locus)
            .ok_or_else(|| anyhow::anyhow!("Source locus not placed"))?;
        let target_hw = locus_to_hardware.get(&route.target_locus)
            .ok_or_else(|| anyhow::anyhow!("Target locus not placed"))?;

        // Generate route path (simplified - real implementation would use routing algorithms)
        let route_path = generate_route_path(source_hw, target_hw)?;

        routing_entries.push(RoutingTableEntry {
            source_hardware_id: (*source_hw).clone(),
            target_hardware_id: (*target_hw).clone(),
            route_path,
            priority: 128, // Default priority
            qos_class: "default".to_string(),
        });
    }

    Ok(routing_entries)
}

fn generate_route_path(_source: &str, _target: &str) -> anyhow::Result<Vec<(u32, u32)>> {
    // Simplified routing - real implementation would use A* or similar
    Ok(vec![(0, 0), (1, 0), (1, 1)]) // Example path
}

fn package_parameters(tables: &[ParameterTable]) -> anyhow::Result<Vec<ParameterBlob>> {
    let mut blobs = Vec::new();

    for table in tables {
        // Serialize parameter data
        let data = serde_json::to_vec(&table.data)?;

        // Calculate checksum
        let checksum = crc32fast::hash(&data);

        blobs.push(ParameterBlob {
            table_id: table.id,
            hardware_address: 0, // Assigned during placement
            size_bytes: data.len(),
            checksum,
            data,
        });
    }

    Ok(blobs)
}

fn create_reconfig_plan(event_graph: &EventGraph, config: &ConfigImage) -> anyhow::Result<ReconfigPlan> {
    let mut steps = Vec::new();

    // Pause affected loci
    let locus_ids: Vec<_> = event_graph.loci.iter().map(|l| l.id).collect();
    steps.push(ReconfigStep::PauseSubgraph { locus_ids: locus_ids.clone() });

    // Update parameters
    steps.push(ReconfigStep::UpdateParameters {
        blobs: config.parameter_blobs.clone()
    });

    // Update routing
    steps.push(ReconfigStep::UpdateRouting {
        entries: config.routing_tables.clone()
    });

    // Update placements
    steps.push(ReconfigStep::UpdatePlacements {
        placements: config.placements.clone()
    });

    // Validate connectivity
    steps.push(ReconfigStep::ValidateConnectivity);

    // Resume loci
    steps.push(ReconfigStep::ResumeSubgraph { locus_ids });

    Ok(ReconfigPlan {
        steps,
        estimated_duration_ms: 10.0, // Rough estimate
        rollback_plan: steps.into_iter().rev().collect(), // Simple reverse for rollback
    })
}

fn add_target_metadata(config: &mut ConfigImage, target: &str) -> anyhow::Result<()> {
    config.metadata.insert("target".to_string(), target.to_string());
    config.metadata.insert("compiled_at".to_string(), chrono::Utc::now().to_rfc3339());
    config.metadata.insert("compiler_version".to_string(), env!("CARGO_PKG_VERSION").to_string());

    // Target-specific metadata
    match target {
        "loihi2" => {
            config.metadata.insert("max_cores".to_string(), "128".to_string());
            config.metadata.insert("synapse_slots".to_string(), "131072".to_string());
        }
        "spinnaker2" => {
            config.metadata.insert("max_cores".to_string(), "18".to_string());
            config.metadata.insert("sdram_mb".to_string(), "128".to_string());
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_image_creation() {
        let config = ConfigImage::new("loihi2");
        assert_eq!(config.target_hardware, "loihi2");
        assert!(config.placements.is_empty());
    }

    #[test]
    fn compile_event_to_physical() {
        let event_graph = EventGraph::new();
        let config = compile_event_to_physical(&event_graph, "loihi2").unwrap();
        assert_eq!(config.target_hardware, "loihi2");
    }
}
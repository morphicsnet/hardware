//! BEIR/GESC/GESA event semantics for NIR.
//!
//! Integrates event-driven, causal propagation into the neuromorphic IR.

use serde::{Deserialize, Serialize};

/// 64-bit GESC Event Packet as per MESC specification.
/// Packed into a u64 for efficiency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EventPacket(pub u64);

impl EventPacket {
    pub fn new(src: u32, dst: u32, class: EventClass, tstamp: u16, qos: QosClass, payload: u8) -> Self {
        let src_masked = (src & 0x3FFFF) as u64; // 18 bits
        let dst_masked = ((dst & 0x3FFFF) as u64) << 18; // next 18
        let class_val = (class as u64) << 36; // 4 bits
        let tstamp_val = (tstamp as u64) << 40; // 16 bits
        let qos_val = (qos as u64) << 56; // 4 bits
        let payload_val = ((payload & 0xF) as u64) << 60; // 4 bits at the end

        let packed = src_masked | dst_masked | class_val | tstamp_val | qos_val | payload_val;
        EventPacket(packed)
    }

    pub fn src_locus(&self) -> u32 {
        (self.0 & 0x3FFFF) as u32
    }

    pub fn dst_scope(&self) -> u32 {
        ((self.0 >> 18) & 0x3FFFF) as u32
    }

    pub fn class(&self) -> EventClass {
        match (self.0 >> 36) & 0xF {
            0 => EventClass::Geom,
            1 => EventClass::Field,
            2 => EventClass::Mod,
            3 => EventClass::Topo,
            4 => EventClass::EngineSwitch,
            _ => EventClass::Geom, // default
        }
    }

    pub fn tstamp(&self) -> u16 {
        ((self.0 >> 40) & 0xFFFF) as u16
    }

    pub fn qos(&self) -> QosClass {
        match (self.0 >> 56) & 0xF {
            0 => QosClass::Low,
            1 => QosClass::Medium,
            2 => QosClass::High,
            3 => QosClass::Critical,
            _ => QosClass::Low,
        }
    }

    pub fn payload_id(&self) -> u8 {
        ((self.0 >> 60) & 0xF) as u8
    }
}



/// Event semantic classes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EventClass {
    Geom = 0x0,      // Metric updates (edge lengths, curvature)
    Field = 0x1,     // State updates (membranes, fields)
    Mod = 0x2,       // Modulation (learning rates, global params)
    Topo = 0x3,      // Structural events (connect/disconnect/spawn)
    EngineSwitch = 0x4, // Engine handoff events
}

/// QoS classes for latency envelopes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QosClass {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Event-driven locus for causal propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLocus {
    pub id: String,
    pub payload_table: Vec<serde_json::Value>, // Indexed by payload_id
}

impl EventLocus {
    pub fn new(id: String) -> Self {
        Self {
            id,
            payload_table: Vec::new(),
        }
    }

    pub fn add_payload(&mut self, payload: serde_json::Value) -> usize {
        self.payload_table.push(payload);
        self.payload_table.len() - 1
    }
}

/// Causal event fabric for BEIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFabric {
    pub loci: Vec<EventLocus>,
    pub events: Vec<EventPacket>, // Event log for replay
}

impl EventFabric {
    pub fn new() -> Self {
        Self {
            loci: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn add_locus(&mut self, locus: EventLocus) {
        self.loci.push(locus);
    }

    pub fn emit_event(&mut self, event: EventPacket) {
        self.events.push(event);
    }

    /// Replay events for deterministic execution.
    pub fn replay(&self) -> Vec<EventPacket> {
        self.events.clone()
    }
}
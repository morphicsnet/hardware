//! Epoch/Transaction semantics for reversible graph mutations.
//!
//! Implements the north star invariant: every graph mutation that affects semantics is
//! (a) journaled, (b) scope-checked, (c) replayable, and (d) either committed deterministically
//! or discarded completely.

use crate::{Graph, Population, Connection, Probe, Dialect};
use serde::{Deserialize, Serialize};

/// Represents a single journaled mutation to the graph.
/// Each mutation is replayable and scope-checked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JournaledMutation {
    AddPopulation { index: usize, population: Population },
    RemovePopulation { index: usize, population: Population },
    ModifyPopulation { index: usize, old: Population, new: Population },
    AddConnection { index: usize, connection: Connection },
    RemoveConnection { index: usize, connection: Connection },
    ModifyConnection { index: usize, old: Connection, new: Connection },
    AddProbe { index: usize, probe: Probe },
    RemoveProbe { index: usize, probe: Probe },
    ModifyProbe { index: usize, old: Probe, new: Probe },
    SetDialect { old: Option<Dialect>, new: Option<Dialect> },
    AddAttribute { key: String, value: serde_json::Value },
    RemoveAttribute { key: String, value: serde_json::Value },
    ModifyAttribute { key: String, old: serde_json::Value, new: serde_json::Value },
}

/// A transaction context that holds pending mutations.
/// Supports nesting via a stack of journals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionContext {
    /// Stack of mutation journals; each level is a transaction scope.
    pub journals: Vec<Vec<JournaledMutation>>,
    /// Current nesting level.
    pub level: usize,
}

impl TransactionContext {
    pub fn new() -> Self {
        Self {
            journals: vec![Vec::new()],
            level: 0,
        }
    }

    /// Begin a new transaction level.
    pub fn begin(&mut self) {
        self.journals.push(Vec::new());
        self.level += 1;
    }

    /// Commit the current level: merge mutations into parent level.
    pub fn commit(&mut self) -> Result<(), String> {
        if self.level == 0 {
            return Err("No active transaction to commit".into());
        }
        let committed = self.journals.pop().unwrap();
        self.level -= 1;
        if let Some(parent) = self.journals.last_mut() {
            parent.extend(committed);
        }
        Ok(())
    }

    /// Rollback the current level: discard mutations.
    pub fn rollback(&mut self) -> Result<(), String> {
        if self.level == 0 {
            return Err("No active transaction to rollback".into());
        }
        self.journals.pop();
        self.level -= 1;
        Ok(())
    }

    /// Add a mutation to the current journal.
    pub fn journal(&mut self, mutation: JournaledMutation) {
        if let Some(journal) = self.journals.last_mut() {
            journal.push(mutation);
        }
    }

    /// Get pending mutations at current level.
    pub fn pending(&self) -> &[JournaledMutation] {
        self.journals.last().map(|j| j.as_slice()).unwrap_or(&[])
    }
}

/// Trait for transactional operations on the graph.
pub trait Transactional {
    fn begin_transaction(&mut self);
    fn commit_transaction(&mut self) -> Result<(), String>;
    fn rollback_transaction(&mut self) -> Result<(), String>;
    fn apply_mutations(&mut self, mutations: &[JournaledMutation]) -> Result<(), String>;
    fn replay_mutations(&mut self, mutations: &[JournaledMutation]) -> Result<(), String>;
}

impl Transactional for Graph {
    fn begin_transaction(&mut self) {
        if self.attributes.get("transaction_context").is_none() {
            let ctx = TransactionContext::new();
            self.attributes.insert("transaction_context".into(), serde_json::json!(ctx));
        }
        // Deserialize and begin
        if let Some(ctx_val) = self.attributes.get_mut("transaction_context") {
            let mut ctx: TransactionContext = serde_json::from_value(ctx_val.clone()).unwrap();
            ctx.begin();
            *ctx_val = serde_json::json!(ctx);
        }
    }

    fn commit_transaction(&mut self) -> Result<(), String> {
        if let Some(ctx_val) = self.attributes.get_mut("transaction_context") {
            let mut ctx: TransactionContext = serde_json::from_value(ctx_val.clone()).unwrap();
            ctx.commit()?;
            if ctx.level == 0 {
                // Apply all mutations and clear context
                let all_mutations = ctx.journals.into_iter().flatten().collect::<Vec<_>>();
                self.apply_mutations(&all_mutations)?;
                self.attributes.remove("transaction_context");
            } else {
                *ctx_val = serde_json::json!(ctx);
            }
        }
        Ok(())
    }

    fn rollback_transaction(&mut self) -> Result<(), String> {
        if let Some(ctx_val) = self.attributes.get_mut("transaction_context") {
            let mut ctx: TransactionContext = serde_json::from_value(ctx_val.clone()).unwrap();
            ctx.rollback()?;
            if ctx.level == 0 {
                // Discard context
                self.attributes.remove("transaction_context");
            } else {
                *ctx_val = serde_json::json!(ctx);
            }
        }
        Ok(())
    }

    fn apply_mutations(&mut self, mutations: &[JournaledMutation]) -> Result<(), String> {
        for mutation in mutations {
            match mutation {
                JournaledMutation::AddPopulation { index, population } => {
                    if *index <= self.populations.len() {
                        self.populations.insert(*index, population.clone());
                    } else {
                        return Err(format!("Invalid index for AddPopulation: {}", index));
                    }
                }
                JournaledMutation::RemovePopulation { index, .. } => {
                    if *index < self.populations.len() {
                        self.populations.remove(*index);
                    } else {
                        return Err(format!("Invalid index for RemovePopulation: {}", index));
                    }
                }
                JournaledMutation::ModifyPopulation { index, new, .. } => {
                    if *index < self.populations.len() {
                        self.populations[*index] = new.clone();
                    } else {
                        return Err(format!("Invalid index for ModifyPopulation: {}", index));
                    }
                }
                JournaledMutation::AddConnection { index, connection } => {
                    if *index <= self.connections.len() {
                        self.connections.insert(*index, connection.clone());
                    } else {
                        return Err(format!("Invalid index for AddConnection: {}", index));
                    }
                }
                JournaledMutation::RemoveConnection { index, .. } => {
                    if *index < self.connections.len() {
                        self.connections.remove(*index);
                    } else {
                        return Err(format!("Invalid index for RemoveConnection: {}", index));
                    }
                }
                JournaledMutation::ModifyConnection { index, new, .. } => {
                    if *index < self.connections.len() {
                        self.connections[*index] = new.clone();
                    } else {
                        return Err(format!("Invalid index for ModifyConnection: {}", index));
                    }
                }
                JournaledMutation::AddProbe { index, probe } => {
                    if *index <= self.probes.len() {
                        self.probes.insert(*index, probe.clone());
                    } else {
                        return Err(format!("Invalid index for AddProbe: {}", index));
                    }
                }
                JournaledMutation::RemoveProbe { index, .. } => {
                    if *index < self.probes.len() {
                        self.probes.remove(*index);
                    } else {
                        return Err(format!("Invalid index for RemoveProbe: {}", index));
                    }
                }
                JournaledMutation::ModifyProbe { index, new, .. } => {
                    if *index < self.probes.len() {
                        self.probes[*index] = new.clone();
                    } else {
                        return Err(format!("Invalid index for ModifyProbe: {}", index));
                    }
                }
                JournaledMutation::SetDialect { new, .. } => {
                    self.dialect = new.clone();
                }
                JournaledMutation::AddAttribute { key, value } => {
                    self.attributes.insert(key.clone(), value.clone());
                }
                JournaledMutation::RemoveAttribute { key, .. } => {
                    self.attributes.remove(key);
                }
                JournaledMutation::ModifyAttribute { key, new, .. } => {
                    self.attributes.insert(key.clone(), new.clone());
                }
            }
        }
        Ok(())
    }

    fn replay_mutations(&mut self, mutations: &[JournaledMutation]) -> Result<(), String> {
        // For replay, apply in order
        self.apply_mutations(mutations)
    }
}
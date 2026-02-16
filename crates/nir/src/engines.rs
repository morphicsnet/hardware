use crate::Graph;
use anyhow::Result;

pub trait Engine {
    fn select_for_graph(&self, graph: &Graph) -> bool;
    fn apply_transforms(&self, graph: &mut Graph) -> Result<()>;
    fn verify_stability(&self, graph: &Graph) -> Result<()>;
}
//! Computation graph
//!
//! This module provides the computation graph structure for GGML.
//! A computation graph is a directed acyclic graph (DAG) where nodes are tensors
//! and edges represent operations. The graph enables:
//! - Forward pass computation
//! - Backward pass (automatic differentiation)
//! - Graph optimization
//! - Efficient execution scheduling

use crate::tensor::{OpType, Tensor};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Maximum number of nodes in a graph
pub const GGML_MAX_NODES: usize = 16384;

/// Maximum number of leaf nodes (inputs/parameters)
pub const GGML_MAX_LEAFS: usize = 16384;

/// Graph errors
#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Graph is full: maximum {0} nodes reached")]
    GraphFull(usize),
    
    #[error("Cycle detected in computation graph")]
    CycleDetected,
    
    #[error("Node not found in graph")]
    NodeNotFound,
    
    #[error("Invalid graph structure: {0}")]
    InvalidStructure(String),
}

pub type Result<T> = std::result::Result<T, GraphError>;

/// Graph node representing a tensor in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Index in the graph
    pub index: usize,
    
    /// Operation type
    pub op: OpType,
    
    /// Indices of source nodes (inputs to this operation)
    pub sources: Vec<usize>,
    
    /// Whether this is a leaf node (input/parameter)
    pub is_leaf: bool,
    
    /// Whether this node needs gradient computation
    pub requires_grad: bool,
    
    /// Gradient node index (if computed)
    pub grad_index: Option<usize>,
}

/// Computation graph
///
/// A directed acyclic graph (DAG) representing a computation.
/// Nodes are tensors, edges are operations.
pub struct ComputeGraph {
    /// All nodes in the graph
    nodes: Vec<GraphNode>,
    
    /// Leaf nodes (inputs and parameters)
    leafs: Vec<usize>,
    
    /// Mapping from tensor pointer to node index
    tensor_map: HashMap<usize, usize>,
    
    /// Execution order (topologically sorted)
    execution_order: Vec<usize>,
    
    /// Whether the graph has been built
    built: bool,
    
    /// Whether gradients are enabled
    grad_enabled: bool,
}

impl ComputeGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            leafs: Vec::with_capacity(256),
            tensor_map: HashMap::new(),
            execution_order: Vec::new(),
            built: false,
            grad_enabled: false,
        }
    }
    
    /// Create a new graph with gradient computation enabled
    pub fn with_gradients() -> Self {
        let mut graph = Self::new();
        graph.grad_enabled = true;
        graph
    }
    
    /// Add a tensor to the graph
    pub fn add_node(&mut self, tensor: &Tensor, is_leaf: bool) -> Result<usize> {
        if self.nodes.len() >= GGML_MAX_NODES {
            return Err(GraphError::GraphFull(GGML_MAX_NODES));
        }
        
        // Check if tensor already in graph
        let tensor_ptr = tensor as *const Tensor as usize;
        if let Some(&index) = self.tensor_map.get(&tensor_ptr) {
            return Ok(index);
        }
        
        let index = self.nodes.len();
        
        let node = GraphNode {
            index,
            op: tensor.op,
            sources: Vec::new(),
            is_leaf,
            requires_grad: self.grad_enabled, // All nodes participate in gradient computation when enabled
            grad_index: None,
        };
        
        self.nodes.push(node);
        self.tensor_map.insert(tensor_ptr, index);
        
        if is_leaf {
            if self.leafs.len() >= GGML_MAX_LEAFS {
                return Err(GraphError::GraphFull(GGML_MAX_LEAFS));
            }
            self.leafs.push(index);
        }
        
        self.built = false; // Need to rebuild execution order
        Ok(index)
    }
    
    /// Add an edge from source to destination
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<()> {
        if to >= self.nodes.len() {
            return Err(GraphError::NodeNotFound);
        }
        
        self.nodes[to].sources.push(from);
        self.built = false;
        Ok(())
    }
    
    /// Build the graph (compute execution order)
    pub fn build(&mut self) -> Result<()> {
        if self.built {
            return Ok(());
        }
        
        // Topological sort using Kahn's algorithm
        self.execution_order = self.topological_sort()?;
        self.built = true;
        
        Ok(())
    }
    
    /// Perform topological sort to determine execution order
    fn topological_sort(&self) -> Result<Vec<usize>> {
        let n = self.nodes.len();
        
        // Calculate in-degrees
        let mut in_degree = vec![0; n];
        for node in &self.nodes {
            for &_source in &node.sources {
                in_degree[node.index] += 1;
            }
        }
        
        // Queue of nodes with no incoming edges
        let mut queue = VecDeque::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }
        
        let mut order = Vec::with_capacity(n);
        
        while let Some(node_idx) = queue.pop_front() {
            order.push(node_idx);
            
            // Find all nodes that depend on this node
            for i in 0..n {
                if self.nodes[i].sources.contains(&node_idx) {
                    in_degree[i] -= 1;
                    if in_degree[i] == 0 {
                        queue.push_back(i);
                    }
                }
            }
        }
        
        // Check for cycles
        if order.len() != n {
            return Err(GraphError::CycleDetected);
        }
        
        Ok(order)
    }
    
    /// Get the execution order
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }
    
    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get the number of leaf nodes
    pub fn num_leafs(&self) -> usize {
        self.leafs.len()
    }
    
    /// Get a node by index
    pub fn get_node(&self, index: usize) -> Option<&GraphNode> {
        self.nodes.get(index)
    }
    
    /// Get all leaf nodes
    pub fn leafs(&self) -> &[usize] {
        &self.leafs
    }
    
    /// Check if the graph is built
    pub fn is_built(&self) -> bool {
        self.built
    }
    
    /// Check if gradients are enabled
    pub fn has_gradients(&self) -> bool {
        self.grad_enabled
    }
    
    /// Reset the graph
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.leafs.clear();
        self.tensor_map.clear();
        self.execution_order.clear();
        self.built = false;
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let mut num_ops = HashMap::new();
        for node in &self.nodes {
            *num_ops.entry(node.op).or_insert(0) += 1;
        }
        
        GraphStats {
            num_nodes: self.nodes.len(),
            num_leafs: self.leafs.len(),
            num_ops,
            is_built: self.built,
            has_gradients: self.grad_enabled,
        }
    }
    
    /// Validate the graph structure
    pub fn validate(&self) -> Result<()> {
        // Check for invalid source indices
        for node in &self.nodes {
            for &source in &node.sources {
                if source >= self.nodes.len() {
                    return Err(GraphError::InvalidStructure(
                        format!("Invalid source index: {}", source)
                    ));
                }
            }
        }
        
        // Check for cycles (if built)
        if self.built {
            let _ = self.topological_sort()?;
        }
        
        Ok(())
    }
    
    /// Find all nodes that depend on a given node
    pub fn find_dependents(&self, node_idx: usize) -> Vec<usize> {
        let mut dependents = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.sources.contains(&node_idx) {
                dependents.push(i);
            }
        }
        dependents
    }
    
    /// Find all ancestors of a node (nodes it depends on)
    pub fn find_ancestors(&self, node_idx: usize) -> HashSet<usize> {
        let mut ancestors = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_idx);
        
        while let Some(idx) = queue.pop_front() {
            if let Some(node) = self.nodes.get(idx) {
                for &source in &node.sources {
                    if ancestors.insert(source) {
                        queue.push_back(source);
                    }
                }
            }
        }
        
        ancestors
    }
    
    /// Get the depth of the graph (longest path from leaf to any node)
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        
        let mut depths = vec![0; self.nodes.len()];
        
        // Process in topological order
        for &node_idx in &self.execution_order {
            let node = &self.nodes[node_idx];
            let max_source_depth = node.sources.iter()
                .map(|&s| depths[s])
                .max()
                .unwrap_or(0);
            depths[node_idx] = max_source_depth + 1;
        }
        
        *depths.iter().max().unwrap_or(&0)
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_leafs: usize,
    pub num_ops: HashMap<OpType, usize>,
    pub is_built: bool,
    pub has_gradients: bool,
}

impl GraphStats {
    /// Get the total number of operations
    pub fn total_ops(&self) -> usize {
        self.num_ops.values().sum()
    }
    
    /// Get the most common operation
    pub fn most_common_op(&self) -> Option<(OpType, usize)> {
        self.num_ops.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&op, &count)| (op, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::tensor::TensorType;
    use crate::ops;

    #[test]
    fn test_graph_creation() {
        let graph = ComputeGraph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_leafs(), 0);
        assert!(!graph.is_built());
    }

    #[test]
    fn test_add_node() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        let tensor = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let idx = graph.add_node(&tensor, true).unwrap();
        
        assert_eq!(idx, 0);
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.num_leafs(), 1);
    }

    #[test]
    fn test_simple_graph() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        // Create a simple computation: c = a + b
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let c = ops::add(&mut ctx, &a, &b).unwrap();
        
        let idx_a = graph.add_node(&a, true).unwrap();
        let idx_b = graph.add_node(&b, true).unwrap();
        let idx_c = graph.add_node(&c, false).unwrap();
        
        graph.add_edge(idx_a, idx_c).unwrap();
        graph.add_edge(idx_b, idx_c).unwrap();
        
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_leafs(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        // Create: d = (a + b) * c
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let c = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let sum = ops::add(&mut ctx, &a, &b).unwrap();
        let result = ops::mul(&mut ctx, &sum, &c).unwrap();
        
        let idx_a = graph.add_node(&a, true).unwrap();
        let idx_b = graph.add_node(&b, true).unwrap();
        let idx_c = graph.add_node(&c, true).unwrap();
        let idx_sum = graph.add_node(&sum, false).unwrap();
        let idx_result = graph.add_node(&result, false).unwrap();
        
        graph.add_edge(idx_a, idx_sum).unwrap();
        graph.add_edge(idx_b, idx_sum).unwrap();
        graph.add_edge(idx_sum, idx_result).unwrap();
        graph.add_edge(idx_c, idx_result).unwrap();
        
        graph.build().unwrap();
        
        assert!(graph.is_built());
        assert_eq!(graph.execution_order().len(), 5);
        
        // Verify execution order is valid
        let order = graph.execution_order();
        assert!(order.iter().position(|&x| x == idx_a).unwrap() < 
                order.iter().position(|&x| x == idx_sum).unwrap());
        assert!(order.iter().position(|&x| x == idx_b).unwrap() < 
                order.iter().position(|&x| x == idx_sum).unwrap());
        assert!(order.iter().position(|&x| x == idx_sum).unwrap() < 
                order.iter().position(|&x| x == idx_result).unwrap());
    }

    #[test]
    fn test_graph_stats() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let c = ops::add(&mut ctx, &a, &b).unwrap();
        
        graph.add_node(&a, true).unwrap();
        graph.add_node(&b, true).unwrap();
        graph.add_node(&c, false).unwrap();
        
        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_leafs, 2);
    }

    #[test]
    fn test_graph_depth() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        // Create a chain: d = ((a + b) + c)
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let c = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let ab = ops::add(&mut ctx, &a, &b).unwrap();
        let abc = ops::add(&mut ctx, &ab, &c).unwrap();
        
        let idx_a = graph.add_node(&a, true).unwrap();
        let idx_b = graph.add_node(&b, true).unwrap();
        let idx_c = graph.add_node(&c, true).unwrap();
        let idx_ab = graph.add_node(&ab, false).unwrap();
        let idx_abc = graph.add_node(&abc, false).unwrap();
        
        graph.add_edge(idx_a, idx_ab).unwrap();
        graph.add_edge(idx_b, idx_ab).unwrap();
        graph.add_edge(idx_ab, idx_abc).unwrap();
        graph.add_edge(idx_c, idx_abc).unwrap();
        
        graph.build().unwrap();
        
        let depth = graph.depth();
        assert_eq!(depth, 3); // Leaf -> intermediate -> result
    }

    #[test]
    fn test_find_dependents() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let c = ops::add(&mut ctx, &a, &b).unwrap();
        let d = ops::mul(&mut ctx, &a, &b).unwrap();
        
        let idx_a = graph.add_node(&a, true).unwrap();
        let idx_b = graph.add_node(&b, true).unwrap();
        let idx_c = graph.add_node(&c, false).unwrap();
        let idx_d = graph.add_node(&d, false).unwrap();
        
        graph.add_edge(idx_a, idx_c).unwrap();
        graph.add_edge(idx_b, idx_c).unwrap();
        graph.add_edge(idx_a, idx_d).unwrap();
        graph.add_edge(idx_b, idx_d).unwrap();
        
        let dependents = graph.find_dependents(idx_a);
        assert_eq!(dependents.len(), 2);
        assert!(dependents.contains(&idx_c));
        assert!(dependents.contains(&idx_d));
    }

    #[test]
    fn test_graph_reset() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut graph = ComputeGraph::new();
        
        let a = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        graph.add_node(&a, true).unwrap();
        
        assert_eq!(graph.num_nodes(), 1);
        
        graph.reset();
        
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_leafs(), 0);
        assert!(!graph.is_built());
    }

    #[test]
    fn test_gradient_graph() {
        let graph = ComputeGraph::with_gradients();
        assert!(graph.has_gradients());
    }
}

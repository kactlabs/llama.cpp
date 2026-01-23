//! Automatic differentiation and backward pass implementation
//!
//! This module provides automatic differentiation capabilities for the computation graph,
//! enabling gradient computation through backpropagation.

use crate::context::Context;
use crate::graph::{ComputeGraph, GraphNode};
use crate::ops::{self, OpError};
use crate::tensor::{OpType, Tensor};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during autodiff operations
#[derive(Debug, Error)]
pub enum AutodiffError {
    #[error("Graph error: {0}")]
    Graph(#[from] crate::graph::GraphError),

    #[error("Operation error: {0}")]
    Op(#[from] OpError),

    #[error("Context error: {0}")]
    Context(#[from] crate::context::ContextError),

    #[error("Node {0} not found in graph")]
    NodeNotFound(usize),

    #[error("Node {0} does not require gradients")]
    NoGradient(usize),

    #[error("Gradient not computed for node {0}")]
    GradientNotComputed(usize),

    #[error("Unsupported operation for backward pass: {0:?}")]
    UnsupportedOp(OpType),

    #[error("Invalid gradient shape for node {0}")]
    InvalidGradientShape(usize),
}

pub type Result<T> = std::result::Result<T, AutodiffError>;

/// Gradient accumulator for managing gradients during backward pass
pub struct GradientAccumulator {
    /// Map from node index to gradient tensor
    gradients: HashMap<usize, Tensor>,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
        }
    }

    /// Set the gradient for a node
    pub fn set_gradient(&mut self, node_idx: usize, gradient: Tensor) {
        self.gradients.insert(node_idx, gradient);
    }

    /// Get the gradient for a node
    pub fn get_gradient(&self, node_idx: usize) -> Option<&Tensor> {
        self.gradients.get(&node_idx)
    }

    /// Accumulate gradient for a node (add to existing gradient)
    pub fn accumulate_gradient(
        &mut self,
        ctx: &mut Context,
        node_idx: usize,
        gradient: Tensor,
    ) -> Result<()> {
        if let Some(existing) = self.gradients.get(&node_idx) {
            // Add new gradient to existing
            let accumulated = ops::add(ctx, existing, &gradient)?;
            self.gradients.insert(node_idx, accumulated);
        } else {
            // First gradient for this node
            self.gradients.insert(node_idx, gradient);
        }
        Ok(())
    }

    /// Check if gradient exists for a node
    pub fn has_gradient(&self, node_idx: usize) -> bool {
        self.gradients.contains_key(&node_idx)
    }

    /// Get all gradients
    pub fn gradients(&self) -> &HashMap<usize, Tensor> {
        &self.gradients
    }

    /// Clear all gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl Default for GradientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward pass implementation
pub struct BackwardPass<'a> {
    graph: &'a ComputeGraph,
    ctx: &'a mut Context,
    accumulator: GradientAccumulator,
    /// Map from node index to tensor (needed for gradient computation)
    tensors: HashMap<usize, Tensor>,
}

impl<'a> BackwardPass<'a> {
    /// Create a new backward pass
    pub fn new(graph: &'a ComputeGraph, ctx: &'a mut Context) -> Self {
        Self {
            graph,
            ctx,
            accumulator: GradientAccumulator::new(),
            tensors: HashMap::new(),
        }
    }

    /// Register a tensor for a node (needed for gradient computation)
    pub fn register_tensor(&mut self, node_idx: usize, tensor: Tensor) {
        self.tensors.insert(node_idx, tensor);
    }

    /// Run backward pass from a given output node
    ///
    /// # Arguments
    /// * `output_idx` - Index of the output node to backpropagate from
    /// * `output_grad` - Optional gradient of the output (defaults to ones)
    pub fn backward(&mut self, output_idx: usize, output_grad: Option<Tensor>) -> Result<()> {
        // Get the output node
        let _output_node = self
            .graph
            .get_node(output_idx)
            .ok_or(AutodiffError::NodeNotFound(output_idx))?;

        // Initialize output gradient (default to ones if not provided)
        let grad = if let Some(g) = output_grad {
            g
        } else {
            // Create ones tensor - for now just create a placeholder
            // In real implementation, this would be filled with 1.0
            let tensor = self.tensors.get(&output_idx)
                .ok_or(AutodiffError::NodeNotFound(output_idx))?;
            // Create tensor with same type and shape
            let shape: Vec<usize> = tensor.ne[..tensor.n_dims()].to_vec();
            self.ctx.new_tensor(tensor.tensor_type, &shape)?
        };

        self.accumulator.set_gradient(output_idx, grad);

        // Get reverse execution order (topological sort reversed)
        let execution_order = self.graph.execution_order();
        let reverse_order: Vec<_> = execution_order.iter().rev().copied().collect();

        // Backpropagate through each node
        for &node_idx in &reverse_order {
            let node = self
                .graph
                .get_node(node_idx)
                .ok_or(AutodiffError::NodeNotFound(node_idx))?;

            // Skip if this node doesn't require gradients
            if !node.requires_grad {
                continue;
            }

            // Skip if no gradient has been computed yet
            if !self.accumulator.has_gradient(node_idx) {
                continue;
            }

            // Get the gradient for this node
            let node_grad = self
                .accumulator
                .get_gradient(node_idx)
                .ok_or(AutodiffError::GradientNotComputed(node_idx))?
                .clone();

            // Compute gradients for input nodes
            self.backward_node(node, &node_grad)?;
        }

        Ok(())
    }

    /// Compute gradients for a single node's inputs
    fn backward_node(&mut self, node: &GraphNode, grad_output: &Tensor) -> Result<()> {
        match &node.op {
            OpType::None => {
                // Leaf node - no backward pass needed
                Ok(())
            }

            OpType::Add => {
                // d/dx (x + y) = 1, d/dy (x + y) = 1
                // Gradient flows equally to both inputs
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_output.clone())?;
                Ok(())
            }

            OpType::Mul => {
                // d/dx (x * y) = y, d/dy (x * y) = x
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let x = self.tensors.get(&node.sources[0])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[0]))?;
                let y = self.tensors.get(&node.sources[1])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[1]))?;

                // grad_x = grad_output * y
                let grad_x = ops::mul(self.ctx, grad_output, y)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_x)?;

                // grad_y = grad_output * x
                let grad_y = ops::mul(self.ctx, grad_output, x)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_y)?;

                Ok(())
            }

            OpType::Sub => {
                // d/dx (x - y) = 1, d/dy (x - y) = -1
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // grad_x = grad_output
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;

                // grad_y = -grad_output
                let grad_y = ops::neg(self.ctx, grad_output)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_y)?;

                Ok(())
            }

            OpType::Div => {
                // d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let x = self.tensors.get(&node.sources[0])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[0]))?;
                let y = self.tensors.get(&node.sources[1])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[1]))?;

                // grad_x = grad_output / y
                let grad_x = ops::div(self.ctx, grad_output, y)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_x)?;

                // grad_y = -grad_output * x / y^2
                let y_squared = ops::sqr(self.ctx, y)?;
                let numerator = ops::mul(self.ctx, grad_output, x)?;
                let grad_y_pos = ops::div(self.ctx, &numerator, &y_squared)?;
                let grad_y = ops::neg(self.ctx, &grad_y_pos)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_y)?;

                Ok(())
            }

            OpType::MulMat => {
                // Matrix multiplication in GGML convention: C = A @ B
                // Where A[a0,a1] @ B[b0,b1] = C[b0,a1] (requires a0 == b1)
                // 
                // For gradients, we need:
                // grad_A[a0,a1] and grad_B[b0,b1] from grad_C[b0,a1]
                //
                // Working out the shapes:
                // grad_A[a0,a1]: need result with ne[0]=a0, ne[1]=a1
                //   In GGML matmul X[x0,x1] @ Y[y0,y1] = Z[y0,x1] where x0==y1
                //   So: ?[a0,?] @ ?[?,a1] where ?.ne[0] == ?.ne[1]
                //   grad_C[b0,a1] @ B^T[b1,b0] => check b0==b0 ✓, result [b1,a1]=[a0,a1] ✓
                //   So: grad_A = grad_C @ transpose(B)
                //
                // grad_B[b0,b1]: need result with ne[0]=b0, ne[1]=b1  
                //   ?[b0,?] @ ?[?,b1] where ?.ne[0] == ?.ne[1]
                //   A^T[a1,a0] @ grad_C[b0,a1] => check a1==b0? NO
                //   grad_C^T[a1,b0] @ A[a0,a1] => check a1==a1 ✓, result [a0,b0]? NO, result is [a1,a1]
                //   Let me try: grad_C^T[a1,b0] @ A^T[a1,a0] => check a1==a1 ✓, result [a0,b0]=[b1,b0]? NO
                //   Hmm, try: A^T[a1,a0] @ grad_C^T[a1,b0] => check a1==a1 ✓, result [b0,a0]=[b0,b1]? Only if a0==b1 which is true!
                //   So: grad_B = transpose(A) @ transpose(grad_C)
                
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let a = self.tensors.get(&node.sources[0])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[0]))?;
                let b = self.tensors.get(&node.sources[1])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[1]))?;

                // grad_a = grad_output @ transpose(b)
                let b_t = ops::transpose(self.ctx, b)?;
                let grad_a = ops::matmul(self.ctx, grad_output, &b_t)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_a)?;

                // grad_b = transpose(a) @ grad_output
                let a_t = ops::transpose(self.ctx, a)?;
                let grad_b = ops::matmul(self.ctx, &a_t, grad_output)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_b)?;

                Ok(())
            }

            OpType::Sqrt => {
                // d/dx sqrt(x) = 1 / (2 * sqrt(x))
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let output = self.tensors.get(&node.index)
                    .ok_or(AutodiffError::NodeNotFound(node.index))?;
                let two_sqrt_x = ops::scale(self.ctx, output, 2.0)?;
                let grad_input = ops::div(self.ctx, grad_output, &two_sqrt_x)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_input)?;
                Ok(())
            }

            OpType::Sqr => {
                // d/dx (x^2) = 2x
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let input = self.tensors.get(&node.sources[0])
                    .ok_or(AutodiffError::NodeNotFound(node.sources[0]))?;
                let two_x = ops::scale(self.ctx, input, 2.0)?;
                let grad_input = ops::mul(self.ctx, grad_output, &two_x)?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_input)?;
                Ok(())
            }

            OpType::Unary => {
                // For unary operations, pass gradient through (simplified)
                // TODO: Implement proper gradients for each unary op
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::RmsNorm | OpType::Norm => {
                // Normalization gradient (simplified for now)
                // TODO: Implement proper normalization gradients
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Sum | OpType::Mean | OpType::SumRows => {
                // Reduction operations (simplified)
                // TODO: Implement proper reduction gradients with broadcasting
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Reshape | OpType::View | OpType::Transpose | OpType::Permute => {
                // Shape operations - gradient flows through with shape adjustment
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Dup | OpType::Scale | OpType::Cpy | OpType::Cont => {
                // Copy/scale operations
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Rope => {
                // RoPE backward
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // Get the forward tensor to access op_params
                let forward_tensor = self.tensors.get(&node.index)
                    .ok_or(AutodiffError::NodeNotFound(node.index))?;
                let n_past = forward_tensor.op_params[0] as usize;
                let n_dims = forward_tensor.op_params[1] as usize;
                let mode = forward_tensor.op_params[2];
                let n_ctx = forward_tensor.op_params[3] as usize;
                
                let grad_input = crate::ops_advanced::rope_back(
                    self.ctx,
                    grad_output,
                    n_past,
                    n_dims,
                    mode,
                    n_ctx,
                )?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_input)?;
                Ok(())
            }

            OpType::SoftMax => {
                // Softmax backward
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                let output = self.tensors.get(&node.index)
                    .ok_or(AutodiffError::NodeNotFound(node.index))?;
                let grad_input = crate::ops_advanced::soft_max_back(
                    self.ctx,
                    grad_output,
                    output,
                )?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_input)?;
                Ok(())
            }

            OpType::Flash => {
                // Flash attention backward
                // This is complex and requires all Q, K, V tensors
                // For now, simplified version
                if node.sources.len() != 3 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // TODO: Implement proper flash attention backward
                // For now, just propagate gradient to all inputs
                for &source in &node.sources {
                    self.accumulator
                        .accumulate_gradient(self.ctx, source, grad_output.clone())?;
                }
                Ok(())
            }

            OpType::GetRows => {
                // Embedding lookup backward
                // Gradient only flows to the embedding table (first source)
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // TODO: Implement proper get_rows backward (scatter operation)
                // For now, simplified
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Concat => {
                // Concatenation backward - split gradient
                if node.sources.len() != 2 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // TODO: Implement proper concat backward (split along axis)
                // For now, simplified - just pass gradient to both
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[1], grad_output.clone())?;
                Ok(())
            }

            OpType::Clamp => {
                // Clamp backward: gradient passes through where input is in range
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // TODO: Implement proper clamp backward with masking
                // For now, simplified
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::Leaky => {
                // Leaky ReLU backward
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // TODO: Implement proper leaky relu backward
                // For now, simplified
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            OpType::DiagMaskInf | OpType::DiagMaskZero => {
                // Masking operations - gradient flows through masked positions
                if node.sources.len() != 1 {
                    return Err(AutodiffError::UnsupportedOp(node.op));
                }
                // Gradient is zero for masked positions, passes through for others
                self.accumulator
                    .accumulate_gradient(self.ctx, node.sources[0], grad_output.clone())?;
                Ok(())
            }

            _ => {
                // Unsupported operation - return error
                Err(AutodiffError::UnsupportedOp(node.op))
            }
        }
    }

    /// Get the computed gradients
    pub fn gradients(&self) -> &HashMap<usize, Tensor> {
        self.accumulator.gradients()
    }

    /// Get gradient for a specific node
    pub fn get_gradient(&self, node_idx: usize) -> Option<&Tensor> {
        self.accumulator.get_gradient(node_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::graph::ComputeGraph;
    use crate::ops;
    use crate::tensor::TensorType;

    #[test]
    fn test_gradient_accumulator() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let mut acc = GradientAccumulator::new();

        let t1 = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        let t2 = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();

        acc.set_gradient(0, t1);
        assert!(acc.has_gradient(0));
        assert!(!acc.has_gradient(1));

        acc.accumulate_gradient(&mut ctx, 1, t2).unwrap();
        assert!(acc.has_gradient(1));
    }

    #[test]
    fn test_backward_simple() -> Result<()> {
        let mut ctx = Context::new(10 * 1024 * 1024)?;
        let mut graph = ComputeGraph::with_gradients();

        // Simple: z = x (identity, leaf node)
        let x = ctx.new_tensor_1d(TensorType::F32, 5)?;
        let idx_x = graph.add_node(&x, true)?;

        graph.build()?;

        // Create explicit gradient for x
        let grad_x = ctx.new_tensor_1d(TensorType::F32, 5)?;
        
        let mut backward = BackwardPass::new(&graph, &mut ctx);
        backward.register_tensor(idx_x, x.clone());
        backward.accumulator.set_gradient(idx_x, grad_x);

        // Check gradient exists
        assert!(backward.get_gradient(idx_x).is_some());

        Ok(())
    }

    #[test]
    fn test_backward_add() -> Result<()> {
        let mut ctx = Context::new(10 * 1024 * 1024)?;
        let mut graph = ComputeGraph::with_gradients();

        // Create inputs
        let x = ctx.new_tensor_1d(TensorType::F32, 5)?;
        let y = ctx.new_tensor_1d(TensorType::F32, 5)?;

        let idx_x = graph.add_node(&x, true)?;
        let idx_y = graph.add_node(&y, true)?;

        // z = x + y
        let z = ops::add(&mut ctx, &x, &y)?;
        let idx_z = graph.add_node(&z, false)?;
        graph.add_edge(idx_x, idx_z)?;
        graph.add_edge(idx_y, idx_z)?;

        graph.build()?;

        // Create explicit output gradient
        let grad_z = ctx.new_tensor_1d(TensorType::F32, 5)?;

        // Backward pass
        let mut backward = BackwardPass::new(&graph, &mut ctx);
        backward.register_tensor(idx_x, x);
        backward.register_tensor(idx_y, y);
        backward.register_tensor(idx_z, z);
        backward.backward(idx_z, Some(grad_z))?;

        // Check gradients exist
        assert!(backward.get_gradient(idx_x).is_some());
        assert!(backward.get_gradient(idx_y).is_some());

        Ok(())
    }

    #[test]
    fn test_backward_mul() -> Result<()> {
        let mut ctx = Context::new(10 * 1024 * 1024)?;
        let mut graph = ComputeGraph::with_gradients();

        let x = ctx.new_tensor_1d(TensorType::F32, 5)?;
        let y = ctx.new_tensor_1d(TensorType::F32, 5)?;

        let idx_x = graph.add_node(&x, true)?;
        let idx_y = graph.add_node(&y, true)?;

        // z = x * y
        let z = ops::mul(&mut ctx, &x, &y)?;
        let idx_z = graph.add_node(&z, false)?;
        graph.add_edge(idx_x, idx_z)?;
        graph.add_edge(idx_y, idx_z)?;

        graph.build()?;

        // Create explicit output gradient
        let grad_z = ctx.new_tensor_1d(TensorType::F32, 5)?;

        let mut backward = BackwardPass::new(&graph, &mut ctx);
        backward.register_tensor(idx_x, x);
        backward.register_tensor(idx_y, y);
        backward.register_tensor(idx_z, z);
        backward.backward(idx_z, Some(grad_z))?;

        assert!(backward.get_gradient(idx_x).is_some());
        assert!(backward.get_gradient(idx_y).is_some());

        Ok(())
    }

    #[test]
    fn test_backward_chain() -> Result<()> {
        let mut ctx = Context::new(10 * 1024 * 1024)?;
        let mut graph = ComputeGraph::with_gradients();

        // x -> y = x^2 -> z = y + x
        let x = ctx.new_tensor_1d(TensorType::F32, 5)?;
        let idx_x = graph.add_node(&x, true)?;

        let y = ops::sqr(&mut ctx, &x)?;
        let idx_y = graph.add_node(&y, false)?;
        graph.add_edge(idx_x, idx_y)?;

        let z = ops::add(&mut ctx, &y, &x)?;
        let idx_z = graph.add_node(&z, false)?;
        graph.add_edge(idx_y, idx_z)?;
        graph.add_edge(idx_x, idx_z)?;

        graph.build()?;

        // Create explicit output gradient
        let grad_z = ctx.new_tensor_1d(TensorType::F32, 5)?;

        let mut backward = BackwardPass::new(&graph, &mut ctx);
        backward.register_tensor(idx_x, x);
        backward.register_tensor(idx_y, y);
        backward.register_tensor(idx_z, z);
        backward.backward(idx_z, Some(grad_z))?;

        // x should have gradient (accumulated from both paths)
        assert!(backward.get_gradient(idx_x).is_some());

        Ok(())
    }

    #[test]
    fn test_backward_matmul() -> Result<()> {
        let mut ctx = Context::new(10 * 1024 * 1024)?;
        let mut graph = ComputeGraph::with_gradients();

        // C = A @ B in GGML convention
        // In GGML: A[a0,a1] @ B[b0,b1] = C[b0,a1] where a0 == b1
        // To get standard A[3,4] @ B[4,5] = C[3,5]:
        // We need C[3,5] which means b0=3, a1=5
        // Inner dimension 4 means a0=4, b1=4
        // So: A is [4,5], B is [3,4]
        let a = ctx.new_tensor_2d(TensorType::F32, 4, 5)?;  // a.ne[0]=4, a.ne[1]=5
        let b = ctx.new_tensor_2d(TensorType::F32, 3, 4)?;  // b.ne[0]=3, b.ne[1]=4

        let idx_a = graph.add_node(&a, true)?;
        let idx_b = graph.add_node(&b, true)?;

        let c = ops::matmul(&mut ctx, &a, &b)?;  // c will be [3, 5]
        let idx_c = graph.add_node(&c, false)?;
        graph.add_edge(idx_a, idx_c)?;
        graph.add_edge(idx_b, idx_c)?;

        graph.build()?;

        // Create explicit output gradient with correct shape [3, 5]
        let grad_c = ctx.new_tensor_2d(TensorType::F32, 3, 5)?;

        let mut backward = BackwardPass::new(&graph, &mut ctx);
        backward.register_tensor(idx_a, a);
        backward.register_tensor(idx_b, b);
        backward.register_tensor(idx_c, c);
        backward.backward(idx_c, Some(grad_c))?;

        assert!(backward.get_gradient(idx_a).is_some());
        assert!(backward.get_gradient(idx_b).is_some());

        Ok(())
    }
}

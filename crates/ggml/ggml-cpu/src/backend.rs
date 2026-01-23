//! CPU backend implementation
//!
//! Provides the backend interface for CPU execution

use ggml_core::{Context, Tensor, ComputeGraph};
use thiserror::Error;

/// CPU backend errors
#[derive(Debug, Error)]
pub enum CpuBackendError {
    #[error("Computation error: {0}")]
    ComputeError(String),
    
    #[error("Unsupported operation: {0:?}")]
    UnsupportedOp(ggml_core::OpType),
    
    #[error("Invalid tensor data")]
    InvalidData,
    
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        expected: ggml_core::TensorType,
        got: ggml_core::TensorType,
    },
}

pub type Result<T> = std::result::Result<T, CpuBackendError>;

/// CPU backend for executing computation graphs
pub struct CpuBackend {
    /// Number of threads to use (0 = auto-detect)
    num_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }
    
    /// Create a CPU backend with specific number of threads
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads: if num_threads == 0 {
                num_cpus::get()
            } else {
                num_threads
            },
        }
    }
    
    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
    
    /// Execute a computation graph
    pub fn compute_graph(&self, graph: &ComputeGraph, ctx: &mut Context) -> Result<()> {
        // Get execution order
        let order = graph.execution_order();
        
        // Store tensors by node index for easy lookup
        let mut tensors: std::collections::HashMap<usize, Tensor> = std::collections::HashMap::new();
        
        // Execute each node in order
        for &node_idx in order {
            let node = graph.get_node(node_idx)
                .ok_or_else(|| CpuBackendError::ComputeError(
                    format!("Node {} not found", node_idx)
                ))?;
            
            // Skip leaf nodes (they're inputs, already have data)
            if node.is_leaf {
                continue;
            }
            
            // Execute the operation
            let result = self.compute_node(node, &tensors, ctx)?;
            tensors.insert(node_idx, result);
        }
        
        Ok(())
    }
    
    /// Compute a single node
    fn compute_node(
        &self,
        node: &ggml_core::GraphNode,
        tensors: &std::collections::HashMap<usize, Tensor>,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        use ggml_core::OpType;
        
        match node.op {
            OpType::None => {
                // Leaf node - should not be called
                Err(CpuBackendError::ComputeError(
                    "Cannot compute leaf node".to_string()
                ))
            }
            
            OpType::Add => {
                if node.sources.len() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "Add requires 2 sources".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                let b = tensors.get(&node.sources[1])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 1 not found".to_string()))?;
                
                // Create result tensor
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                // Get data slices - need to get raw pointers to avoid borrow conflicts
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let b_ptr = b.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let b_data = std::slice::from_raw_parts(b_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    // Compute
                    crate::compute::CpuCompute::add_f32(a_data, b_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::Mul => {
                if node.sources.len() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "Mul requires 2 sources".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                let b = tensors.get(&node.sources[1])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 1 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let b_ptr = b.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let b_data = std::slice::from_raw_parts(b_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::mul_f32(a_data, b_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::MulMat => {
                if node.sources.len() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "MulMat requires 2 sources".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                let b = tensors.get(&node.sources[1])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 1 not found".to_string()))?;
                
                // Result shape in GGML convention: [b.ne[0], a.ne[1]]
                let result_shape = [b.ne[0], a.ne[1]];
                let result = ctx.new_tensor(a.tensor_type, &result_shape)
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let b_ptr = b.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let a_shape = [a.ne[0], a.ne[1]];
                let b_shape = [b.ne[0], b.ne[1]];
                let result_elements = result.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, a.n_elements());
                    let b_data = std::slice::from_raw_parts(b_ptr.as_ptr() as *const f32, b.n_elements());
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, result_elements);
                    
                    crate::compute::CpuCompute::matmul_f32(
                        a_data, a_shape,
                        b_data, b_shape,
                        result_data
                    )?;
                }
                
                Ok(result)
            }
            
            OpType::Sub => {
                if node.sources.len() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "Sub requires 2 sources".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                let b = tensors.get(&node.sources[1])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 1 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let b_ptr = b.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let b_data = std::slice::from_raw_parts(b_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::sub_f32(a_data, b_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::Div => {
                if node.sources.len() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "Div requires 2 sources".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                let b = tensors.get(&node.sources[1])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 1 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let b_ptr = b.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let b_data = std::slice::from_raw_parts(b_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::div_f32(a_data, b_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::Silu => {
                if node.sources.len() != 1 {
                    return Err(CpuBackendError::ComputeError(
                        "Silu requires 1 source".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::silu_f32(a_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::RmsNorm => {
                if node.sources.len() != 1 {
                    return Err(CpuBackendError::ComputeError(
                        "RmsNorm requires 1 source".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                // Use default epsilon (we'll need to pass this properly later)
                let eps = 1e-5f32;
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::rms_norm_f32(a_data, result_data, eps)?;
                }
                
                Ok(result)
            }
            
            OpType::SoftMax => {
                if node.sources.len() != 1 {
                    return Err(CpuBackendError::ComputeError(
                        "SoftMax requires 1 source".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                
                let result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                let n_elements = a.n_elements();
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, n_elements);
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, n_elements);
                    
                    crate::compute::CpuCompute::softmax_f32(a_data, result_data)?;
                }
                
                Ok(result)
            }
            
            OpType::Transpose => {
                if node.sources.len() != 1 {
                    return Err(CpuBackendError::ComputeError(
                        "Transpose requires 1 source".to_string()
                    ));
                }
                
                let a = tensors.get(&node.sources[0])
                    .ok_or_else(|| CpuBackendError::ComputeError("Source 0 not found".to_string()))?;
                
                if a.n_dims() != 2 {
                    return Err(CpuBackendError::ComputeError(
                        "Transpose requires 2D tensor".to_string()
                    ));
                }
                
                let result_shape = [a.ne[1], a.ne[0]];
                let result = ctx.new_tensor(a.tensor_type, &result_shape)
                    .map_err(|e| CpuBackendError::ComputeError(e.to_string()))?;
                
                let a_ptr = a.data().ok_or(CpuBackendError::InvalidData)?;
                let result_ptr = result.data().ok_or(CpuBackendError::InvalidData)?;
                
                unsafe {
                    let a_data = std::slice::from_raw_parts(a_ptr.as_ptr() as *const f32, a.n_elements());
                    let result_data = std::slice::from_raw_parts_mut(result_ptr.as_ptr() as *mut f32, result.n_elements());
                    
                    crate::compute::CpuCompute::transpose_f32(a_data, [a.ne[0], a.ne[1]], result_data)?;
                }
                
                Ok(result)
            }
            
            _ => Err(CpuBackendError::UnsupportedOp(node.op)),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backend_creation() {
        let backend = CpuBackend::new();
        assert!(backend.num_threads() > 0);
    }
    
    #[test]
    fn test_backend_with_threads() {
        let backend = CpuBackend::with_threads(4);
        assert_eq!(backend.num_threads(), 4);
    }
    
    #[test]
    fn test_backend_auto_threads() {
        let backend = CpuBackend::with_threads(0);
        assert!(backend.num_threads() > 0);
    }
}

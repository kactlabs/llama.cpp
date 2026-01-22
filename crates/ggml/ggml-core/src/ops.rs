//! Tensor operations
//!
//! This module provides the core tensor operations for GGML.
//! Operations are organized into categories:
//! - Element-wise: add, mul, sub, div
//! - Unary: neg, abs, sqrt, sqr
//! - Reduction: sum, mean, max, min
//! - Matrix: matmul, transpose

use crate::context::{Context, ContextError};
use crate::tensor::{OpType, Tensor, TensorType};
use thiserror::Error;

/// Operation errors
#[derive(Error, Debug)]
pub enum OpError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        expected: TensorType,
        got: TensorType,
    },
    
    #[error("Invalid dimensions for operation: {0}")]
    InvalidDimensions(String),
    
    #[error("Unsupported operation for type {0:?}")]
    UnsupportedType(TensorType),
    
    #[error("Context error: {0}")]
    Context(#[from] ContextError),
}

pub type Result<T> = std::result::Result<T, OpError>;

/// Check if two tensors have the same shape
fn check_same_shape(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.ne != b.ne {
        return Err(OpError::ShapeMismatch(format!(
            "Tensors have different shapes: {:?} vs {:?}",
            &a.ne[..a.n_dims()],
            &b.ne[..b.n_dims()]
        )));
    }
    Ok(())
}

/// Check if two tensors have compatible types
fn check_same_type(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.tensor_type != b.tensor_type {
        return Err(OpError::TypeMismatch {
            expected: a.tensor_type,
            got: b.tensor_type,
        });
    }
    Ok(())
}

//
// Element-wise operations
//

/// Add two tensors element-wise: result = a + b
pub fn add(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_shape(a, b)?;
    check_same_type(a, b)?;
    
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Add;
    
    // For now, just create the operation node
    // Actual computation will be done by the backend
    Ok(result)
}

/// Multiply two tensors element-wise: result = a * b
pub fn mul(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_shape(a, b)?;
    check_same_type(a, b)?;
    
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Mul;
    
    Ok(result)
}

/// Subtract two tensors element-wise: result = a - b
pub fn sub(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_shape(a, b)?;
    check_same_type(a, b)?;
    
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Sub;
    
    Ok(result)
}

/// Divide two tensors element-wise: result = a / b
pub fn div(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_shape(a, b)?;
    check_same_type(a, b)?;
    
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Div;
    
    Ok(result)
}

//
// Unary operations
//

/// Negate a tensor: result = -a
pub fn neg(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Unary;
    Ok(result)
}

/// Absolute value: result = |a|
pub fn abs(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Unary;
    Ok(result)
}

/// Square root: result = sqrt(a)
pub fn sqrt(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Sqrt;
    Ok(result)
}

/// Square: result = a^2
pub fn sqr(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Sqr;
    Ok(result)
}

//
// Reduction operations
//

/// Sum all elements: result = sum(a)
pub fn sum(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    // Result is a scalar
    let mut result = ctx.new_tensor(a.tensor_type, &[1])?;
    result.op = OpType::Sum;
    Ok(result)
}

/// Mean of all elements: result = mean(a)
pub fn mean(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    // Result is a scalar
    let mut result = ctx.new_tensor(a.tensor_type, &[1])?;
    result.op = OpType::Mean;
    Ok(result)
}

/// Sum along rows: result[i] = sum(a[i, :])
pub fn sum_rows(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    if a.n_dims() < 2 {
        return Err(OpError::InvalidDimensions(
            "sum_rows requires at least 2D tensor".to_string(),
        ));
    }
    
    // Result has shape [ne[1], ne[2], ne[3], 1]
    let mut shape = a.ne;
    shape[0] = 1;
    
    let mut result = ctx.new_tensor(a.tensor_type, &shape[..a.n_dims()])?;
    result.op = OpType::SumRows;
    Ok(result)
}

/// Maximum value: result = max(a)
pub fn max(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &[1])?;
    result.op = OpType::Argmax; // Will be changed to Max when we add it
    Ok(result)
}

/// Minimum value: result = min(a)
pub fn min(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &[1])?;
    result.op = OpType::Argmax; // Will be changed to Min when we add it
    Ok(result)
}

//
// Matrix operations
//

/// Matrix multiplication: result = a @ b
/// 
/// For 2D tensors: C[i,j] = sum_k A[i,k] * B[k,j]
/// Shapes: [m, k] @ [k, n] -> [m, n]
pub fn matmul(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check dimensions
    if a.n_dims() < 2 || b.n_dims() < 2 {
        return Err(OpError::InvalidDimensions(
            "matmul requires at least 2D tensors".to_string(),
        ));
    }
    
    // Check inner dimensions match: a.ne[0] == b.ne[1]
    if a.ne[0] != b.ne[1] {
        return Err(OpError::ShapeMismatch(format!(
            "Inner dimensions don't match: {} vs {}",
            a.ne[0], b.ne[1]
        )));
    }
    
    check_same_type(a, b)?;
    
    // Result shape: [b.ne[0], a.ne[1], a.ne[2], a.ne[3]]
    let mut shape = [1; 4];
    shape[0] = b.ne[0];
    shape[1] = a.ne[1];
    if a.n_dims() > 2 {
        shape[2] = a.ne[2];
    }
    if a.n_dims() > 3 {
        shape[3] = a.ne[3];
    }
    
    let n_dims = a.n_dims();
    let mut result = ctx.new_tensor(a.tensor_type, &shape[..n_dims])?;
    result.op = OpType::MulMat;
    
    Ok(result)
}

/// Transpose a 2D tensor: result = a^T
pub fn transpose(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    if a.n_dims() != 2 {
        return Err(OpError::InvalidDimensions(
            "transpose requires 2D tensor".to_string(),
        ));
    }
    
    // Swap dimensions
    let shape = [a.ne[1], a.ne[0]];
    let mut result = ctx.new_tensor(a.tensor_type, &shape)?;
    result.op = OpType::Transpose;
    
    Ok(result)
}

/// Reshape a tensor (must preserve total number of elements)
pub fn reshape(ctx: &mut Context, a: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
    let old_elements = a.n_elements();
    let new_elements: usize = new_shape.iter().product();
    
    if old_elements != new_elements {
        return Err(OpError::InvalidDimensions(format!(
            "Cannot reshape: element count mismatch ({} vs {})",
            old_elements, new_elements
        )));
    }
    
    let mut result = ctx.new_tensor(a.tensor_type, new_shape)?;
    result.op = OpType::Reshape;
    
    Ok(result)
}

/// Create a view of a tensor (shares data, different shape/strides)
pub fn view(ctx: &mut Context, a: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, new_shape)?;
    result.op = OpType::View;
    
    // Copy the data pointer (view shares data)
    if let Some(data) = a.data() {
        unsafe {
            result.set_data(data);
        }
    }
    
    Ok(result)
}

/// Duplicate a tensor (deep copy)
pub fn dup(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Dup;
    Ok(result)
}

/// Scale a tensor by a scalar: result = a * scale
pub fn scale(ctx: &mut Context, a: &Tensor, scale: f32) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Scale;
    result.op_params[0] = scale.to_bits() as i32;
    Ok(result)
}

//
// Activation functions
//

/// ReLU activation: result = max(0, a)
pub fn relu(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Unary;
    Ok(result)
}

/// GELU activation (Gaussian Error Linear Unit)
pub fn gelu(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Unary;
    Ok(result)
}

/// SiLU activation (Sigmoid Linear Unit): result = a * sigmoid(a)
pub fn silu(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Silu;
    Ok(result)
}

/// Softmax: result[i] = exp(a[i]) / sum(exp(a))
pub fn softmax(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::SoftMax;
    Ok(result)
}

//
// Normalization
//

/// RMS normalization
pub fn rms_norm(ctx: &mut Context, a: &Tensor, eps: f32) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::RmsNorm;
    result.op_params[0] = eps.to_bits() as i32;
    Ok(result)
}

/// Layer normalization
pub fn norm(ctx: &mut Context, a: &Tensor, eps: f32) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Norm;
    result.op_params[0] = eps.to_bits() as i32;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorType;

    #[test]
    fn test_add() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        
        let c = add(&mut ctx, &a, &b).unwrap();
        assert_eq!(c.n_elements(), 200);
        assert_eq!(c.op, OpType::Add);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = ctx.new_tensor_2d(TensorType::F32, 10, 30).unwrap();
        
        let result = add(&mut ctx, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        let c = mul(&mut ctx, &a, &b).unwrap();
        assert_eq!(c.n_elements(), 100);
        assert_eq!(c.op, OpType::Mul);
    }

    #[test]
    fn test_unary_ops() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        let neg_a = neg(&mut ctx, &a).unwrap();
        assert_eq!(neg_a.n_elements(), 100);
        
        let abs_a = abs(&mut ctx, &a).unwrap();
        assert_eq!(abs_a.n_elements(), 100);
        
        let sqrt_a = sqrt(&mut ctx, &a).unwrap();
        assert_eq!(sqrt_a.n_elements(), 100);
        assert_eq!(sqrt_a.op, OpType::Sqrt);
        
        let sqr_a = sqr(&mut ctx, &a).unwrap();
        assert_eq!(sqr_a.n_elements(), 100);
        assert_eq!(sqr_a.op, OpType::Sqr);
    }

    #[test]
    fn test_reduction_ops() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        
        let sum_a = sum(&mut ctx, &a).unwrap();
        assert_eq!(sum_a.n_elements(), 1);
        assert_eq!(sum_a.op, OpType::Sum);
        
        let mean_a = mean(&mut ctx, &a).unwrap();
        assert_eq!(mean_a.n_elements(), 1);
        assert_eq!(mean_a.op, OpType::Mean);
    }

    #[test]
    fn test_matmul() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        // [10, 20] @ [30, 10] -> [30, 20]
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = ctx.new_tensor_2d(TensorType::F32, 30, 10).unwrap();
        
        let c = matmul(&mut ctx, &a, &b).unwrap();
        assert_eq!(c.ne[0], 30);
        assert_eq!(c.ne[1], 20);
        assert_eq!(c.op, OpType::MulMat);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = ctx.new_tensor_2d(TensorType::F32, 30, 15).unwrap(); // Wrong inner dim
        
        let result = matmul(&mut ctx, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = transpose(&mut ctx, &a).unwrap();
        
        assert_eq!(b.ne[0], 20);
        assert_eq!(b.ne[1], 10);
        assert_eq!(b.op, OpType::Transpose);
    }

    #[test]
    fn test_reshape() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = reshape(&mut ctx, &a, &[5, 40]).unwrap();
        
        assert_eq!(b.ne[0], 5);
        assert_eq!(b.ne[1], 40);
        assert_eq!(b.n_elements(), 200);
    }

    #[test]
    fn test_reshape_invalid() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let result = reshape(&mut ctx, &a, &[5, 50]); // Wrong element count
        
        assert!(result.is_err());
    }

    #[test]
    fn test_scale() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        let b = scale(&mut ctx, &a, 2.5).unwrap();
        
        assert_eq!(b.n_elements(), 100);
        assert_eq!(b.op, OpType::Scale);
    }

    #[test]
    fn test_activation_functions() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        let relu_a = relu(&mut ctx, &a).unwrap();
        assert_eq!(relu_a.n_elements(), 100);
        
        let gelu_a = gelu(&mut ctx, &a).unwrap();
        assert_eq!(gelu_a.n_elements(), 100);
        
        let silu_a = silu(&mut ctx, &a).unwrap();
        assert_eq!(silu_a.n_elements(), 100);
        assert_eq!(silu_a.op, OpType::Silu);
    }

    #[test]
    fn test_normalization() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        
        let rms = rms_norm(&mut ctx, &a, 1e-5).unwrap();
        assert_eq!(rms.n_elements(), 200);
        assert_eq!(rms.op, OpType::RmsNorm);
        
        let norm_a = norm(&mut ctx, &a, 1e-5).unwrap();
        assert_eq!(norm_a.n_elements(), 200);
        assert_eq!(norm_a.op, OpType::Norm);
    }
}

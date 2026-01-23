//! Advanced tensor operations for neural networks
//!
//! This module provides advanced operations needed for modern neural networks,
//! particularly transformer architectures:
//! - RoPE (Rotary Position Embedding)
//! - Attention mechanisms
//! - Advanced activations with proper gradients
//! - Layer normalization variants

use crate::context::Context;
use crate::ops::{OpError, Result};
use crate::tensor::{OpType, Tensor};

/// RoPE (Rotary Position Embedding) operation
///
/// Applies rotary position embeddings to a tensor, used in models like LLaMA, Mistral, Qwen.
/// This is a key component for positional encoding in modern transformers.
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor [batch, seq_len, n_heads, head_dim]
/// * `n_past` - Number of past tokens (for KV cache)
/// * `n_dims` - Number of dimensions to apply RoPE to (typically head_dim or head_dim/2)
/// * `mode` - RoPE mode (0 = normal, 1 = neox style)
/// * `n_ctx` - Context size
///
/// # Returns
/// Tensor with rotary position embeddings applied
pub fn rope(
    ctx: &mut Context,
    a: &Tensor,
    n_past: usize,
    n_dims: usize,
    mode: i32,
    n_ctx: usize,
) -> Result<Tensor> {
    if a.n_dims() < 3 {
        return Err(OpError::InvalidDimensions(
            "RoPE requires at least 3D tensor".to_string(),
        ));
    }

    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Rope;
    
    // Store RoPE parameters
    result.op_params[0] = n_past as i32;
    result.op_params[1] = n_dims as i32;
    result.op_params[2] = mode;
    result.op_params[3] = n_ctx as i32;

    Ok(result)
}

/// RoPE backward pass
pub fn rope_back(
    ctx: &mut Context,
    a: &Tensor,
    n_past: usize,
    n_dims: usize,
    mode: i32,
    n_ctx: usize,
) -> Result<Tensor> {
    if a.n_dims() < 3 {
        return Err(OpError::InvalidDimensions(
            "RoPE backward requires at least 3D tensor".to_string(),
        ));
    }

    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::RopeBack;
    
    result.op_params[0] = n_past as i32;
    result.op_params[1] = n_dims as i32;
    result.op_params[2] = mode;
    result.op_params[3] = n_ctx as i32;

    Ok(result)
}

/// Softmax with numerical stability
///
/// Computes softmax along the last dimension with proper numerical stability
/// using the log-sum-exp trick: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
///
/// # Returns
/// Tensor with softmax applied along last dimension
pub fn soft_max(ctx: &mut Context, a: &Tensor) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::SoftMax;
    Ok(result)
}

/// Softmax backward pass
pub fn soft_max_back(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.n_dims() != b.n_dims() {
        return Err(OpError::ShapeMismatch(
            "Softmax backward requires matching dimensions".to_string(),
        ));
    }

    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::SoftMaxBack;
    Ok(result)
}

/// Flash Attention
///
/// Efficient attention implementation with O(N) memory complexity.
/// Computes: softmax(Q @ K^T / sqrt(d)) @ V
///
/// # Arguments
/// * `ctx` - Memory context
/// * `q` - Query tensor [batch, n_heads, seq_len_q, head_dim]
/// * `k` - Key tensor [batch, n_heads, seq_len_k, head_dim]
/// * `v` - Value tensor [batch, n_heads, seq_len_k, head_dim]
/// * `masked` - Whether to apply causal masking
///
/// # Returns
/// Attention output [batch, n_heads, seq_len_q, head_dim]
pub fn flash_attn(
    ctx: &mut Context,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    masked: bool,
) -> Result<Tensor> {
    if q.n_dims() != 4 || k.n_dims() != 4 || v.n_dims() != 4 {
        return Err(OpError::InvalidDimensions(
            "Flash attention requires 4D tensors".to_string(),
        ));
    }

    // Check dimensions match
    if q.ne[0] != k.ne[0] || k.ne[0] != v.ne[0] {
        return Err(OpError::ShapeMismatch(
            "Head dimension must match across Q, K, V".to_string(),
        ));
    }

    if k.ne[1] != v.ne[1] {
        return Err(OpError::ShapeMismatch(
            "Sequence length must match for K and V".to_string(),
        ));
    }

    // Result shape: [head_dim, seq_len_q, n_heads, batch]
    let shape = [q.ne[0], q.ne[1], q.ne[2], q.ne[3]];
    let mut result = ctx.new_tensor(q.tensor_type, &shape)?;
    result.op = OpType::Flash;
    result.op_params[0] = masked as i32;

    Ok(result)
}

/// Flash Attention backward pass
pub fn flash_attn_back(
    ctx: &mut Context,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d: &Tensor,
    masked: bool,
) -> Result<Tensor> {
    if q.n_dims() != 4 || k.n_dims() != 4 || v.n_dims() != 4 || d.n_dims() != 4 {
        return Err(OpError::InvalidDimensions(
            "Flash attention backward requires 4D tensors".to_string(),
        ));
    }

    let shape = [q.ne[0], q.ne[1], q.ne[2], q.ne[3]];
    let mut result = ctx.new_tensor(q.tensor_type, &shape)?;
    result.op = OpType::FlashBack;
    result.op_params[0] = masked as i32;

    Ok(result)
}

/// Scaled Dot-Product Attention (standard implementation)
///
/// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # Arguments
/// * `ctx` - Memory context
/// * `q` - Query tensor [batch, n_heads, seq_len_q, head_dim]
/// * `k` - Key tensor [batch, n_heads, seq_len_k, head_dim]
/// * `v` - Value tensor [batch, n_heads, seq_len_k, head_dim]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
/// * `mask` - Optional attention mask
///
/// # Returns
/// Attention output [batch, n_heads, seq_len_q, head_dim]
pub fn scaled_dot_product_attention(
    ctx: &mut Context,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // Q @ K^T
    let qk = crate::ops::matmul(ctx, q, k)?;
    
    // Scale
    let qk_scaled = crate::ops::scale(ctx, &qk, scale)?;
    
    // Add mask if provided
    let qk_masked = if let Some(m) = mask {
        crate::ops::add(ctx, &qk_scaled, m)?
    } else {
        qk_scaled
    };
    
    // Softmax
    let attn_weights = soft_max(ctx, &qk_masked)?;
    
    // @ V
    let output = crate::ops::matmul(ctx, &attn_weights, v)?;
    
    Ok(output)
}

/// Clamp values to a range
///
/// Clamps all values in the tensor to [min, max]
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
/// * `min` - Minimum value
/// * `max` - Maximum value
pub fn clamp(ctx: &mut Context, a: &Tensor, min: f32, max: f32) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Clamp;
    
    // Store min/max as op params (reinterpret as i32)
    result.op_params[0] = min.to_bits() as i32;
    result.op_params[1] = max.to_bits() as i32;

    Ok(result)
}

/// Leaky ReLU activation
///
/// f(x) = x if x > 0, else alpha * x
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
/// * `alpha` - Negative slope (typically 0.01)
pub fn leaky_relu(ctx: &mut Context, a: &Tensor, alpha: f32) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::Leaky;
    result.op_params[0] = alpha.to_bits() as i32;
    Ok(result)
}

/// Alibi (Attention with Linear Biases)
///
/// Adds position-dependent bias to attention scores
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor (attention scores)
/// * `n_past` - Number of past tokens
/// * `n_head` - Number of attention heads
/// * `bias_max` - Maximum bias value
pub fn alibi(
    ctx: &mut Context,
    a: &Tensor,
    n_past: usize,
    n_head: usize,
    bias_max: f32,
) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::DiagMaskInf;
    
    result.op_params[0] = n_past as i32;
    result.op_params[1] = n_head as i32;
    result.op_params[2] = bias_max.to_bits() as i32;

    Ok(result)
}

/// Diagonal mask with infinity
///
/// Sets values above diagonal to -inf (for causal masking)
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
/// * `n_past` - Number of past tokens
pub fn diag_mask_inf(ctx: &mut Context, a: &Tensor, n_past: usize) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::DiagMaskInf;
    result.op_params[0] = n_past as i32;
    Ok(result)
}

/// Diagonal mask with zero
///
/// Sets values above diagonal to 0 (for causal masking)
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
/// * `n_past` - Number of past tokens
pub fn diag_mask_zero(ctx: &mut Context, a: &Tensor, n_past: usize) -> Result<Tensor> {
    let mut result = ctx.new_tensor(a.tensor_type, &a.ne[..a.n_dims()])?;
    result.op = OpType::DiagMaskZero;
    result.op_params[0] = n_past as i32;
    Ok(result)
}

/// Get rows from a tensor (embedding lookup)
///
/// Extracts specific rows from a 2D tensor, used for embedding lookups
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor [vocab_size, embed_dim]
/// * `b` - Row indices [batch_size, seq_len]
pub fn get_rows(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.n_dims() != 2 {
        return Err(OpError::InvalidDimensions(
            "get_rows requires 2D input tensor".to_string(),
        ));
    }

    // Result shape: [a.ne[0], b.ne[0], b.ne[1], ...]
    let mut shape = [1; 4];
    shape[0] = a.ne[0]; // embed_dim
    shape[1] = b.ne[0]; // batch or seq_len
    if b.n_dims() > 1 {
        shape[2] = b.ne[1];
    }
    if b.n_dims() > 2 {
        shape[3] = b.ne[2];
    }

    let n_dims = 1 + b.n_dims();
    let mut result = ctx.new_tensor(a.tensor_type, &shape[..n_dims])?;
    result.op = OpType::GetRows;

    Ok(result)
}

/// Permute tensor dimensions
///
/// Rearranges the dimensions of a tensor according to the given permutation
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - Input tensor
/// * `axis0` - New position for dimension 0
/// * `axis1` - New position for dimension 1
/// * `axis2` - New position for dimension 2
/// * `axis3` - New position for dimension 3
pub fn permute(
    ctx: &mut Context,
    a: &Tensor,
    axis0: usize,
    axis1: usize,
    axis2: usize,
    axis3: usize,
) -> Result<Tensor> {
    // Validate axes
    let axes = [axis0, axis1, axis2, axis3];
    for &axis in &axes {
        if axis >= 4 {
            return Err(OpError::InvalidDimensions(
                format!("Invalid axis: {}", axis),
            ));
        }
    }

    // Check for duplicates
    for i in 0..4 {
        for j in (i + 1)..4 {
            if axes[i] == axes[j] {
                return Err(OpError::InvalidDimensions(
                    "Duplicate axes in permutation".to_string(),
                ));
            }
        }
    }

    // Compute new shape
    let mut new_shape = [1; 4];
    new_shape[0] = a.ne[axis0];
    new_shape[1] = a.ne[axis1];
    new_shape[2] = a.ne[axis2];
    new_shape[3] = a.ne[axis3];

    let mut result = ctx.new_tensor(a.tensor_type, &new_shape[..a.n_dims()])?;
    result.op = OpType::Permute;
    
    result.op_params[0] = axis0 as i32;
    result.op_params[1] = axis1 as i32;
    result.op_params[2] = axis2 as i32;
    result.op_params[3] = axis3 as i32;

    Ok(result)
}

/// Concatenate tensors along a dimension
///
/// # Arguments
/// * `ctx` - Memory context
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `axis` - Dimension to concatenate along
pub fn concat(ctx: &mut Context, a: &Tensor, b: &Tensor, axis: usize) -> Result<Tensor> {
    if a.n_dims() != b.n_dims() {
        return Err(OpError::ShapeMismatch(
            "Tensors must have same number of dimensions".to_string(),
        ));
    }

    if axis >= a.n_dims() {
        return Err(OpError::InvalidDimensions(
            format!("Invalid axis: {}", axis),
        ));
    }

    // Check all dimensions except concat axis match
    for i in 0..a.n_dims() {
        if i != axis && a.ne[i] != b.ne[i] {
            return Err(OpError::ShapeMismatch(
                format!("Dimension {} doesn't match: {} vs {}", i, a.ne[i], b.ne[i]),
            ));
        }
    }

    // Result shape: same as a, but concat dimension is sum
    let mut shape = a.ne;
    shape[axis] = a.ne[axis] + b.ne[axis];

    let mut result = ctx.new_tensor(a.tensor_type, &shape[..a.n_dims()])?;
    result.op = OpType::Concat;
    result.op_params[0] = axis as i32;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::tensor::TensorType;

    #[test]
    fn test_rope() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        // [batch=2, seq_len=10, n_heads=8, head_dim=64]
        let a = ctx.new_tensor_4d(TensorType::F32, 64, 10, 8, 2).unwrap();
        
        let result = rope(&mut ctx, &a, 0, 64, 0, 2048).unwrap();
        assert_eq!(result.ne[0], 64);
        assert_eq!(result.ne[1], 10);
        assert_eq!(result.op, OpType::Rope);
    }

    #[test]
    fn test_soft_max() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        
        let result = soft_max(&mut ctx, &a).unwrap();
        assert_eq!(result.ne[0], 10);
        assert_eq!(result.ne[1], 20);
        assert_eq!(result.op, OpType::SoftMax);
    }

    #[test]
    fn test_flash_attn() {
        let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
        
        // [head_dim=64, seq_len=128, n_heads=8, batch=2]
        let q = ctx.new_tensor_4d(TensorType::F32, 64, 128, 8, 2).unwrap();
        let k = ctx.new_tensor_4d(TensorType::F32, 64, 128, 8, 2).unwrap();
        let v = ctx.new_tensor_4d(TensorType::F32, 64, 128, 8, 2).unwrap();
        
        let result = flash_attn(&mut ctx, &q, &k, &v, true).unwrap();
        assert_eq!(result.ne[0], 64);
        assert_eq!(result.ne[1], 128);
        assert_eq!(result.op, OpType::Flash);
    }

    #[test]
    fn test_clamp() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        let result = clamp(&mut ctx, &a, -1.0, 1.0).unwrap();
        assert_eq!(result.ne[0], 100);
        assert_eq!(result.op, OpType::Clamp);
    }

    #[test]
    fn test_get_rows() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        // Embedding table: [vocab_size=50000, embed_dim=768]
        let embeddings = ctx.new_tensor_2d(TensorType::F32, 768, 50000).unwrap();
        
        // Token indices: [batch=4, seq_len=128]
        let indices = ctx.new_tensor_2d(TensorType::I32, 128, 4).unwrap();
        
        let result = get_rows(&mut ctx, &embeddings, &indices).unwrap();
        assert_eq!(result.ne[0], 768);  // embed_dim
        assert_eq!(result.ne[1], 128);  // seq_len
        assert_eq!(result.ne[2], 4);    // batch
        assert_eq!(result.op, OpType::GetRows);
    }

    #[test]
    fn test_permute() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        // [2, 3, 4, 5]
        let a = ctx.new_tensor_4d(TensorType::F32, 2, 3, 4, 5).unwrap();
        
        // Permute to [3, 2, 5, 4] (swap 0<->1 and 2<->3)
        let result = permute(&mut ctx, &a, 1, 0, 3, 2).unwrap();
        assert_eq!(result.ne[0], 3);
        assert_eq!(result.ne[1], 2);
        assert_eq!(result.ne[2], 5);
        assert_eq!(result.ne[3], 4);
        assert_eq!(result.op, OpType::Permute);
    }

    #[test]
    fn test_concat() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let a = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        let b = ctx.new_tensor_2d(TensorType::F32, 10, 30).unwrap();
        
        // Concatenate along axis 1
        let result = concat(&mut ctx, &a, &b, 1).unwrap();
        assert_eq!(result.ne[0], 10);
        assert_eq!(result.ne[1], 50);  // 20 + 30
        assert_eq!(result.op, OpType::Concat);
    }

    #[test]
    fn test_leaky_relu() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        let a = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        let result = leaky_relu(&mut ctx, &a, 0.01).unwrap();
        assert_eq!(result.ne[0], 100);
        assert_eq!(result.op, OpType::Leaky);
    }
}

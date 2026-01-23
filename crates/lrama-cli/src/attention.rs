//! Multi-head attention mechanism with KV caching

use anyhow::Result;
use ggml_cpu::compute::CpuCompute;
use crate::kv_cache::KVCache;

/// Apply RoPE (Rotary Position Embedding) to query and key
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    pos: usize,
    rope_freq_base: f32,
) -> Result<()> {
    let n_rot = head_dim; // Usually full head_dim
    
    // Apply RoPE to query heads
    for head_idx in 0..n_head {
        let q_head_start = head_idx * head_dim;
        
        // Apply rotation to pairs of dimensions
        for dim_pair in 0..(n_rot / 2) {
            let i0 = dim_pair * 2;
            let i1 = i0 + 1;
            
            // Calculate rotation angle
            let theta = pos as f32 * rope_freq_base.powf(-2.0 * dim_pair as f32 / n_rot as f32);
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            
            // Rotate Q
            let q0 = q[q_head_start + i0];
            let q1 = q[q_head_start + i1];
            q[q_head_start + i0] = q0 * cos_theta - q1 * sin_theta;
            q[q_head_start + i1] = q0 * sin_theta + q1 * cos_theta;
        }
    }
    
    // Apply RoPE to key heads (may be fewer than query heads)
    for head_idx in 0..n_head_kv {
        let k_head_start = head_idx * head_dim;
        
        // Apply rotation to pairs of dimensions
        for dim_pair in 0..(n_rot / 2) {
            let i0 = dim_pair * 2;
            let i1 = i0 + 1;
            
            // Calculate rotation angle
            let theta = pos as f32 * rope_freq_base.powf(-2.0 * dim_pair as f32 / n_rot as f32);
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            
            // Rotate K
            let k0 = k[k_head_start + i0];
            let k1 = k[k_head_start + i1];
            k[k_head_start + i0] = k0 * cos_theta - k1 * sin_theta;
            k[k_head_start + i1] = k0 * sin_theta + k1 * cos_theta;
        }
    }
    
    Ok(())
}

/// Multi-head attention with KV caching
pub fn attention(
    q: &[f32],              // Query: [n_embd]
    k: &[f32],              // Key: [n_embd_kv]
    v: &[f32],              // Value: [n_embd_kv]
    kv_cache: &mut KVCache,
    layer_idx: usize,
    pos: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    let head_dim = n_embd / n_head;
    let n_embd_kv = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Store current K, V in cache
    kv_cache.store(layer_idx, pos, k, v)?;
    
    // Get all keys and values up to current position
    let n_tokens = pos + 1;
    let all_keys = kv_cache.get_keys(layer_idx, n_tokens);
    let all_values = kv_cache.get_values(layer_idx, n_tokens);
    
    // Output: [n_embd]
    let mut output = vec![0.0f32; n_embd];
    
    // Process each query head
    // For grouped-query attention, multiple query heads share the same KV head
    let group_size = n_head / n_head_kv;
    
    for head_idx in 0..n_head {
        let kv_head_idx = head_idx / group_size;
        
        let q_head_start = head_idx * head_dim;
        let q_head = &q[q_head_start..q_head_start + head_dim];
        
        // Compute attention scores for this head
        let mut scores = vec![0.0f32; n_tokens];
        
        for token_idx in 0..n_tokens {
            let k_start = token_idx * n_embd_kv + kv_head_idx * head_dim;
            let k_head = &all_keys[k_start..k_start + head_dim];
            
            // Dot product: Q @ K^T
            let mut score = 0.0;
            for i in 0..head_dim {
                score += q_head[i] * k_head[i];
            }
            
            // Scale
            scores[token_idx] = score * scale;
        }
        
        // Apply causal mask (already implicit since we only look at past tokens)
        // Softmax
        let mut scores_out = vec![0.0f32; n_tokens];
        CpuCompute::softmax_f32(&scores, &mut scores_out)?;
        
        // Weighted sum of values
        let out_head_start = head_idx * head_dim;
        for token_idx in 0..n_tokens {
            let v_start = token_idx * n_embd_kv + kv_head_idx * head_dim;
            let v_head = &all_values[v_start..v_start + head_dim];
            let weight = scores_out[token_idx];
            
            for i in 0..head_dim {
                output[out_head_start + i] += weight * v_head[i];
            }
        }
    }
    
    Ok(output)
}

/// RMS normalization
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != weight.len() {
        return Err(anyhow::anyhow!("RMS norm size mismatch"));
    }
    
    let n = x.len();
    
    // Compute RMS
    let mut sum_sq = 0.0;
    for &val in x {
        sum_sq += val * val;
    }
    let rms = ((sum_sq / n as f32) + eps).sqrt();
    
    // Normalize and scale
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        output[i] = (x[i] / rms) * weight[i];
    }
    
    Ok(output)
}

/// SwiGLU activation: SiLU(gate) * up
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(anyhow::anyhow!("SwiGLU size mismatch"));
    }
    
    let n = gate.len();
    let mut output = vec![0.0f32; n];
    
    for i in 0..n {
        // SiLU(gate[i]) = gate[i] / (1 + exp(-gate[i]))
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        output[i] = silu * up[i];
    }
    
    Ok(output)
}

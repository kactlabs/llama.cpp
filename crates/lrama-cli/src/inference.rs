//! Forward pass and inference logic

use anyhow::{Context, Result};
use crate::model::{LlamaModel, LlamaLayer};
use crate::kv_cache::KVCache;
use crate::attention::{apply_rope, attention, rms_norm, swiglu};

/// Run forward pass for a single token
pub fn forward(
    model: &LlamaModel,
    token: u32,
    pos: usize,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>> {
    let n_embd = model.n_embd;
    let n_head = model.n_head;
    let n_head_kv = model.n_head_kv;
    
    // 1. Token embedding
    let mut x = get_embedding(model, token)?;
    
    if x.iter().any(|v| !v.is_finite()) {
        return Err(anyhow::anyhow!("NaN/Inf in embedding for token {}", token));
    }
    
    // 2. Process through each transformer layer
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        x = forward_layer(
            &x,
            layer,
            layer_idx,
            pos,
            kv_cache,
            n_head,
            n_head_kv,
            n_embd,
        )?;
        
        if x.iter().any(|v| !v.is_finite()) {
            return Err(anyhow::anyhow!("NaN/Inf after layer {}", layer_idx));
        }
    }
    
    // 3. Final RMS normalization
    x = rms_norm(&x, &model.output_norm_weight, 1e-5)?;
    
    if x.iter().any(|v| !v.is_finite()) {
        return Err(anyhow::anyhow!("NaN/Inf after final norm"));
    }
    
    // 4. Output projection (logits)
    let logits = output_projection(model, &x)?;
    
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(anyhow::anyhow!("NaN/Inf in output logits"));
    }
    
    Ok(logits)
}

/// Get token embedding
fn get_embedding(model: &LlamaModel, token: u32) -> Result<Vec<f32>> {
    let n_embd = model.n_embd;
    let n_vocab = model.n_vocab;
    let token_idx = token as usize;
    
    if token_idx >= n_vocab {
        return Err(anyhow::anyhow!("Token index out of bounds"));
    }
    
    // token_embd is stored as [n_embd, n_vocab] in GGUF dimensions
    // But in memory, it's actually ROW-MAJOR: [n_embd][n_vocab]
    // So to get embedding for token i: embedding[d] = token_embd[d * n_vocab + i]
    // This extracts column i from the matrix
    
    let mut embedding = vec![0.0f32; n_embd];
    for emb_dim in 0..n_embd {
        let idx = emb_dim * n_vocab + token_idx;
        if idx >= model.token_embd.len() {
            return Err(anyhow::anyhow!("Token embedding index out of bounds"));
        }
        embedding[emb_dim] = model.token_embd[idx];
    }
    
    // Debug: check if embeddings are reasonable
    if token == 1 {
        let non_zero = embedding.iter().filter(|&&x| x != 0.0).count();
        let sum: f32 = embedding.iter().sum();
        let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!("Token 1 embedding: {} non-zero, sum={:.6}, range=[{:.6}, {:.6}]", 
                 non_zero, sum, min, max);
    }
    
    Ok(embedding)
}

/// Forward pass through one transformer layer
fn forward_layer(
    x: &[f32],
    layer: &LlamaLayer,
    layer_idx: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    n_head: usize,
    n_head_kv: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    let head_dim = n_embd / n_head;
    let n_embd_kv = n_head_kv * head_dim;
    
    // 1. Attention block
    // Pre-norm
    let x_norm = rms_norm(x, &layer.attn_norm_weight, 1e-5)
        .with_context(|| format!("Layer {} attn_norm", layer_idx))?;
    
    if layer_idx == 0 {
        let sum: f32 = x_norm.iter().sum();
        let max = x_norm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = x_norm.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!("Layer 0 x_norm: sum={:.6}, range=[{:.6}, {:.6}]", sum, min, max);
        
        // Check wq weights
        let wq_sum: f32 = layer.wq.iter().sum();
        let wq_max = layer.wq.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let wq_min = layer.wq.iter().copied().fold(f32::INFINITY, f32::min);
        let wq_nan = layer.wq.iter().filter(|x| !x.is_finite()).count();
        eprintln!("Layer 0 wq: sum={:.6}, range=[{:.6}, {:.6}], nan_count={}", 
                 wq_sum, wq_min, wq_max, wq_nan);
    }
    
    // Project to Q, K, V
    let q = matmul_vec(&layer.wq, &x_norm, n_embd, n_embd)
        .with_context(|| format!("Layer {} wq", layer_idx))?;
    
    if layer_idx == 0 {
        let q_sum: f32 = q.iter().sum();
        let q_max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let q_min = q.iter().copied().fold(f32::INFINITY, f32::min);
        let q_nan = q.iter().filter(|x| !x.is_finite()).count();
        eprintln!("Layer 0 q after matmul: sum={:.6}, range=[{:.6}, {:.6}], nan_count={}", 
                 q_sum, q_min, q_max, q_nan);
        
        if q_nan > 0 {
            return Err(anyhow::anyhow!("NaN/Inf after wq in layer 0"));
        }
    }
    
    let k = matmul_vec(&layer.wk, &x_norm, n_embd_kv, n_embd)
        .with_context(|| format!("Layer {} wk", layer_idx))?;
    
    let v = matmul_vec(&layer.wv, &x_norm, n_embd_kv, n_embd)
        .with_context(|| format!("Layer {} wv", layer_idx))?;
    // Apply RoPE
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    apply_rope(&mut q_rope, &mut k_rope, n_head, n_head_kv, head_dim, pos, 10000.0)?;
    
    // Multi-head attention
    let attn_out = attention(
        &q_rope,
        &k_rope,
        &v,
        kv_cache,
        layer_idx,
        pos,
        n_head,
        n_head_kv,
        n_embd,
    )?;
    
    // Output projection
    let attn_proj = matmul_vec(&layer.wo, &attn_out, n_embd, n_embd)?;
    
    // Residual connection
    let mut x = x.to_vec();
    for i in 0..n_embd {
        x[i] += attn_proj[i];
    }
    
    // 2. Feed-forward block
    // Pre-norm
    let x_norm = rms_norm(&x, &layer.ffn_norm_weight, 1e-5)?;
    
    // FFN projections
    let n_ff = layer.w1.len() / n_embd;
    let gate = matmul_vec(&layer.w1, &x_norm, n_ff, n_embd)?;
    let up = matmul_vec(&layer.w3, &x_norm, n_ff, n_embd)?;
    
    // SwiGLU activation
    let ffn_hidden = swiglu(&gate, &up)?;
    
    // Down projection
    let ffn_out = matmul_vec(&layer.w2, &ffn_hidden, n_embd, n_ff)?;
    
    // Residual connection
    for i in 0..n_embd {
        x[i] += ffn_out[i];
    }
    
    Ok(x)
}

/// Matrix-vector multiplication following GGML convention
/// GGML's ggml_mul_mat(W, x) computes: result = W^T @ x
/// W is stored with dimensions [n_in, n_out] in column-major format
/// So we compute: y[i] = sum_j W[j, i] * x[j]
fn matmul_vec(w: &[f32], x: &[f32], n_out: usize, n_in: usize) -> Result<Vec<f32>> {
    if w.len() != n_out * n_in {
        return Err(anyhow::anyhow!(
            "Weight matrix size mismatch: expected {}x{}={}, got {}",
            n_in, n_out, n_out * n_in, w.len()
        ));
    }
    
    if x.len() != n_in {
        return Err(anyhow::anyhow!(
            "Input vector size mismatch: expected {}, got {}",
            n_in, x.len()
        ));
    }
    
    let mut y = vec![0.0f32; n_out];
    
    // W is [n_in, n_out] in column-major: W[j, i] = w[j + i * n_in]
    // We want: y[i] = sum_j W[j, i] * x[j]
    
    for i in 0..n_out {
        let mut sum = 0.0;
        for j in 0..n_in {
            sum += w[j + i * n_in] * x[j];
        }
        y[i] = sum;
    }
    
    Ok(y)
}

/// Output projection to vocabulary logits
fn output_projection(model: &LlamaModel, x: &[f32]) -> Result<Vec<f32>> {
    // Use token embeddings as output weights (weight tying)
    // Compute logits = token_embd^T @ x
    // token_embd is [n_embd, n_vocab] stored in row-major
    // We want: logits[i] = dot(token_embd[:, i], x)
    // In row-major: column i is at indices [i, i+n_vocab, i+2*n_vocab, ...]
    
    let n_vocab = model.n_vocab;
    let n_embd = model.n_embd;
    
    if x.len() != n_embd {
        return Err(anyhow::anyhow!("Input size mismatch"));
    }
    
    let mut logits = vec![0.0f32; n_vocab];
    
    // For each vocabulary token
    for vocab_idx in 0..n_vocab {
        // Dot product with x
        let mut dot = 0.0;
        for emb_dim in 0..n_embd {
            let embd_idx = emb_dim * n_vocab + vocab_idx;
            dot += model.token_embd[embd_idx] * x[emb_dim];
        }
        logits[vocab_idx] = dot;
    }
    
    Ok(logits)
}

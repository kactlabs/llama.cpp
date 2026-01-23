//! KV Cache for efficient autoregressive generation
//!
//! Stores key and value tensors from previous tokens to avoid recomputation.

use anyhow::Result;

/// KV cache for storing attention keys and values
pub struct KVCache {
    /// Cached keys for each layer: [n_layer][n_head_kv * head_dim * n_ctx]
    pub k_cache: Vec<Vec<f32>>,
    /// Cached values for each layer: [n_layer][n_head_kv * head_dim * n_ctx]
    pub v_cache: Vec<Vec<f32>>,
    /// Maximum context length
    pub n_ctx: usize,
    /// Number of layers
    pub n_layer: usize,
    /// KV embedding dimension (n_head_kv * head_dim)
    pub n_embd_kv: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(n_layer: usize, n_embd_kv: usize, n_ctx: usize) -> Self {
        let cache_size = n_embd_kv * n_ctx;
        
        let k_cache = (0..n_layer)
            .map(|_| vec![0.0f32; cache_size])
            .collect();
        
        let v_cache = (0..n_layer)
            .map(|_| vec![0.0f32; cache_size])
            .collect();
        
        Self {
            k_cache,
            v_cache,
            n_ctx,
            n_layer,
            n_embd_kv,
        }
    }
    
    /// Store key and value for a specific layer and position
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.n_layer {
            return Err(anyhow::anyhow!("Layer index out of bounds"));
        }
        
        if pos >= self.n_ctx {
            return Err(anyhow::anyhow!("Position out of bounds"));
        }
        
        if k.len() != self.n_embd_kv || v.len() != self.n_embd_kv {
            return Err(anyhow::anyhow!("Key/value size mismatch"));
        }
        
        // Store in cache at position
        let offset = pos * self.n_embd_kv;
        self.k_cache[layer][offset..offset + self.n_embd_kv].copy_from_slice(k);
        self.v_cache[layer][offset..offset + self.n_embd_kv].copy_from_slice(v);
        
        Ok(())
    }
    
    /// Get all keys up to current position for a layer
    pub fn get_keys(&self, layer: usize, n_tokens: usize) -> &[f32] {
        let end = n_tokens * self.n_embd_kv;
        &self.k_cache[layer][..end]
    }
    
    /// Get all values up to current position for a layer
    pub fn get_values(&self, layer: usize, n_tokens: usize) -> &[f32] {
        let end = n_tokens * self.n_embd_kv;
        &self.v_cache[layer][..end]
    }
}

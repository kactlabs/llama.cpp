//! LLaMA model architecture

#![allow(dead_code)]

use anyhow::{Context as AnyhowContext, Result};
use ggml_core::{Context, Tensor, TensorType};
use ggml_quants::{dequantize_f32, dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q6_k};
use gguf::{GGMLType, GGUFReader};
use std::io::Write;

pub struct LlamaModel {
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
    
    // Weights (all dequantized to F32)
    pub token_embd: Vec<f32>,
    pub layers: Vec<LlamaLayer>,
    pub output_norm_weight: Vec<f32>,
}

pub struct LlamaLayer {
    // Attention
    pub attn_norm_weight: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    
    // Feed-forward
    pub ffn_norm_weight: Vec<f32>,
    pub w1: Vec<f32>,  // gate
    pub w2: Vec<f32>,  // down
    pub w3: Vec<f32>,  // up
}

impl LlamaModel {
    pub fn from_gguf(reader: &GGUFReader, _ctx: &mut Context) -> Result<Self> {
        println!("Loading model architecture...");
        
        // Read hyperparameters from actual GGUF keys
        let n_embd = reader.get_metadata("llama.embedding_length")
            .and_then(|v| v.as_u32().ok())
            .context("Missing llama.embedding_length")? as usize;
        
        let n_head = reader.get_metadata("llama.attention.head_count")
            .and_then(|v| v.as_u32().ok())
            .context("Missing llama.attention.head_count")? as usize;
        
        let n_layer = reader.get_metadata("llama.block_count")
            .and_then(|v| v.as_u32().ok())
            .context("Missing llama.block_count")? as usize;
        
        let n_ctx = reader.get_metadata("llama.context_length")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(2048) as usize;
        
        // Check for grouped-query attention
        let n_head_kv = reader.get_metadata("llama.attention.head_count_kv")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(n_head as u32) as usize;
        
        // Get vocab size from tokenizer tokens array
        let tokens = reader.get_metadata("tokenizer.ggml.tokens")
            .context("Missing tokenizer.ggml.tokens")?;
        let n_vocab = match tokens {
            gguf::MetadataValue::Array(arr) => arr.values.len(),
            _ => return Err(anyhow::anyhow!("tokenizer.ggml.tokens must be an array")),
        };
        
        println!("  vocab={}, embd={}, heads={}, kv_heads={}, layers={}, ctx={}", 
                 n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ctx);
        
        // Load token embeddings
        println!("Loading token embeddings...");
        let token_embd = load_tensor_data(reader, "token_embd.weight")?;
        
        // Verify size
        let expected_size = n_embd * n_vocab;
        if token_embd.len() != expected_size {
            eprintln!("Warning: token_embd size mismatch. Expected {}, got {}", 
                     expected_size, token_embd.len());
        }
        
        // Check for NaN/Inf in embeddings
        let nan_count = token_embd.iter().filter(|&&x| x.is_nan() || x.is_infinite()).count();
        if nan_count > 0 {
            eprintln!("Warning: {} NaN/Inf values in token embeddings", nan_count);
        }
        
        // Load output norm
        println!("Loading output norm...");
        let output_norm_weight = load_tensor_data(reader, "output_norm.weight")?;
        
        // Load layers
        println!("Loading {} layers...", n_layer);
        let mut layers = Vec::new();
        for i in 0..n_layer {
            eprint!("  Loading layer {}/{}...", i + 1, n_layer);
            std::io::stderr().flush().ok();
            layers.push(LlamaLayer::load(reader, i)?);
            eprintln!(" ✓");
        }
        
        println!("Model loaded successfully!");
        
        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_head_kv,
            n_layer,
            n_ctx,
            token_embd,
            layers,
            output_norm_weight,
        })
    }
}

impl LlamaLayer {
    fn load(reader: &GGUFReader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);
        
        // Try to load each tensor, with better error messages
        let load_weight = |name: &str| -> Result<Vec<f32>> {
            let weights = load_tensor_data(reader, name)
                .with_context(|| format!("Failed to load {}", name))?;
            
            // Check for NaN/Inf in first layer only
            if layer_idx == 0 {
                let nan_count = weights.iter().filter(|x| !x.is_finite()).count();
                if nan_count > 0 {
                    eprintln!("WARNING: {} has {} NaN/Inf values out of {}", 
                             name, nan_count, weights.len());
                }
                
                // Print dimensions and first few values for debugging
                if let Some(info) = reader.get_tensor_info(name) {
                    eprintln!("  {} dimensions: {:?}, {} elements", 
                             name, info.dimensions, weights.len());
                    
                    if name.contains("attn_q.weight") {
                        eprintln!("  First 20 values of {}: {:?}", name, &weights[..20.min(weights.len())]);
                    }
                }
            }
            
            Ok(weights)
        };
        
        Ok(Self {
            attn_norm_weight: load_weight(&format!("{}.attn_norm.weight", prefix))?,
            wq: load_weight(&format!("{}.attn_q.weight", prefix))?,
            wk: load_weight(&format!("{}.attn_k.weight", prefix))?,
            wv: load_weight(&format!("{}.attn_v.weight", prefix))?,
            wo: load_weight(&format!("{}.attn_output.weight", prefix))?,
            
            ffn_norm_weight: load_weight(&format!("{}.ffn_norm.weight", prefix))?,
            w1: load_weight(&format!("{}.ffn_gate.weight", prefix))?,
            w2: load_weight(&format!("{}.ffn_down.weight", prefix))?,
            w3: load_weight(&format!("{}.ffn_up.weight", prefix))?,
        })
    }
}

/// Load a tensor into the context
fn load_tensor(reader: &GGUFReader, ctx: &mut Context, name: &str) -> Result<Tensor> {
    let info = reader.get_tensor_info(name)
        .context(format!("Tensor not found: {}", name))?;
    
    let n_elements = info.n_elements() as usize;
    let tensor = ctx.new_tensor(TensorType::F32, &[n_elements])?;
    
    // Load and dequantize data
    let raw_data = reader.get_tensor_data(name)?;
    let mut f32_data = vec![0.0f32; n_elements];
    
    match info.tensor_type {
        GGMLType::F32 => dequantize_f32(raw_data, &mut f32_data)?,
        GGMLType::Q2_K => dequantize_q2_k(raw_data, &mut f32_data)?,
        GGMLType::Q3_K => dequantize_q3_k(raw_data, &mut f32_data)?,
        GGMLType::Q4_K => dequantize_q4_k(raw_data, &mut f32_data)?,
        GGMLType::Q6_K => dequantize_q6_k(raw_data, &mut f32_data)?,
        _ => return Err(anyhow::anyhow!("Unsupported tensor type: {:?}", info.tensor_type)),
    }
    
    ctx.set_tensor_data_f32(&tensor, &f32_data)?;
    
    Ok(tensor)
}

/// Load tensor data directly (not into context)
fn load_tensor_data(reader: &GGUFReader, name: &str) -> Result<Vec<f32>> {
    let info = reader.get_tensor_info(name)
        .context(format!("Tensor not found: {}", name))?;
    
    let n_elements = info.n_elements() as usize;
    let raw_data = reader.get_tensor_data(name)?;
    let mut f32_data = vec![0.0f32; n_elements];
    
    match info.tensor_type {
        GGMLType::F32 => dequantize_f32(raw_data, &mut f32_data)?,
        GGMLType::Q2_K => dequantize_q2_k(raw_data, &mut f32_data)?,
        GGMLType::Q3_K => dequantize_q3_k(raw_data, &mut f32_data)?,
        GGMLType::Q4_K => dequantize_q4_k(raw_data, &mut f32_data)?,
        GGMLType::Q6_K => dequantize_q6_k(raw_data, &mut f32_data)?,
        _ => return Err(anyhow::anyhow!("Unsupported tensor type: {:?}", info.tensor_type)),
    }
    
    Ok(f32_data)
}

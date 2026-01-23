//! Text generation with sampling

use anyhow::Result;
use crate::model::LlamaModel;
use crate::tokenizer::Tokenizer;
use crate::kv_cache::KVCache;
use crate::inference::forward;

/// Generation configuration
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub n_predict: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            n_predict: 128,
        }
    }
}

/// Generate text from a prompt
pub fn generate(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    println!("\n🚀 Starting generation...");
    
    // 1. Tokenize prompt
    let tokens = tokenizer.encode(prompt, true);
    println!("  Prompt tokens: {} tokens", tokens.len());
    
    // 2. Initialize KV cache
    let head_dim = model.n_embd / model.n_head;
    let n_embd_kv = model.n_head_kv * head_dim;
    let mut kv_cache = KVCache::new(model.n_layer, n_embd_kv, model.n_ctx);
    
    // 3. Process prompt tokens (prefill)
    println!("  Processing prompt...");
    for (pos, &token) in tokens.iter().enumerate() {
        let _logits = forward(model, token, pos, &mut kv_cache)?;
        if pos % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!(" ✓");
    
    // 4. Generate new tokens
    println!("  Generating {} tokens...", config.n_predict);
    let mut generated_tokens = Vec::new();
    
    for i in 0..config.n_predict {
        let pos = tokens.len() + i;
        
        if pos >= model.n_ctx {
            println!("\n  Reached context limit");
            break;
        }
        
        // Get last token
        let last_token = if i == 0 {
            *tokens.last().unwrap()
        } else {
            *generated_tokens.last().unwrap()
        };
        
        // Forward pass
        let logits = forward(model, last_token, pos, &mut kv_cache)?;
        
        // Debug first token
        if i == 0 {
            let logits_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let logits_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!("First token logits: min={:.3}, max={:.3}", logits_min, logits_max);
            
            // Show top 10 tokens with their text
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("Top 10 logits:");
            for (idx, logit) in indexed.iter().take(10) {
                let token_text = tokenizer.decode(&[*idx as u32]);
                eprintln!("  Token {}: {:.3} = {:?}", idx, logit, token_text);
            }
        }
        
        // Sample next token
        let next_token = sample(&logits, config)?;
        
        // Check for EOS
        if next_token == tokenizer.eos_token {
            println!("\n  Reached EOS token");
            break;
        }
        
        generated_tokens.push(next_token);
        
        // Print progress
        if i % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!(" ✓");
    
    // 5. Decode generated tokens
    println!("  Decoding...");
    let generated_text = tokenizer.decode(&generated_tokens);
    
    Ok(generated_text)
}

/// Sample next token from logits
fn sample(logits: &[f32], config: &GenerationConfig) -> Result<u32> {
    let n_vocab = logits.len();
    
    // Apply temperature
    let mut probs = logits.to_vec();
    if config.temperature > 0.0 {
        for p in &mut probs {
            *p /= config.temperature;
        }
    }
    
    // Softmax
    let max_logit = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for p in &mut probs {
        *p = (*p - max_logit).exp();
        sum += *p;
    }
    for p in &mut probs {
        *p /= sum;
    }
    
    // Top-k filtering
    if config.top_k > 0 && config.top_k < n_vocab {
        let mut indices: Vec<usize> = (0..n_vocab).collect();
        indices.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        for &idx in &indices[config.top_k..] {
            probs[idx] = 0.0;
        }
        
        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
    }
    
    // Top-p (nucleus) sampling
    if config.top_p < 1.0 {
        let mut indices: Vec<usize> = (0..n_vocab).collect();
        indices.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut cumsum = 0.0;
        let mut cutoff = n_vocab;
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum > config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        
        for &idx in &indices[cutoff..] {
            probs[idx] = 0.0;
        }
        
        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
    }
    
    // Sample from distribution
    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u32);
        }
    }
    
    // Fallback: return highest probability token
    let max_idx = probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    Ok(max_idx as u32)
}

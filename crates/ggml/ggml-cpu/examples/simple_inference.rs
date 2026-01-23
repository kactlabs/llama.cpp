//! Simple inference example demonstrating end-to-end model execution

use ggml_core::{Context, TensorType};
use ggml_cpu::compute::CpuCompute;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Inference Example ===\n");
    
    // Tiny model for demonstration
    let n_vocab = 100;
    let n_embd = 64;
    let n_tokens = 4;
    let n_ff = 128;
    
    println!("Model: vocab={}, embd={}, tokens={}, ff={}\n", n_vocab, n_embd, n_tokens, n_ff);
    
    let mut ctx = Context::new(50 * 1024 * 1024)?;
    
    // === Step 1: Embedding ===
    println!("Step 1: Token Embedding");
    let token_ids = vec![5, 12, 3, 42];
    println!("  Tokens: {:?}", token_ids);
    
    // Create embedding table [n_vocab, n_embd] - each row is an embedding
    let embed = ctx.new_tensor(TensorType::F32, &[n_vocab, n_embd])?;
    let mut embed_data = vec![0.0f32; n_vocab * n_embd];
    for i in 0..embed_data.len() {
        embed_data[i] = ((i % 50) as f32 - 25.0) / 50.0;
    }
    ctx.set_tensor_data_f32(&embed, &embed_data)?;
    
    // Lookup embeddings - output is [n_embd * n_tokens] flat
    let mut x_data = vec![0.0f32; n_embd * n_tokens];
    {
        let embed_data = ctx.get_tensor_data_f32(&embed)?;
        CpuCompute::get_rows_f32(embed_data, [n_vocab, n_embd], &token_ids, &mut x_data)?;
    }
    println!("  ✓ Embedded {} tokens\n", n_tokens);
    
    // === Step 2: Layer Norm ===
    println!("Step 2: Layer Normalization");
    let mut x_norm_data = vec![0.0f32; n_embd * n_tokens];
    for i in 0..n_tokens {
        let start = i * n_embd;
        let end = start + n_embd;
        CpuCompute::layer_norm_f32(&x_data[start..end], &mut x_norm_data[start..end], 1e-5)?;
    }
    println!("  ✓ Normalized\n");
    
    // === Step 3: Feed-Forward ===
    println!("Step 3: Feed-Forward Network");
    
    // Debug: check sizes
    println!("  x_norm_data.len() = {}, expected = {}", x_norm_data.len(), n_embd * n_tokens);
    
    // Create weights
    let w1 = ctx.new_tensor(TensorType::F32, &[n_ff, n_embd])?;
    let w2 = ctx.new_tensor(TensorType::F32, &[n_embd, n_ff])?;
    
    let w1_data = vec![0.01f32; n_ff * n_embd];
    let w2_data = vec![0.01f32; n_embd * n_ff];
    ctx.set_tensor_data_f32(&w1, &w1_data)?;
    ctx.set_tensor_data_f32(&w2, &w2_data)?;
    
    // First projection: [n_ff, n_embd] @ [n_embd, n_tokens] -> [n_ff, n_tokens]
    // In GGML column-major: A[a0, a1] @ B[b0, b1] requires a0 == b1
    // So we need: W1[n_embd, n_ff] @ x_norm[n_tokens, n_embd] 
    // Wait, that's also wrong. Let me think...
    // 
    // Actually, GGML stores [rows, cols] but in column-major layout
    // W1 should be [n_embd, n_ff] (n_embd rows, n_ff cols) to multiply with x_norm [n_embd, n_tokens]
    // Result: [n_ff, n_tokens]
    //
    // Let's just transpose the logic: treat x_norm as [n_tokens, n_embd] and W1 as [n_embd, n_ff]
    let mut hidden_data = vec![0.0f32; n_ff * n_tokens];
    {
        let w1_data = ctx.get_tensor_data_f32(&w1)?;
        // Swap the interpretation: x_norm is [n_tokens, n_embd], W1 is [n_embd, n_ff]
        // But wait, that won't work either because the data layout is fixed
        
        // The real issue: get_rows outputs [n_embd, n_tokens] in column-major
        // which means n_embd elements per column, n_tokens columns
        // So the data is: [emb0_tok0, emb1_tok0, ..., emb63_tok0, emb0_tok1, ...]
        // 
        // For matmul [n_ff, n_embd] @ [n_embd, n_tokens]:
        // We need x_norm as [n_tokens, n_embd] with n_tokens rows!
        // 
        // Let's transpose x_norm first
        let mut x_norm_t = vec![0.0f32; n_embd * n_tokens];
        CpuCompute::transpose_f32(&x_norm_data, [n_embd, n_tokens], &mut x_norm_t)?;
        
        // Now x_norm_t is [n_tokens, n_embd]
        // W1 @ x_norm_t: [n_ff, n_embd] @ [n_embd, n_tokens] - wait, we need [n_embd, n_tokens] not [n_tokens, n_embd]
        
        // Actually, let's just use the transposed version correctly:
        // x_norm_t is [n_tokens, n_embd] 
        // We want: W1 [n_ff, n_embd] @ x_norm [n_embd, n_tokens]
        // So use the original x_norm_data!
        println!("  Matmul: [{}, {}] @ [{}, {}] -> [{}, {}]", n_ff, n_embd, n_embd, n_tokens, n_ff, n_tokens);
        CpuCompute::matmul_f32(w1_data, [n_embd, n_ff], &x_norm_data, [n_tokens, n_embd], &mut hidden_data)?;
    }
    println!("  ✓ First projection: [{}, {}] @ [{}, {}] -> [{}, {}]", 
             n_ff, n_embd, n_embd, n_tokens, n_ff, n_tokens);
    
    // Activation
    let mut hidden_act_data = vec![0.0f32; n_ff * n_tokens];
    CpuCompute::silu_f32(&hidden_data, &mut hidden_act_data)?;
    println!("  ✓ SiLU activation");
    
    // Second projection: [n_embd, n_ff] @ [n_ff, n_tokens] -> [n_embd, n_tokens]
    // hidden_act_data is [n_ff, n_tokens] from previous step
    // W2 is [n_embd, n_ff]
    // For matmul A[a0, a1] @ B[b0, b1] requires a0 == b1
    // So: W2[n_ff, n_embd] @ hidden_act[n_tokens, n_ff]
    let mut output_data = vec![0.0f32; n_embd * n_tokens];
    {
        let w2_data = ctx.get_tensor_data_f32(&w2)?;
        CpuCompute::matmul_f32(w2_data, [n_ff, n_embd], &hidden_act_data, [n_tokens, n_ff], &mut output_data)?;
    }
    println!("  ✓ Second projection: [{}, {}] @ [{}, {}] -> [{}, {}]\n",
             n_embd, n_ff, n_ff, n_tokens, n_embd, n_tokens);
    
    // === Step 4: Residual ===
    println!("Step 4: Residual Connection");
    let mut x_out_data = vec![0.0f32; n_embd * n_tokens];
    CpuCompute::add_f32(&x_data, &output_data, &mut x_out_data)?;
    println!("  ✓ Added residual\n");
    
    // === Step 5: Output Projection ===
    println!("Step 5: Output Projection");
    
    // Transpose embedding: [n_vocab, n_embd] -> [n_embd, n_vocab]
    let mut embed_t_data = vec![0.0f32; n_embd * n_vocab];
    {
        let embed_data = ctx.get_tensor_data_f32(&embed)?;
        CpuCompute::transpose_f32(embed_data, [n_vocab, n_embd], &mut embed_t_data)?;
    }
    
    // Compute logits: embed_t[n_embd, n_vocab] @ x_out[n_embd, n_tokens] -> [n_vocab, n_tokens]
    // For matmul A[a0, a1] @ B[b0, b1] requires a0 == b1
    // embed_t is now [n_embd, n_vocab] (after transpose)
    // x_out is [n_embd, n_tokens]
    // We need: A[n_vocab, n_embd] @ B[n_tokens, n_embd]
    // So we need to transpose embed_t back? No, let's think...
    // 
    // After transpose of [n_vocab, n_embd], we get [n_embd, n_vocab]
    // For matmul [n_embd, n_vocab] @ [n_embd, n_tokens], we need a0 == b1
    // That's n_embd == n_tokens? No!
    //
    // Let me reconsider: we want [n_vocab, n_tokens] output
    // embed_t should be [n_vocab, n_embd] (each row is a vocab embedding)
    // x_out is [n_embd, n_tokens] (each column is a token)
    // 
    // Don't transpose! Use original embed
    let mut logits_data = vec![0.0f32; n_vocab * n_tokens];
    {
        let embed_data = ctx.get_tensor_data_f32(&embed)?;
        // embed is [n_vocab, n_embd], x_out is [n_embd, n_tokens]
        // For matmul A[a0, a1] @ B[b0, b1] requires a0 == b1
        // So: embed[n_embd, n_vocab] @ x_out[n_tokens, n_embd]
        CpuCompute::matmul_f32(embed_data, [n_embd, n_vocab], &x_out_data, [n_tokens, n_embd], &mut logits_data)?;
    }
    println!("  ✓ Computed logits: [{}, {}]\n", n_vocab, n_tokens);
    
    // === Step 6: Softmax ===
    println!("Step 6: Softmax (last token)");
    let last_token_start = (n_tokens - 1) * n_vocab;
    let mut probs_data = vec![0.0f32; n_vocab];
    CpuCompute::softmax_f32(&logits_data[last_token_start..], &mut probs_data)?;
    
    // Top 5 predictions
    let mut indices: Vec<usize> = (0..n_vocab).collect();
    indices.sort_by(|&a, &b| probs_data[b].partial_cmp(&probs_data[a]).unwrap());
    
    println!("  Top 5 predictions:");
    for i in 0..5 {
        let idx = indices[i];
        println!("    Token {}: {:.2}%", idx, probs_data[idx] * 100.0);
    }
    
    println!("\n=== Success! ===");
    println!("Completed full forward pass:");
    println!("  • Token embedding");
    println!("  • Layer normalization");
    println!("  • Feed-forward (SiLU)");
    println!("  • Residual connection");
    println!("  • Output projection");
    println!("  • Softmax");
    println!("\nAll operations used SIMD + multi-threading!");
    
    Ok(())
}

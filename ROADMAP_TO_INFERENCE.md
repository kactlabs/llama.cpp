# Roadmap: lrama-cli Full Inference Implementation

## Current Status ✅

**Infrastructure: 100% Complete**
- ✅ 31 tensor operations (all working)
- ✅ SIMD optimizations (4x speedup)
- ✅ Multi-threading (7-10x speedup)
- ✅ GGUF model loading
- ✅ 20-25 GFLOPS performance
- ✅ 38 tests passing

**Demo Working:**
```bash
$ cargo run --package lrama-cli --release -- \
    --model ~/Library/Caches/llama.cpp/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
    --prompt "Once upon a time"

⚠️  Inference not yet implemented!
   Next steps:
   1. Tokenize prompt
   2. Load model weights into tensors
   3. Run forward pass
   4. Sample next token
   5. Repeat for n_predict tokens
```

## Implementation Plan

### Step 1: Tokenizer (Priority: CRITICAL)
**Estimated: 300-400 lines, 4-6 hours**

Load vocabulary and implement BPE tokenization.

**Files to create:**
- `crates/lrama-cli/src/tokenizer.rs`

**What to implement:**
```rust
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    token_to_piece: Vec<String>,
    bos_token: u32,
    eos_token: u32,
}

impl Tokenizer {
    // Load from GGUF metadata
    pub fn from_gguf(reader: &GGUFReader) -> Result<Self>;
    
    // Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32>;
    
    // Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String;
}
```

**GGUF metadata keys:**
- `tokenizer.ggml.tokens` - array of token strings
- `tokenizer.ggml.token_type` - token types
- `tokenizer.ggml.bos_token_id` - beginning of sequence
- `tokenizer.ggml.eos_token_id` - end of sequence

**Complexity:** Medium (BPE algorithm is tricky)

---

### Step 2: Quantization Dequantization (Priority: CRITICAL)
**Estimated: 500-700 lines, 8-12 hours**

Implement dequantization for Q2_K, Q3_K, Q4_K formats.

**Files to create:**
- `crates/ggml/ggml-quants/src/dequant.rs`

**What to implement:**
```rust
pub trait Dequantize {
    fn dequantize(&self, output: &mut [f32]) -> Result<()>;
}

// Q2_K: 2-bit quantization with K-means
pub struct Q2KBlock {
    scales: [u8; 16],
    qs: [u8; 64],
    d: f16,
    dmin: f16,
}

// Q3_K: 3-bit quantization
pub struct Q3KBlock {
    hmask: [u8; 32],
    qs: [u8; 96],
    scales: [u8; 12],
    d: f16,
}

// Similar for Q4_K, Q5_K, Q6_K
```

**Block formats (from GGML):**
- Q2_K: 82 bytes per 256 elements
- Q3_K: 110 bytes per 256 elements
- Q4_K: 144 bytes per 256 elements

**Complexity:** High (bit manipulation, careful indexing)

---

### Step 3: Model Architecture (Priority: HIGH)
**Estimated: 400-500 lines, 6-8 hours**

Implement LLaMA model structure and weight loading.

**Files to create:**
- `crates/lrama-cli/src/model.rs`

**What to implement:**
```rust
pub struct LlamaModel {
    // Hyperparameters
    n_vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    n_ff: usize,
    
    // Weights (dequantized)
    token_embd: Tensor,
    layers: Vec<LlamaLayer>,
    output_norm: Tensor,
}

pub struct LlamaLayer {
    attn_norm: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_output: Tensor,
    
    ffn_norm: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
}

impl LlamaModel {
    // Load from GGUF
    pub fn from_gguf(reader: &GGUFReader, ctx: &mut Context) -> Result<Self>;
    
    // Forward pass for one token
    pub fn forward(&self, token: u32, pos: usize, kv_cache: &mut KVCache) -> Result<Vec<f32>>;
}
```

**Complexity:** Medium-High (many tensors to manage)

---

### Step 4: Attention + KV Cache (Priority: HIGH)
**Estimated: 300-400 lines, 6-8 hours**

Implement multi-head attention with KV caching.

**Files to create:**
- `crates/lrama-cli/src/attention.rs`

**What to implement:**
```rust
pub struct KVCache {
    k_cache: Vec<Tensor>,  // [n_layer][n_head, n_ctx, head_dim]
    v_cache: Vec<Tensor>,  // [n_layer][n_head, n_ctx, head_dim]
    n_past: usize,
}

pub fn attention(
    q: &[f32],           // Query [n_head, head_dim]
    k: &[f32],           // Key [n_head, head_dim]
    v: &[f32],           // Value [n_head, head_dim]
    kv_cache: &mut KVCache,
    layer_idx: usize,
    pos: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>>;
```

**Steps:**
1. Apply RoPE to Q and K
2. Store K, V in cache
3. Compute attention scores: Q @ K^T / sqrt(d)
4. Apply causal mask
5. Softmax
6. Weighted sum with V
7. Concatenate heads

**Complexity:** High (many moving parts, cache management)

---

### Step 5: Inference Loop (Priority: MEDIUM)
**Estimated: 200-300 lines, 4-6 hours**

Implement autoregressive generation with sampling.

**Files to create:**
- `crates/lrama-cli/src/generate.rs`

**What to implement:**
```rust
pub struct GenerationConfig {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
}

pub fn generate(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    config: &GenerationConfig,
) -> Result<String> {
    // 1. Tokenize prompt
    let tokens = tokenizer.encode(prompt);
    
    // 2. Initialize KV cache
    let mut kv_cache = KVCache::new(model.n_layer, model.n_head, 2048);
    
    // 3. Process prompt tokens
    for (pos, &token) in tokens.iter().enumerate() {
        model.forward(token, pos, &mut kv_cache)?;
    }
    
    // 4. Generate tokens
    let mut generated = Vec::new();
    for pos in tokens.len()..tokens.len() + n_predict {
        // Forward pass
        let logits = model.forward(last_token, pos, &mut kv_cache)?;
        
        // Sample next token
        let next_token = sample(&logits, config);
        
        if next_token == tokenizer.eos_token {
            break;
        }
        
        generated.push(next_token);
    }
    
    // 5. Decode to text
    Ok(tokenizer.decode(&generated))
}

fn sample(logits: &[f32], config: &GenerationConfig) -> u32 {
    // Apply temperature
    // Apply top-k filtering
    // Apply top-p (nucleus) sampling
    // Sample from distribution
}
```

**Complexity:** Medium (sampling algorithms)

---

## Total Effort Estimate

| Component | Lines | Hours | Priority |
|-----------|-------|-------|----------|
| Tokenizer | 300-400 | 4-6 | CRITICAL |
| Quantization | 500-700 | 8-12 | CRITICAL |
| Model Architecture | 400-500 | 6-8 | HIGH |
| Attention + KV Cache | 300-400 | 6-8 | HIGH |
| Inference Loop | 200-300 | 4-6 | MEDIUM |
| **TOTAL** | **1,700-2,300** | **28-40** | - |

## Recommended Order

### Phase 1: Minimal Working Inference (16-24 hours)
1. **Tokenizer** (4-6h) - Can test with simple encode/decode
2. **Quantization** (8-12h) - Start with Q4_K (simplest), test dequantization
3. **Model Loading** (4-6h) - Load weights, verify shapes

### Phase 2: Forward Pass (12-16 hours)
4. **Attention** (6-8h) - Implement without cache first
5. **Model Forward** (6-8h) - Single token forward pass

### Phase 3: Generation (4-6 hours)
6. **Inference Loop** (4-6h) - Autoregressive generation with sampling

## Testing Strategy

After each phase:
1. **Unit tests** - Test individual components
2. **Integration tests** - Test component interactions
3. **Comparison tests** - Compare outputs with C++ llama.cpp

## Expected Performance

With current optimizations (SIMD + multi-threading):
- **Prompt processing**: 10-50 tokens/sec
- **Generation**: 5-20 tokens/sec
- **Memory**: ~2-4 GB for TinyLLaMA-1.1B

Compare to C++ llama.cpp with Metal:
- **Prompt**: 1104 tokens/sec (22-110x faster)
- **Generation**: 276 tokens/sec (14-55x faster)

CPU-only is expected to be much slower than GPU, but still usable for small models.

## Quick Start Guide (Once Complete)

```bash
# Generate text
cargo run --package lrama-cli --release -- \
    --model path/to/model.gguf \
    --prompt "Once upon a time" \
    --n-predict 128 \
    --temperature 0.8

# Interactive chat
cargo run --package lrama-cli --release -- \
    --model path/to/model.gguf \
    --interactive

# Benchmark
cargo run --package lrama-cli --release -- \
    --model path/to/model.gguf \
    --benchmark
```

## Resources

**Reference implementations:**
- llama.cpp: https://github.com/ggerganov/llama.cpp
- GGML quantization: `ggml-quants.c`
- LLaMA architecture: `llama.cpp`

**Documentation:**
- GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- LLaMA paper: https://arxiv.org/abs/2302.13971
- RoPE: https://arxiv.org/abs/2104.09864

## Current Status Summary

✅ **Complete (100%)**
- Tensor operations
- SIMD optimizations
- Multi-threading
- GGUF loading
- Testing infrastructure

⏳ **In Progress (0%)**
- Tokenizer
- Quantization
- Model architecture
- Attention mechanism
- Inference loop

🎯 **Goal**: Full text generation with TinyLLaMA-1.1B model

**Estimated time to completion: 28-40 hours of focused development**

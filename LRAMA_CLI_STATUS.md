# lrama-cli: Rust LLaMA Inference CLI

## Status: Initial Implementation Complete ✅

### What Works
- ✅ GGUF model loading
- ✅ Model metadata parsing
- ✅ Tensor information display
- ✅ CLI argument parsing

### Test Results
```bash
$ cargo run --package lrama-cli --release -- \
    --model ~/Library/Caches/llama.cpp/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
    --info

🦙 lrama-cli - Rust LLaMA Inference
=====================================

Loading model: .../tinyllama-1.1b-chat-v1.0.Q2_K.gguf
📊 Model Information:
   Version: V3
   Tensors: 201
   Name: String("tinyllama_tinyllama-1.1b-chat-v1.0")
   Architecture: String("llama")

📦 Sample tensors (201 total):
   1. blk.16.attn_k.weight - Q2_K [2048, 256]
   2. blk.3.ffn_down.weight - Q3_K [5632, 2048]
   ...
```

### What's Next: Implement Inference

To make `lrama-cli` actually run inference, we need:

1. **Tokenizer** (CRITICAL)
   - Load vocabulary from GGUF metadata
   - Implement BPE tokenization
   - Encode prompt to token IDs
   - Decode token IDs back to text

2. **Model Architecture** (CRITICAL)
   - Load all tensor weights from GGUF
   - Implement LLaMA architecture:
     - Token embedding
     - RoPE (Rotary Position Embedding)
     - Multi-head attention with KV cache
     - Feed-forward network (SwiGLU)
     - RMS normalization
     - Output projection

3. **Inference Loop**
   - Forward pass through all layers
   - Sampling (temperature, top-p, top-k)
   - Autoregressive generation
   - KV cache management

4. **Quantization Support**
   - Dequantize Q2_K, Q3_K, Q4_K, etc.
   - Currently model uses Q2_K and Q3_K quantization

### Current Infrastructure

**Available Operations** (from ggml-cpu):
- ✅ Matrix multiplication (parallel, SIMD-optimized)
- ✅ RoPE (Rotary Position Embedding)
- ✅ RMS normalization
- ✅ SiLU activation
- ✅ Softmax
- ✅ Element-wise ops (add, mul, etc.)
- ✅ get_rows (embedding lookup)

**Performance**:
- 20-25 GFLOPS on 14-core Apple Silicon
- SIMD optimizations (NEON)
- Multi-threading with rayon

**Missing**:
- ❌ Quantization dequantization (Q2_K, Q3_K, Q4_K, etc.)
- ❌ Tokenizer
- ❌ KV cache
- ❌ Attention implementation
- ❌ Full LLaMA model architecture

### Comparison with C++ llama-cli

**C++ llama-cli** (existing):
- Uses Metal GPU backend on Apple Silicon
- Performance: 1104 t/s (prompt), 276 t/s (generation)
- Full feature set

**lrama-cli** (Rust, our implementation):
- Currently: Model loading only
- Target: CPU-based inference with SIMD + multi-threading
- Expected performance: 10-50 t/s (much slower than GPU, but pure Rust)

### Next Steps

**Phase 1: Basic Inference** (Recommended)
1. Implement tokenizer (load vocab from GGUF)
2. Implement attention mechanism
3. Implement KV cache
4. Build LLaMA forward pass
5. Implement basic sampling
6. Test with TinyLLaMA model

**Phase 2: Optimization**
1. Add quantization support (Q2_K, Q3_K, Q4_K)
2. Optimize attention computation
3. Improve KV cache efficiency
4. Add more sampling strategies

**Phase 3: Features**
1. Interactive chat mode
2. Streaming output
3. Context management
4. Model presets

### Files Created
- `crates/lrama-cli/Cargo.toml` - Package definition
- `crates/lrama-cli/src/main.rs` - CLI implementation
- `LRAMA_CLI_STATUS.md` - This status document

### How to Use (Current)
```bash
# Show model info
cargo run --package lrama-cli --release -- \
    --model path/to/model.gguf \
    --info

# Run inference (not yet implemented)
cargo run --package lrama-cli --release -- \
    --model path/to/model.gguf \
    --prompt "Once upon a time" \
    --n-predict 128
```

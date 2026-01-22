# llama.cpp → Rust Conversion Plan

## Project Overview
Complete conversion of llama.cpp (C/C++) to Rust, maintaining feature parity with all backends, tools, and model architectures.

**Estimated Timeline**: 6-12 months (full-time equivalent)
**Complexity**: Very High
**Lines of Code**: ~200,000+ LOC

---

## Phase 1: Foundation & Core Infrastructure (Weeks 1-4)

### 1.1 Project Setup
- [x] Create Cargo workspace structure
- [ ] Set up CI/CD pipeline
- [ ] Configure cross-compilation
- [ ] Set up benchmarking infrastructure
- [ ] Create documentation framework

### 1.2 GGUF Format Support
**Priority**: Critical
**Complexity**: Medium
**Files**: `gguf-py/`, model loading code

**Tasks**:
- [ ] Implement GGUF file format parser
- [ ] Create metadata extraction
- [ ] Implement tensor loading
- [ ] Add validation and checksums
- [ ] Support memory-mapped files

**Rust Crates**:
- `memmap2` for memory mapping
- `byteorder` for endianness
- `serde` for metadata

### 1.3 Core Data Structures
**Priority**: Critical
**Complexity**: High

**Tasks**:
- [ ] `ggml_tensor` → Rust tensor type
- [ ] `ggml_context` → Memory arena
- [ ] `ggml_cgraph` → Computation graph
- [ ] Type system for quantization formats
- [ ] Error handling framework

---

## Phase 2: GGML Core (Weeks 5-12)

### 2.1 Tensor Operations
**Priority**: Critical
**Complexity**: Very High
**Files**: `ggml/src/ggml.c`, `ggml/include/ggml.h`

**Operations to Implement** (~200 ops):
- [ ] Basic: add, sub, mul, div
- [ ] Matrix: mul_mat, mul_mat_id
- [ ] Activation: relu, gelu, silu, swish
- [ ] Normalization: rms_norm, layer_norm, group_norm
- [ ] Attention: rope, flash_attn, alibi
- [ ] Pooling: pool_1d, pool_2d
- [ ] Convolution: conv_1d, conv_2d, conv_transpose
- [ ] Specialized: ssm_conv, ssm_scan, rwkv_wkv
- [ ] Quantization: quantize, dequantize

**Rust Approach**:
- Trait-based operation system
- Generic over data types
- SIMD via `std::simd` or `packed_simd`

### 2.2 Computation Graph
**Priority**: Critical
**Complexity**: High

**Tasks**:
- [ ] Graph construction API
- [ ] Topological sorting
- [ ] Gradient computation (autodiff)
- [ ] Graph optimization passes
- [ ] Memory planning

### 2.3 Memory Management
**Priority**: Critical
**Complexity**: High
**Files**: `ggml/src/ggml-alloc.c`

**Tasks**:
- [ ] Custom allocator implementation
- [ ] Buffer management
- [ ] Memory pooling
- [ ] Alignment handling
- [ ] Memory-mapped file support

---

## Phase 3: CPU Backend (Weeks 13-18)

### 3.1 CPU Operations
**Priority**: Critical
**Complexity**: Very High
**Files**: `ggml/src/ggml-cpu/`

**Tasks**:
- [ ] Scalar implementations (reference)
- [ ] SIMD implementations:
  - [ ] x86_64: AVX, AVX2, AVX512, AMX
  - [ ] ARM: NEON
  - [ ] RISC-V: RVV
- [ ] Quantization kernels (Q2-Q8, IQ variants)
- [ ] Threading and parallelization
- [ ] Cache optimization

**Rust Crates**:
- `std::simd` (nightly) or `packed_simd`
- `rayon` for parallelization
- `num_cpus` for thread detection

### 3.2 Quantization
**Priority**: Critical
**Complexity**: Very High
**Files**: `ggml/src/ggml-quants.c`

**Formats to Implement**:
- [ ] F16, BF16
- [ ] Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- [ ] Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- [ ] IQ1_S, IQ1_M
- [ ] IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M
- [ ] IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M
- [ ] IQ4_NL, IQ4_XS

---

## Phase 4: Backend Abstraction (Weeks 19-22)

### 4.1 Backend Interface
**Priority**: Critical
**Complexity**: High
**Files**: `ggml/src/ggml-backend.cpp`

**Tasks**:
- [ ] Backend trait definition
- [ ] Buffer type abstraction
- [ ] Device enumeration
- [ ] Backend registration system
- [ ] Multi-backend scheduler
- [ ] Async operation support

**Rust Approach**:
```rust
trait Backend {
    fn name(&self) -> &str;
    fn alloc_buffer(&self, size: usize) -> Result<Buffer>;
    fn compute_graph(&self, graph: &ComputeGraph) -> Result<()>;
    // ...
}
```

### 4.2 Scheduler
**Priority**: High
**Complexity**: High

**Tasks**:
- [ ] Graph splitting across backends
- [ ] Tensor placement optimization
- [ ] Data transfer minimization
- [ ] Parallel execution
- [ ] Event synchronization

---

## Phase 5: GPU Backends (Weeks 23-36)

### 5.1 CUDA Backend
**Priority**: High
**Complexity**: Very High
**Files**: `ggml/src/ggml-cuda/` (100+ files)

**Tasks**:
- [ ] CUDA kernel compilation
- [ ] Matrix multiplication kernels
- [ ] Attention kernels (flash attention)
- [ ] Quantization kernels
- [ ] Memory management
- [ ] Stream management
- [ ] cuBLAS integration

**Rust Crates**:
- `cuda-sys` or `cudarc`
- `bindgen` for CUDA headers
- Consider keeping CUDA kernels in .cu files

### 5.2 Metal Backend
**Priority**: High (for Apple Silicon)
**Complexity**: High
**Files**: `ggml/src/ggml-metal/`

**Tasks**:
- [ ] Metal shader compilation
- [ ] Compute pipeline setup
- [ ] Buffer management
- [ ] Command encoding
- [ ] MPS integration

**Rust Crates**:
- `metal-rs`
- `objc` for Objective-C interop

### 5.3 Vulkan Backend
**Priority**: Medium
**Complexity**: Very High
**Files**: `ggml/src/ggml-vulkan/`

**Tasks**:
- [ ] Vulkan device setup
- [ ] Shader compilation (SPIR-V)
- [ ] Descriptor sets
- [ ] Pipeline management
- [ ] Memory allocation

**Rust Crates**:
- `vulkano` or `ash`
- `shaderc` for shader compilation

### 5.4 Other Backends
**Priority**: Low-Medium
**Complexity**: High

- [ ] SYCL (Intel oneAPI)
- [ ] HIP (AMD)
- [ ] MUSA (Moore Threads)
- [ ] OpenCL
- [ ] WebGPU
- [ ] RPC

---

## Phase 6: llama.cpp Layer (Weeks 37-44)

### 6.1 Model Loading
**Priority**: Critical
**Complexity**: Medium
**Files**: `src/llama-model-loader.cpp`

**Tasks**:
- [ ] GGUF model loading
- [ ] Hyperparameter extraction
- [ ] Tensor mapping
- [ ] Vocabulary loading
- [ ] Metadata handling

### 6.2 Tokenization
**Priority**: Critical
**Complexity**: Medium
**Files**: `src/llama-vocab.cpp`, `src/unicode.cpp`

**Tasks**:
- [ ] SPM (SentencePiece)
- [ ] BPE (Byte-Pair Encoding)
- [ ] WPM (WordPiece)
- [ ] UGM (Unigram)
- [ ] RWKV tokenizer
- [ ] Unicode normalization

**Rust Crates**:
- `tokenizers` (Hugging Face)
- `unicode-normalization`
- `unicode-segmentation`

### 6.3 Context Management
**Priority**: Critical
**Complexity**: High
**Files**: `src/llama-context.cpp`

**Tasks**:
- [ ] Context initialization
- [ ] KV cache management
- [ ] Batch processing
- [ ] Sequence management
- [ ] Memory tracking

### 6.4 KV Cache
**Priority**: Critical
**Complexity**: High
**Files**: `src/llama-kv-cache.cpp`

**Tasks**:
- [ ] Cache allocation
- [ ] Sequence tracking
- [ ] Cache shifting
- [ ] Multi-sequence support
- [ ] ISWA (Infinite Sliding Window Attention)

### 6.5 Sampling
**Priority**: High
**Complexity**: Medium
**Files**: `src/llama-sampling.cpp`

**Tasks**:
- [ ] Temperature sampling
- [ ] Top-k sampling
- [ ] Top-p (nucleus) sampling
- [ ] Min-p sampling
- [ ] Typical sampling
- [ ] Mirostat v1/v2
- [ ] Repetition penalties
- [ ] Frequency/presence penalties

### 6.6 Grammar Support
**Priority**: Medium
**Complexity**: High
**Files**: `src/llama-grammar.cpp`

**Tasks**:
- [ ] GBNF parser
- [ ] Grammar-constrained generation
- [ ] JSON schema support
- [ ] State machine implementation

---

## Phase 7: Model Architectures (Weeks 45-56)

### 7.1 Core Architectures
**Priority**: Critical
**Complexity**: Very High
**Files**: `src/models/*.cpp` (100+ files)

**Tier 1 (Essential)**:
- [ ] LLaMA (llama.cpp)
- [ ] Mistral (mistral3.cpp)
- [ ] Qwen (qwen.cpp, qwen2.cpp, qwen3.cpp)
- [ ] Gemma (gemma.cpp, gemma2.cpp, gemma3.cpp)
- [ ] Phi (phi.cpp, phi2.cpp, phi3.cpp)

**Tier 2 (Popular)**:
- [ ] Falcon
- [ ] MPT
- [ ] Bloom
- [ ] StableLM
- [ ] GPT-2
- [ ] GPT-NeoX

**Tier 3 (Specialized)**:
- [ ] Mamba
- [ ] RWKV
- [ ] Jamba
- [ ] Grok
- [ ] Bitnet

**Tier 4 (Multimodal)**:
- [ ] CogVLM
- [ ] Qwen2VL
- [ ] Qwen3VL
- [ ] LLaVA

**Tier 5 (Encoder-Decoder)**:
- [ ] T5
- [ ] BERT
- [ ] Chameleon

### 7.2 Graph Building
**Priority**: Critical
**Complexity**: Very High
**Files**: `src/llama-graph.cpp`

**Tasks**:
- [ ] Architecture-specific graph builders
- [ ] Attention mechanisms
- [ ] Feed-forward networks
- [ ] Normalization layers
- [ ] Embedding layers
- [ ] Output layers

---

## Phase 8: Tools & Utilities (Weeks 57-64)

### 8.1 Core Tools
**Priority**: High
**Complexity**: Medium

**Tools to Implement**:
- [ ] `llama-cli` - CLI interface
- [ ] `llama-server` - HTTP API server
- [ ] `llama-bench` - Benchmarking
- [ ] `llama-quantize` - Model quantization
- [ ] `llama-perplexity` - Evaluation

### 8.2 Model Conversion
**Priority**: High
**Complexity**: Medium
**Files**: `convert_hf_to_gguf.py`

**Options**:
1. Keep Python scripts (easier)
2. Port to Rust (better integration)

**If porting to Rust**:
- [ ] Hugging Face model loading
- [ ] Tensor conversion
- [ ] Metadata extraction
- [ ] GGUF writing

**Rust Crates**:
- `hf-hub` for Hugging Face
- `safetensors` for tensor format
- `ndarray` for array operations

### 8.3 HTTP Server
**Priority**: High
**Complexity**: Medium
**Files**: `tools/server/`

**Tasks**:
- [ ] OpenAI-compatible API
- [ ] WebSocket support
- [ ] Static file serving (WebUI)
- [ ] SSE (Server-Sent Events)
- [ ] Multi-user support
- [ ] Request queuing

**Rust Crates**:
- `axum` or `actix-web`
- `tokio` for async runtime
- `tower` for middleware
- `serde_json` for JSON

---

## Phase 9: Testing & Validation (Weeks 65-72)

### 9.1 Unit Tests
**Priority**: Critical
**Complexity**: Medium

**Test Coverage**:
- [ ] Tensor operations
- [ ] Quantization accuracy
- [ ] Model loading
- [ ] Tokenization
- [ ] Sampling algorithms
- [ ] Backend operations

### 9.2 Integration Tests
**Priority**: Critical
**Complexity**: High

**Tests**:
- [ ] End-to-end inference
- [ ] Multi-backend execution
- [ ] Model conversion pipeline
- [ ] Server API compliance
- [ ] Performance benchmarks

### 9.3 Validation
**Priority**: Critical
**Complexity**: High

**Validation Tasks**:
- [ ] Output parity with C++ version
- [ ] Performance comparison
- [ ] Memory usage analysis
- [ ] Numerical accuracy
- [ ] Cross-platform testing

---

## Phase 10: Optimization & Polish (Weeks 73-80)

### 10.1 Performance Optimization
**Priority**: High
**Complexity**: High

**Tasks**:
- [ ] Profile hot paths
- [ ] Optimize memory allocations
- [ ] Improve cache locality
- [ ] SIMD optimization
- [ ] GPU kernel tuning
- [ ] Reduce overhead

### 10.2 Documentation
**Priority**: High
**Complexity**: Medium

**Documentation**:
- [ ] API documentation (rustdoc)
- [ ] Architecture guide
- [ ] Backend implementation guide
- [ ] Model architecture guide
- [ ] Examples and tutorials
- [ ] Migration guide from C++

### 10.3 Packaging
**Priority**: Medium
**Complexity**: Low

**Tasks**:
- [ ] Crates.io publication
- [ ] Binary releases
- [ ] Docker images
- [ ] Package managers (brew, apt, etc.)
- [ ] Python bindings (PyO3)

---

## Cargo Workspace Structure

```
llama-rs/
├── Cargo.toml (workspace)
├── crates/
│   ├── ggml/
│   │   ├── ggml-core/          # Core tensor library
│   │   ├── ggml-backend/       # Backend abstraction
│   │   ├── ggml-cpu/           # CPU backend
│   │   ├── ggml-cuda/          # CUDA backend
│   │   ├── ggml-metal/         # Metal backend
│   │   ├── ggml-vulkan/        # Vulkan backend
│   │   └── ggml-quants/        # Quantization
│   ├── llama/
│   │   ├── llama-core/         # Core library
│   │   ├── llama-model/        # Model loading
│   │   ├── llama-vocab/        # Tokenization
│   │   ├── llama-sampling/     # Sampling
│   │   └── llama-architectures/# Model architectures
│   ├── gguf/                   # GGUF format
│   └── common/                 # Shared utilities
├── tools/
│   ├── llama-cli/
│   ├── llama-server/
│   ├── llama-bench/
│   ├── llama-quantize/
│   └── llama-convert/
├── examples/
│   ├── simple/
│   ├── chat/
│   └── embedding/
└── tests/
    ├── unit/
    ├── integration/
    └── validation/
```

---

## Key Technical Decisions

### 1. Memory Management
- Use Rust's ownership system
- Custom allocators for performance-critical paths
- Arena allocation for computation graphs
- Memory-mapped files for model loading

### 2. SIMD
- Use `std::simd` (nightly) or `packed_simd`
- Fallback to scalar implementations
- Runtime CPU feature detection
- Architecture-specific optimizations

### 3. GPU Kernels
- Keep CUDA kernels in .cu files (compile separately)
- Use Rust bindings for kernel launch
- Metal shaders in .metal files
- Vulkan SPIR-V compilation

### 4. Error Handling
- Use `Result<T, E>` throughout
- Custom error types per module
- `thiserror` for error definitions
- `anyhow` for application errors

### 5. Async/Await
- Use `tokio` for async runtime
- Async API for server
- Sync API for core inference
- Bridge between sync and async

### 6. FFI Compatibility
- Provide C API for compatibility
- Use `cbindgen` for header generation
- Maintain ABI stability
- Support existing bindings

---

## Risk Mitigation

### High-Risk Areas
1. **GPU Kernels**: Complex, performance-critical
   - **Mitigation**: Keep existing kernels, use FFI
   
2. **Quantization**: Numerical accuracy critical
   - **Mitigation**: Extensive validation tests
   
3. **Performance**: Must match C++ performance
   - **Mitigation**: Continuous benchmarking
   
4. **Backend Compatibility**: Many backends to support
   - **Mitigation**: Phased rollout, prioritize popular backends

### Dependencies
- Minimize external dependencies
- Vendor critical dependencies
- Use well-maintained crates
- Have fallback implementations

---

## Success Criteria

### Functional
- [ ] All model architectures supported
- [ ] All quantization formats working
- [ ] All backends functional
- [ ] All tools implemented
- [ ] API compatibility maintained

### Performance
- [ ] Within 5% of C++ performance (CPU)
- [ ] Within 10% of C++ performance (GPU)
- [ ] Memory usage comparable
- [ ] Startup time acceptable

### Quality
- [ ] >80% test coverage
- [ ] Zero unsafe code in public API
- [ ] Comprehensive documentation
- [ ] Clean, idiomatic Rust code

---

## Resources Required

### Team
- 2-3 senior Rust developers
- 1 GPU programming expert
- 1 ML engineer
- 1 DevOps engineer

### Infrastructure
- CI/CD with GPU runners
- Multiple GPU types for testing
- Cross-platform build machines
- Benchmarking infrastructure

### Timeline
- **Minimum**: 6 months (aggressive, focused team)
- **Realistic**: 9-12 months (full-featured)
- **Conservative**: 12-18 months (production-ready)

---

## Next Steps

1. Set up Cargo workspace
2. Implement GGUF parser
3. Create core tensor types
4. Implement basic CPU operations
5. Build simple inference example
6. Iterate and expand

---

## Notes

This is a **massive undertaking**. Consider:
- Starting with a subset (CPU-only, limited models)
- Using FFI to existing C++ code initially
- Gradual migration approach
- Contributing to existing Rust projects instead

**Existing Rust alternatives**:
- `candle` (Hugging Face)
- `mistral.rs`
- `llm` crate
- Various llama.cpp Rust bindings

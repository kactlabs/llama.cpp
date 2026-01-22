# llama-rs: Rust Implementation of llama.cpp

This is a complete Rust rewrite of [llama.cpp](https://github.com/ggml-org/llama.cpp), a high-performance LLM inference engine.

## Project Status

🚧 **WORK IN PROGRESS** 🚧

This is an ambitious, multi-month project to convert the entire llama.cpp codebase to Rust. See [RUST_CONVERSION_PLAN.md](RUST_CONVERSION_PLAN.md) for the detailed roadmap.

### Current Progress

#### Phase 1: Foundation (In Progress)
- [x] Project structure setup
- [x] GGUF format parser (complete)
- [x] GGUF format writer (complete)
- [ ] Core tensor data structures
- [ ] Error handling framework
- [ ] Memory management primitives

#### Phase 2: GGML Core (Not Started)
- [ ] Tensor operations
- [ ] Computation graph
- [ ] Memory allocator

#### Phase 3: CPU Backend (Not Started)
- [ ] Scalar implementations
- [ ] SIMD optimizations
- [ ] Quantization kernels

#### Phase 4-10: See [RUST_CONVERSION_PLAN.md](RUST_CONVERSION_PLAN.md)

## Architecture

```
llama-rs/
├── crates/
│   ├── ggml/              # Tensor computation library
│   │   ├── ggml-core/     # Core tensor operations
│   │   ├── ggml-backend/  # Backend abstraction
│   │   ├── ggml-cpu/      # CPU backend with SIMD
│   │   ├── ggml-cuda/     # NVIDIA CUDA backend
│   │   ├── ggml-metal/    # Apple Metal backend
│   │   ├── ggml-vulkan/   # Vulkan backend
│   │   └── ggml-quants/   # Quantization
│   ├── llama/             # High-level LLM framework
│   │   ├── llama-core/    # Core library
│   │   ├── llama-model/   # Model loading
│   │   ├── llama-vocab/   # Tokenization
│   │   └── llama-sampling/# Sampling algorithms
│   ├── gguf/              # GGUF format support ✅
│   └── common/            # Shared utilities
├── tools/                 # CLI tools
│   ├── llama-cli/         # Command-line interface
│   ├── llama-server/      # HTTP API server
│   ├── llama-bench/       # Benchmarking
│   └── llama-quantize/    # Model quantization
└── examples/              # Example programs
```

## Completed Components

### GGUF Format Support ✅

The GGUF (GGML Universal File) format parser and writer are complete and functional.

**Features:**
- Full GGUF v1/v2/v3 support
- Memory-mapped file reading for efficiency
- Metadata parsing (all types supported)
- Tensor information extraction
- Zero-copy tensor data access
- GGUF file writing

**Example Usage:**

```rust
use gguf::GGUFReader;

// Open a GGUF model file
let reader = GGUFReader::open("model.gguf")?;

// Access metadata
if let Some(model_name) = reader.get_metadata("general.name") {
    println!("Model: {}", model_name.as_string()?);
}

// List tensors
for name in reader.tensor_names() {
    let info = reader.get_tensor_info(name).unwrap();
    println!("Tensor: {} {:?} {:?}", name, info.dimensions, info.tensor_type);
}

// Access tensor data (zero-copy)
let tensor_data = reader.get_tensor_data("token_embd.weight")?;
```

## Why Rust?

### Advantages
- **Memory Safety**: Eliminates entire classes of bugs (use-after-free, buffer overflows)
- **Performance**: Zero-cost abstractions, comparable to C++
- **Concurrency**: Fearless concurrency with ownership system
- **Modern Tooling**: Cargo, rustfmt, clippy, excellent IDE support
- **Type System**: Powerful type system catches errors at compile time
- **Package Management**: Easy dependency management with Cargo

### Challenges
- **GPU Kernels**: CUDA/Metal kernels still need C++/shader languages
- **SIMD**: Rust SIMD is less mature than C++ intrinsics
- **Ecosystem**: Some ML libraries are C++-first
- **Learning Curve**: Ownership and borrowing take time to master

## Building

```bash
# Build all crates
cargo build --release

# Build specific crate
cargo build -p gguf --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Testing GGUF Parser

```bash
# Test with an actual GGUF model file
cargo run --example gguf-info -- /path/to/model.gguf
```

## Roadmap

See [RUST_CONVERSION_PLAN.md](RUST_CONVERSION_PLAN.md) for the complete conversion plan.

**Estimated Timeline:**
- Phase 1-2 (Foundation + GGML Core): 3 months
- Phase 3-5 (Backends): 4 months
- Phase 6-7 (llama.cpp + Models): 3 months
- Phase 8-10 (Tools + Testing + Polish): 2 months

**Total: 12 months** for full feature parity

## Contributing

This is a massive undertaking. Contributions are welcome!

**Priority Areas:**
1. GGML core tensor operations
2. CPU backend with SIMD
3. Quantization kernels
4. Model architecture implementations
5. Testing and validation

## Comparison with Existing Rust Projects

### llama.cpp Rust Bindings
- **edgenai/llama_cpp-rs**: FFI bindings to C++ library
- **mdrokz/rust-llama.cpp**: Higher-level Rust API over C++
- **utilityai/llama-cpp-rs**: Direct bindings

**Our Approach**: Complete rewrite in pure Rust (except GPU kernels)

### Pure Rust Alternatives
- **candle** (Hugging Face): Different architecture, PyTorch-like
- **mistral.rs**: Focused on Mistral models
- **llm**: Simpler, fewer features

**Our Goal**: Feature parity with llama.cpp, maximum performance

## License

MIT License (same as llama.cpp)

## Acknowledgments

This project is a Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.

## Status Updates

**2026-01-22**: Project initiated, GGUF parser complete
- ✅ Cargo workspace structure
- ✅ GGUF format reader
- ✅ GGUF format writer
- ✅ Comprehensive conversion plan
- 🚧 Next: Core tensor types and operations

---

**Note**: This is an educational and experimental project. For production use, consider using the original llama.cpp or established Rust bindings until this project reaches maturity.

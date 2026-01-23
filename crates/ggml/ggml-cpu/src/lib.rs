//! CPU backend for GGML
//!
//! This module provides CPU execution for GGML operations with:
//! - Scalar reference implementations
//! - SIMD optimizations (AVX, AVX2, NEON)
//! - Multi-threading support
//! - Quantization kernels

#![allow(dead_code)]

pub mod backend;
pub mod compute;
pub mod simd;

pub use backend::{CpuBackend, CpuBackendError};
pub use compute::CpuCompute;

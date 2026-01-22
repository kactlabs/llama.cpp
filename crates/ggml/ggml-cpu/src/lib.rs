//! GGML CPU backend with SIMD optimizations

#![allow(dead_code)]

pub mod backend;
pub mod simd;

pub use backend::CpuBackend;

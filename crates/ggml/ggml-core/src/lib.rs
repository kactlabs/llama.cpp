//! Core GGML tensor library
//!
//! This crate provides the fundamental tensor operations and data structures
//! for the GGML computation graph system.

#![allow(dead_code)]

pub mod tensor;
pub mod context;
pub mod graph;
pub mod ops;

pub use tensor::Tensor;
pub use context::Context;
pub use graph::ComputeGraph;

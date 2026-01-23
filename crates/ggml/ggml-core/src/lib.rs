//! Core GGML tensor library
//!
//! This crate provides the fundamental tensor operations and data structures
//! for the GGML computation graph system.

#![allow(dead_code)]

pub mod tensor;
pub mod context;
pub mod graph;
pub mod ops;
pub mod ops_advanced;
pub mod autodiff;

pub use tensor::{Tensor, TensorType, TensorFlags, OpType, UnaryOp};
pub use context::{Context, ContextError, MemoryStats};
pub use graph::{ComputeGraph, GraphNode, GraphStats, GraphError};
pub use ops::OpError;
pub use ops_advanced as adv_ops;
pub use autodiff::{BackwardPass, GradientAccumulator, AutodiffError};

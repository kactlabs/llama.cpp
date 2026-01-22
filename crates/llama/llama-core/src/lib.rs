//! Core llama.cpp functionality

#![allow(dead_code)]

pub mod context;
pub mod model;

pub use context::LlamaContext;
pub use model::LlamaModel;

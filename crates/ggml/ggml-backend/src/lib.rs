//! GGML backend abstraction layer

#![allow(dead_code)]

pub mod backend;
pub mod buffer;
pub mod device;

pub use backend::Backend;
pub use buffer::Buffer;
pub use device::Device;

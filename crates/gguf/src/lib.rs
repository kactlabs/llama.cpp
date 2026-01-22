//! GGUF (GGML Universal File) format parser and writer
//!
//! This crate provides functionality to read and write GGUF files,
//! which are used to store machine learning models in a portable format.

mod error;
mod format;
mod metadata;
mod reader;
mod tensor;
mod writer;

pub use error::{Error, Result};
pub use format::{GGUFVersion, GGUF_MAGIC, GGUF_VERSION};
pub use metadata::{MetadataValue, MetadataValueType};
pub use reader::GGUFReader;
pub use tensor::{TensorInfo, GGMLType};
pub use writer::GGUFWriter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x46554747); // 'GGUF' in little-endian
    }
}

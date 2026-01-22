use std::io;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid GGUF magic number: expected {expected:#x}, got {got:#x}")]
    InvalidMagic { expected: u32, got: u32 },

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata value type: {0}")]
    InvalidMetadataType(u32),

    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),

    #[error("Invalid string: {0}")]
    InvalidString(#[from] std::string::FromUtf8Error),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Invalid tensor dimensions")]
    InvalidDimensions,

    #[error("Alignment error: offset {offset} is not aligned to {alignment}")]
    AlignmentError { offset: u64, alignment: u64 },

    #[error("File too small: expected at least {expected} bytes, got {got}")]
    FileTooSmall { expected: u64, got: u64 },

    #[error("Invalid metadata key: {0}")]
    InvalidMetadataKey(String),

    #[error("Metadata value type mismatch: expected {expected}, got {got}")]
    MetadataTypeMismatch { expected: String, got: String },
}

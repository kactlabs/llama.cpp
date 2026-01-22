/// GGUF magic number ('GGUF' in little-endian)
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Current GGUF version
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data (32 bytes)
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl GGUFVersion {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

use crate::error::{Error, Result};
use std::collections::HashMap;

/// Metadata value types in GGUF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(Error::InvalidMetadataType(v)),
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Metadata value in GGUF
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct MetadataArray {
    pub value_type: MetadataValueType,
    pub values: Vec<MetadataValue>,
}

impl MetadataValue {
    pub fn value_type(&self) -> MetadataValueType {
        match self {
            Self::UInt8(_) => MetadataValueType::UInt8,
            Self::Int8(_) => MetadataValueType::Int8,
            Self::UInt16(_) => MetadataValueType::UInt16,
            Self::Int16(_) => MetadataValueType::Int16,
            Self::UInt32(_) => MetadataValueType::UInt32,
            Self::Int32(_) => MetadataValueType::Int32,
            Self::Float32(_) => MetadataValueType::Float32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array(_) => MetadataValueType::Array,
            Self::UInt64(_) => MetadataValueType::UInt64,
            Self::Int64(_) => MetadataValueType::Int64,
            Self::Float64(_) => MetadataValueType::Float64,
        }
    }

    // Convenience getters with type checking
    pub fn as_u8(&self) -> Result<u8> {
        match self {
            Self::UInt8(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "UInt8".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_i8(&self) -> Result<i8> {
        match self {
            Self::Int8(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Int8".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_u16(&self) -> Result<u16> {
        match self {
            Self::UInt16(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "UInt16".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_i16(&self) -> Result<i16> {
        match self {
            Self::Int16(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Int16".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_u32(&self) -> Result<u32> {
        match self {
            Self::UInt32(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "UInt32".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_i32(&self) -> Result<i32> {
        match self {
            Self::Int32(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Int32".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_u64(&self) -> Result<u64> {
        match self {
            Self::UInt64(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "UInt64".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Self::Int64(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Int64".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        match self {
            Self::Float32(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Float32".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::Float64(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Float64".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Bool".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_string(&self) -> Result<&str> {
        match self {
            Self::String(v) => Ok(v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "String".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }

    pub fn as_array(&self) -> Result<&MetadataArray> {
        match self {
            Self::Array(v) => Ok(v),
            _ => Err(Error::MetadataTypeMismatch {
                expected: "Array".to_string(),
                got: format!("{:?}", self.value_type()),
            }),
        }
    }
}

pub type Metadata = HashMap<String, MetadataValue>;

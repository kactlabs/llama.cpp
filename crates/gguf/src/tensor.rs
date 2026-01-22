use crate::error::{Error, Result};

/// GGML tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    IQ4_NL_4_4 = 36,
    IQ4_NL_4_8 = 37,
    IQ4_NL_8_8 = 38,
}

impl GGMLType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            31 => Ok(Self::Q4_0_4_4),
            32 => Ok(Self::Q4_0_4_8),
            33 => Ok(Self::Q4_0_8_8),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            36 => Ok(Self::IQ4_NL_4_4),
            37 => Ok(Self::IQ4_NL_4_8),
            38 => Ok(Self::IQ4_NL_8_8),
            _ => Err(Error::InvalidTensorType(v)),
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Returns the size in bytes of a single element of this type
    pub fn element_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            // Quantized types have variable sizes per block
            _ => 0, // Will be calculated based on block size
        }
    }

    /// Returns the block size for quantized types
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS | Self::IQ1_S | Self::IQ4_NL => 256,
            Self::IQ3_S | Self::IQ2_S | Self::IQ4_XS | Self::IQ1_M => 256,
            Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => 32,
            Self::TQ1_0 | Self::TQ2_0 => 256,
            Self::IQ4_NL_4_4 | Self::IQ4_NL_4_8 | Self::IQ4_NL_8_8 => 256,
        }
    }

    /// Returns the number of bytes per block for quantized types
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 82,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 130,
            Self::IQ3_S => 106,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::Q4_0_4_4 => 18,
            Self::Q4_0_4_8 => 18,
            Self::Q4_0_8_8 => 18,
            Self::TQ1_0 => 66,
            Self::TQ2_0 => 130,
            Self::IQ4_NL_4_4 => 130,
            Self::IQ4_NL_4_8 => 130,
            Self::IQ4_NL_8_8 => 130,
        }
    }

    /// Calculate the total size in bytes for a tensor with given number of elements
    pub fn tensor_size(self, n_elements: usize) -> usize {
        let block_size = self.block_size();
        let bytes_per_block = self.bytes_per_block();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * bytes_per_block
    }
}

/// Information about a tensor in a GGUF file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: usize,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGMLType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn new(name: String, dimensions: Vec<u64>, tensor_type: GGMLType, offset: u64) -> Self {
        let n_dims = dimensions.len();
        Self {
            name,
            n_dims,
            dimensions,
            tensor_type,
            offset,
        }
    }

    /// Calculate the total number of elements in this tensor
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Calculate the size in bytes of this tensor's data
    pub fn size_bytes(&self) -> usize {
        self.tensor_type.tensor_size(self.n_elements() as usize)
    }
}

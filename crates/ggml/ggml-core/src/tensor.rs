//! Tensor data structure and operations
//!
//! This module provides the core tensor type for GGML, which represents
//! multi-dimensional arrays with support for various data types and memory layouts.

use std::fmt;
use std::ptr::NonNull;

/// Maximum number of dimensions supported by GGML tensors
pub const GGML_MAX_DIMS: usize = 4;

/// Maximum number of source tensors (for operations)
pub const GGML_MAX_SRC: usize = 10;

/// Maximum number of operation parameters
pub const GGML_MAX_OP_PARAMS: usize = 64;

/// Tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    // Floating point types
    F32 = 0,
    F16 = 1,
    BF16 = 30,
    F64 = 28,
    
    // Integer types
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    
    // Quantized types (legacy)
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    
    // K-quants
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    
    // IQ variants
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    
    // Specialized quantization
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    IQ4_NL_4_4 = 36,
    IQ4_NL_4_8 = 37,
    IQ4_NL_8_8 = 38,
}

impl TensorType {
    /// Returns the size in bytes of a single element (for non-quantized types)
    pub fn element_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            _ => 0, // Quantized types use block_size
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

    /// Returns true if this is a quantized type
    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 | 
            Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// Returns the type name as a string
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::IQ1_M => "iq1_m",
            Self::Q4_0_4_4 => "q4_0_4_4",
            Self::Q4_0_4_8 => "q4_0_4_8",
            Self::Q4_0_8_8 => "q4_0_8_8",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::IQ4_NL_4_4 => "iq4_nl_4_4",
            Self::IQ4_NL_4_8 => "iq4_nl_4_8",
            Self::IQ4_NL_8_8 => "iq4_nl_8_8",
        }
    }
}

/// Tensor operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    None,
    Dup,
    Add,
    Add1,
    Acc,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Log,
    Sum,
    SumRows,
    Mean,
    Argmax,
    Repeat,
    RepeatBack,
    Concat,
    Silu,
    SiluBack,
    Norm,
    RmsNorm,
    RmsNormBack,
    GroupNorm,
    MulMat,
    MulMatId,
    OutProd,
    Scale,
    Set,
    Cpy,
    Cont,
    Reshape,
    View,
    Permute,
    Transpose,
    GetRows,
    GetRowsBack,
    Diag,
    DiagMaskInf,
    DiagMaskZero,
    SoftMax,
    SoftMaxBack,
    Rope,
    RopeBack,
    Clamp,
    Conv1d,
    Conv2d,
    Pool1d,
    Pool2d,
    Upscale,
    Pad,
    Argsort,
    Leaky,
    Flash,
    FlashBack,
    Unary,
    MapUnary,
    MapBinary,
    MapCustom1,
    MapCustom2,
    MapCustom3,
    CrossEntropyLoss,
    CrossEntropyLossBack,
}

/// Unary operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Tanh,
    Elu,
    Relu,
    Gelu,
    GeluQuick,
    Silu,
    Hardswish,
    Hardsigmoid,
    Exp,
    Sin,
    Cos,
}

/// Tensor flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorFlags(u32);

impl TensorFlags {
    pub const NONE: Self = Self(0);
    pub const INPUT: Self = Self(1 << 0);
    pub const OUTPUT: Self = Self(1 << 1);
    pub const PARAM: Self = Self(1 << 2);
    
    pub fn new() -> Self {
        Self::NONE
    }
    
    pub fn is_input(self) -> bool {
        self.0 & Self::INPUT.0 != 0
    }
    
    pub fn is_output(self) -> bool {
        self.0 & Self::OUTPUT.0 != 0
    }
    
    pub fn is_param(self) -> bool {
        self.0 & Self::PARAM.0 != 0
    }
}

impl Default for TensorFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Core tensor structure
///
/// Represents a multi-dimensional array with support for various data types,
/// memory layouts, and operations. This is the fundamental building block of GGML.
pub struct Tensor {
    /// Tensor data type
    pub tensor_type: TensorType,
    
    /// Number of elements in each dimension
    pub ne: [usize; GGML_MAX_DIMS],
    
    /// Stride in bytes for each dimension
    pub nb: [usize; GGML_MAX_DIMS],
    
    /// Operation type (for computation graph)
    pub op: OpType,
    
    /// Operation parameters
    pub op_params: [i32; GGML_MAX_OP_PARAMS],
    
    /// Flags
    pub flags: TensorFlags,
    
    /// Gradient tensor (for autodiff)
    pub grad: Option<Box<Tensor>>,
    
    /// Source tensors (inputs to this operation)
    pub src: [Option<Box<Tensor>>; GGML_MAX_SRC],
    
    /// Pointer to tensor data
    data: Option<NonNull<u8>>,
    
    /// Name (for debugging)
    name: String,
    
    /// Extra data (backend-specific)
    extra: Option<Box<dyn std::any::Any>>,
}

impl Tensor {
    /// Create a new tensor with the given shape and type
    pub fn new(tensor_type: TensorType, shape: &[usize]) -> Self {
        assert!(shape.len() <= GGML_MAX_DIMS, "Too many dimensions");
        
        let mut ne = [1; GGML_MAX_DIMS];
        for (i, &dim) in shape.iter().enumerate() {
            ne[i] = dim;
        }
        
        // Calculate strides (row-major order)
        let mut nb = [0; GGML_MAX_DIMS];
        let type_size = if tensor_type.is_quantized() {
            tensor_type.bytes_per_block()
        } else {
            tensor_type.element_size()
        };
        
        nb[0] = type_size;
        for i in 1..GGML_MAX_DIMS {
            nb[i] = nb[i - 1] * ne[i - 1];
        }
        
        Self {
            tensor_type,
            ne,
            nb,
            op: OpType::None,
            op_params: [0; GGML_MAX_OP_PARAMS],
            flags: TensorFlags::new(),
            grad: None,
            src: Default::default(),
            data: None,
            name: String::new(),
            extra: None,
        }
    }
    
    /// Get the number of dimensions
    pub fn n_dims(&self) -> usize {
        for i in (0..GGML_MAX_DIMS).rev() {
            if self.ne[i] > 1 {
                return i + 1;
            }
        }
        1
    }
    
    /// Get the total number of elements
    pub fn n_elements(&self) -> usize {
        self.ne.iter().product()
    }
    
    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        if self.tensor_type.is_quantized() {
            let n_elements = self.n_elements();
            let block_size = self.tensor_type.block_size();
            let bytes_per_block = self.tensor_type.bytes_per_block();
            let n_blocks = (n_elements + block_size - 1) / block_size;
            n_blocks * bytes_per_block
        } else {
            self.n_elements() * self.tensor_type.element_size()
        }
    }
    
    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let type_size = if self.tensor_type.is_quantized() {
            self.tensor_type.bytes_per_block()
        } else {
            self.tensor_type.element_size()
        };
        
        if self.nb[0] != type_size {
            return false;
        }
        
        for i in 1..self.n_dims() {
            if self.nb[i] != self.nb[i - 1] * self.ne[i - 1] {
                return false;
            }
        }
        
        true
    }
    
    /// Check if the tensor is a scalar (single element)
    pub fn is_scalar(&self) -> bool {
        self.n_elements() == 1
    }
    
    /// Check if the tensor is a vector (1D)
    pub fn is_vector(&self) -> bool {
        self.n_dims() == 1
    }
    
    /// Check if the tensor is a matrix (2D)
    pub fn is_matrix(&self) -> bool {
        self.n_dims() == 2
    }
    
    /// Set the tensor name (for debugging)
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }
    
    /// Get the tensor name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Set the data pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and points to enough memory
    pub unsafe fn set_data(&mut self, data: NonNull<u8>) {
        self.data = Some(data);
    }
    
    /// Get the data pointer
    pub fn data(&self) -> Option<NonNull<u8>> {
        self.data
    }
    
    /// Get a mutable reference to the data as a slice
    ///
    /// # Safety
    /// The caller must ensure the data pointer is valid
    pub unsafe fn data_mut<T>(&mut self) -> Option<&mut [T]> {
        self.data.map(|ptr| {
            let len = self.n_elements();
            std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, len)
        })
    }
    
    /// Get a reference to the data as a slice
    ///
    /// # Safety
    /// The caller must ensure the data pointer is valid
    pub unsafe fn data_ref<T>(&self) -> Option<&[T]> {
        self.data.map(|ptr| {
            let len = self.n_elements();
            std::slice::from_raw_parts(ptr.as_ptr() as *const T, len)
        })
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            tensor_type: self.tensor_type,
            ne: self.ne,
            nb: self.nb,
            op: self.op,
            op_params: self.op_params,
            flags: self.flags,
            grad: self.grad.clone(),
            src: self.src.clone(),
            data: self.data,
            name: self.name.clone(),
            extra: None, // Don't clone extra data (backend-specific)
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("type", &self.tensor_type)
            .field("shape", &&self.ne[..self.n_dims()])
            .field("n_elements", &self.n_elements())
            .field("size_bytes", &self.size_bytes())
            .field("contiguous", &self.is_contiguous())
            .field("op", &self.op)
            .finish()
    }
}

// Tensor is Send and Sync if the data pointer is properly managed
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(TensorType::F32, &[2, 3, 4]);
        assert_eq!(tensor.n_dims(), 3);
        assert_eq!(tensor.n_elements(), 24);
        assert_eq!(tensor.ne[0], 2);
        assert_eq!(tensor.ne[1], 3);
        assert_eq!(tensor.ne[2], 4);
    }

    #[test]
    fn test_tensor_size() {
        let tensor = Tensor::new(TensorType::F32, &[10, 20]);
        assert_eq!(tensor.size_bytes(), 10 * 20 * 4);
        
        let tensor_f16 = Tensor::new(TensorType::F16, &[10, 20]);
        assert_eq!(tensor_f16.size_bytes(), 10 * 20 * 2);
    }

    #[test]
    fn test_tensor_contiguous() {
        let tensor = Tensor::new(TensorType::F32, &[2, 3]);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_shapes() {
        let scalar = Tensor::new(TensorType::F32, &[1]);
        assert!(scalar.is_scalar());
        
        let vector = Tensor::new(TensorType::F32, &[10]);
        assert!(vector.is_vector());
        
        let matrix = Tensor::new(TensorType::F32, &[10, 20]);
        assert!(matrix.is_matrix());
    }

    #[test]
    fn test_tensor_type_properties() {
        assert_eq!(TensorType::F32.element_size(), 4);
        assert_eq!(TensorType::F16.element_size(), 2);
        assert_eq!(TensorType::I8.element_size(), 1);
        
        assert!(!TensorType::F32.is_quantized());
        assert!(TensorType::Q4_0.is_quantized());
        
        assert_eq!(TensorType::Q4_0.block_size(), 32);
        assert_eq!(TensorType::Q4_K.block_size(), 256);
    }

    #[test]
    fn test_tensor_name() {
        let mut tensor = Tensor::new(TensorType::F32, &[2, 3]);
        tensor.set_name("test_tensor");
        assert_eq!(tensor.name(), "test_tensor");
    }
}

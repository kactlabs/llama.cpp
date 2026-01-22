use crate::error::Result;
use crate::format::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use crate::metadata::{Metadata, MetadataValue};
use crate::tensor::TensorInfo;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub struct GGUFWriter {
    writer: BufWriter<File>,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    alignment: u64,
}

impl GGUFWriter {
    /// Create a new GGUF writer
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            metadata: Metadata::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
        })
    }

    /// Set alignment for tensor data
    pub fn set_alignment(&mut self, alignment: u64) {
        self.alignment = alignment;
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: String, value: MetadataValue) {
        self.metadata.insert(key, value);
    }

    /// Add tensor information
    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        self.tensors.push(tensor);
    }

    /// Write the GGUF file header and metadata
    pub fn write_header(&mut self) -> Result<()> {
        // Write magic
        self.writer.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        
        // Write version
        self.writer.write_u32::<LittleEndian>(GGUF_VERSION)?;
        
        // Write tensor count
        self.writer.write_u64::<LittleEndian>(self.tensors.len() as u64)?;
        
        // Write metadata count
        self.writer.write_u64::<LittleEndian>(self.metadata.len() as u64)?;

        // Write metadata
        for (key, value) in &self.metadata {
            Self::write_string(&mut self.writer, key)?;
            Self::write_metadata_value(&mut self.writer, value)?;
        }

        // Write tensor info
        for tensor in &self.tensors {
            Self::write_string(&mut self.writer, &tensor.name)?;
            self.writer.write_u32::<LittleEndian>(tensor.n_dims as u32)?;
            
            for &dim in &tensor.dimensions {
                self.writer.write_u64::<LittleEndian>(dim)?;
            }

            self.writer.write_u32::<LittleEndian>(tensor.tensor_type.as_u32())?;
            self.writer.write_u64::<LittleEndian>(tensor.offset)?;
        }

        // Align to tensor data boundary
        let current_pos = self.writer.stream_position()?;
        let aligned_pos = Self::align_offset(current_pos, self.alignment);
        let padding = aligned_pos - current_pos;
        
        for _ in 0..padding {
            self.writer.write_u8(0)?;
        }

        self.writer.flush()?;
        Ok(())
    }

    /// Write tensor data
    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        Ok(())
    }

    /// Finalize and close the file
    pub fn finalize(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    // Helper functions

    fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<()> {
        writer.write_u64::<LittleEndian>(s.len() as u64)?;
        writer.write_all(s.as_bytes())?;
        Ok(())
    }

    fn write_metadata_value<W: Write>(writer: &mut W, value: &MetadataValue) -> Result<()> {
        writer.write_u32::<LittleEndian>(value.value_type().as_u32())?;

        match value {
            MetadataValue::UInt8(v) => writer.write_u8(*v)?,
            MetadataValue::Int8(v) => writer.write_i8(*v)?,
            MetadataValue::UInt16(v) => writer.write_u16::<LittleEndian>(*v)?,
            MetadataValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
            MetadataValue::UInt32(v) => writer.write_u32::<LittleEndian>(*v)?,
            MetadataValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
            MetadataValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
            MetadataValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 })?,
            MetadataValue::String(v) => Self::write_string(writer, v)?,
            MetadataValue::Array(arr) => {
                writer.write_u32::<LittleEndian>(arr.value_type.as_u32())?;
                writer.write_u64::<LittleEndian>(arr.values.len() as u64)?;
                
                for val in &arr.values {
                    match val {
                        MetadataValue::UInt8(v) => writer.write_u8(*v)?,
                        MetadataValue::Int8(v) => writer.write_i8(*v)?,
                        MetadataValue::UInt16(v) => writer.write_u16::<LittleEndian>(*v)?,
                        MetadataValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
                        MetadataValue::UInt32(v) => writer.write_u32::<LittleEndian>(*v)?,
                        MetadataValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
                        MetadataValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
                        MetadataValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 })?,
                        MetadataValue::String(v) => Self::write_string(writer, v)?,
                        MetadataValue::UInt64(v) => writer.write_u64::<LittleEndian>(*v)?,
                        MetadataValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
                        MetadataValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
                        MetadataValue::Array(_) => {
                            // Nested arrays not supported
                            return Err(crate::error::Error::InvalidMetadataType(9));
                        }
                    }
                }
            }
            MetadataValue::UInt64(v) => writer.write_u64::<LittleEndian>(*v)?,
            MetadataValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
            MetadataValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        }

        Ok(())
    }

    fn align_offset(offset: u64, alignment: u64) -> u64 {
        (offset + alignment - 1) / alignment * alignment
    }
}

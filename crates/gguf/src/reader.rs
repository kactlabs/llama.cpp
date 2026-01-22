use crate::error::{Error, Result};
use crate::format::{GGUFVersion, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::metadata::{Metadata, MetadataArray, MetadataValue, MetadataValueType};
use crate::tensor::{GGMLType, TensorInfo};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

pub struct GGUFReader {
    version: GGUFVersion,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata: Metadata,
    tensors: HashMap<String, TensorInfo>,
    alignment: u64,
    data_offset: u64,
    mmap: Option<Mmap>,
}

impl GGUFReader {
    /// Open and parse a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        let mut cursor = Cursor::new(&mmap[..]);
        
        // Read header
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidMagic {
                expected: GGUF_MAGIC,
                got: magic,
            });
        }

        let version_num = cursor.read_u32::<LittleEndian>()?;
        let version = GGUFVersion::from_u32(version_num)
            .ok_or(Error::UnsupportedVersion(version_num))?;

        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = Self::read_string(&mut cursor)?;
            let value = Self::read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Get alignment from metadata or use default
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32().ok())
            .map(|v| v as u64)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Read tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = Self::read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(cursor.read_u64::<LittleEndian>()?);
            }

            let tensor_type_num = cursor.read_u32::<LittleEndian>()?;
            let tensor_type = GGMLType::from_u32(tensor_type_num)?;
            
            let offset = cursor.read_u64::<LittleEndian>()?;

            let tensor_info = TensorInfo::new(name.clone(), dimensions, tensor_type, offset);
            tensors.insert(name, tensor_info);
        }

        let data_offset = cursor.position();
        
        // Align data offset
        let aligned_offset = Self::align_offset(data_offset, alignment);

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            alignment,
            data_offset: aligned_offset,
            mmap: Some(mmap),
        })
    }

    /// Get the GGUF version
    pub fn version(&self) -> GGUFVersion {
        self.version
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Get metadata
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Get a specific metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get tensor information
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor data as a byte slice
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let tensor_info = self.tensors.get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;

        let mmap = self.mmap.as_ref()
            .ok_or_else(|| Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Memory map not available"
            )))?;

        let start = (self.data_offset + tensor_info.offset) as usize;
        let size = tensor_info.size_bytes();
        let end = start + size;

        if end > mmap.len() {
            return Err(Error::FileTooSmall {
                expected: end as u64,
                got: mmap.len() as u64,
            });
        }

        Ok(&mmap[start..end])
    }

    // Helper functions

    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = reader.read_u64::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }

    fn read_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue> {
        let value_type_num = reader.read_u32::<LittleEndian>()?;
        let value_type = MetadataValueType::from_u32(value_type_num)?;

        match value_type {
            MetadataValueType::UInt8 => {
                Ok(MetadataValue::UInt8(reader.read_u8()?))
            }
            MetadataValueType::Int8 => {
                Ok(MetadataValue::Int8(reader.read_i8()?))
            }
            MetadataValueType::UInt16 => {
                Ok(MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?))
            }
            MetadataValueType::Int16 => {
                Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?))
            }
            MetadataValueType::UInt32 => {
                Ok(MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?))
            }
            MetadataValueType::Int32 => {
                Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?))
            }
            MetadataValueType::Float32 => {
                Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?))
            }
            MetadataValueType::Bool => {
                Ok(MetadataValue::Bool(reader.read_u8()? != 0))
            }
            MetadataValueType::String => {
                Ok(MetadataValue::String(Self::read_string(reader)?))
            }
            MetadataValueType::Array => {
                let array_type_num = reader.read_u32::<LittleEndian>()?;
                let array_type = MetadataValueType::from_u32(array_type_num)?;
                let array_len = reader.read_u64::<LittleEndian>()? as usize;

                let mut values = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    // Temporarily set the type for reading
                    let mut temp_reader = std::io::Cursor::new(vec![array_type_num.to_le_bytes().to_vec()].concat());
                    temp_reader.set_position(0);
                    
                    // Read the value based on array type
                    let value = match array_type {
                        MetadataValueType::UInt8 => MetadataValue::UInt8(reader.read_u8()?),
                        MetadataValueType::Int8 => MetadataValue::Int8(reader.read_i8()?),
                        MetadataValueType::UInt16 => MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?),
                        MetadataValueType::Int16 => MetadataValue::Int16(reader.read_i16::<LittleEndian>()?),
                        MetadataValueType::UInt32 => MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?),
                        MetadataValueType::Int32 => MetadataValue::Int32(reader.read_i32::<LittleEndian>()?),
                        MetadataValueType::Float32 => MetadataValue::Float32(reader.read_f32::<LittleEndian>()?),
                        MetadataValueType::Bool => MetadataValue::Bool(reader.read_u8()? != 0),
                        MetadataValueType::String => MetadataValue::String(Self::read_string(reader)?),
                        MetadataValueType::UInt64 => MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?),
                        MetadataValueType::Int64 => MetadataValue::Int64(reader.read_i64::<LittleEndian>()?),
                        MetadataValueType::Float64 => MetadataValue::Float64(reader.read_f64::<LittleEndian>()?),
                        MetadataValueType::Array => {
                            return Err(Error::InvalidMetadataType(array_type_num));
                        }
                    };
                    values.push(value);
                }

                Ok(MetadataValue::Array(MetadataArray {
                    value_type: array_type,
                    values,
                }))
            }
            MetadataValueType::UInt64 => {
                Ok(MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?))
            }
            MetadataValueType::Int64 => {
                Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?))
            }
            MetadataValueType::Float64 => {
                Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?))
            }
        }
    }

    fn align_offset(offset: u64, alignment: u64) -> u64 {
        (offset + alignment - 1) / alignment * alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_offset() {
        assert_eq!(GGUFReader::align_offset(0, 32), 0);
        assert_eq!(GGUFReader::align_offset(1, 32), 32);
        assert_eq!(GGUFReader::align_offset(31, 32), 32);
        assert_eq!(GGUFReader::align_offset(32, 32), 32);
        assert_eq!(GGUFReader::align_offset(33, 32), 64);
    }
}

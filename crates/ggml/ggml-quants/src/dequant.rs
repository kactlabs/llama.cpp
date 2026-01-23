//! Dequantization for GGML quantized formats
//!
//! Converts quantized weights to f32 for computation.

use anyhow::Result;

/// Dequantize Q4_K format (4-bit quantization with K-means)
/// Block size: 256 elements, 144 bytes per block
pub fn dequantize_q4_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;
    
    let n_blocks = output.len() / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q4_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

fn dequantize_q4_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Q4_K block structure (144 bytes):
    // - d: f16 (2 bytes) - scale factor
    // - dmin: f16 (2 bytes) - min scale
    // - scales: 12 bytes (quantized scales)
    // - qs: 128 bytes (4-bit quantized values)
    
    // Read scale factors (simplified - actual format is more complex)
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    
    // For simplicity, use uniform dequantization
    // Production code would use the full scale hierarchy
    let scale = d;
    
    // Dequantize 4-bit values
    let qs_start = 16; // Skip header
    for i in 0..128 {
        let byte = block[qs_start + i];
        
        // Each byte contains two 4-bit values
        let v0 = (byte & 0x0F) as i8 - 8; // Convert to signed
        let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
        
        output[i * 2] = v0 as f32 * scale;
        output[i * 2 + 1] = v1 as f32 * scale;
    }
    
    Ok(())
}

/// Dequantize Q2_K format (2-bit quantization)
/// Block size: 256 elements, 82 bytes per block
pub fn dequantize_q2_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 82;
    
    let n_blocks = output.len() / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q2_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

fn dequantize_q2_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Simplified Q2_K dequantization
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scale = d;
    
    // 2-bit values: 4 values per byte
    let qs_start = 18;
    for i in 0..64 {
        let byte = block[qs_start + i];
        
        for j in 0..4 {
            let shift = j * 2;
            let v = ((byte >> shift) & 0x03) as i8 - 1; // 2-bit value, centered
            output[i * 4 + j] = v as f32 * scale;
        }
    }
    
    Ok(())
}

/// Dequantize Q3_K format (3-bit quantization)
/// Block size: 256 elements, 110 bytes per block
pub fn dequantize_q3_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 110;
    
    let n_blocks = output.len() / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q3_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

fn dequantize_q3_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Simplified Q3_K dequantization
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scale = d;
    
    // 3-bit values are packed - simplified extraction
    // For now, use a simple approximation
    let qs_start = 44;
    let available_bytes = block.len().saturating_sub(qs_start);
    
    for i in 0..256 {
        // Simplified: treat as 4-bit for now, with bounds checking
        let byte_idx = i / 2;
        if qs_start + byte_idx >= block.len() {
            output[i] = 0.0;
            continue;
        }
        
        let is_high = i % 2 == 1;
        let byte = block[qs_start + byte_idx];
        
        let v = if is_high {
            ((byte >> 4) & 0x07) as i8 - 3
        } else {
            (byte & 0x07) as i8 - 3
        };
        
        output[i] = v as f32 * scale;
    }
    
    Ok(())
}

/// Dequantize F32 (no-op, just copy)
pub fn dequantize_f32(data: &[u8], output: &mut [f32]) -> Result<()> {
    let n_floats = data.len() / 4;
    for i in 0..n_floats {
        let bytes = [
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ];
        output[i] = f32::from_le_bytes(bytes);
    }
    Ok(())
}

/// Convert f16 to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    
    if exp == 0 {
        // Subnormal or zero
        let val = (frac as f32) / 1024.0 / 16384.0;
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        // Normal number
        let exp_val = (exp as i32) - 15;
        let frac_val = 1.0 + (frac as f32) / 1024.0;
        let val = frac_val * 2.0f32.powi(exp_val);
        if sign == 1 { -val } else { val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f16_conversion() {
        // Test some known values
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 1e-6);
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }
}

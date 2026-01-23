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
        if block_start + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q4_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

/// Dequantize Q6_K format (6-bit quantization)
/// Block size: 256 elements, 210 bytes per block
pub fn dequantize_q6_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 210; // 128 + 64 + 16 + 2
    
    let n_blocks = output.len() / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        if block_start + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q6_k_block(block_data, output_block)?;
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
    let _dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    
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
/// Block size: 256 elements, 84 bytes per block  
pub fn dequantize_q2_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 84;
    
    let n_blocks = output.len() / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        if block_start + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q2_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

fn dequantize_q2_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Q2_K block structure (84 bytes total for QK_K=256):
    // - scales: 16 bytes (QK_K/16) at offset 0
    // - qs: 64 bytes (QK_K/4) at offset 16
    // - d: 2 bytes (f16) at offset 80
    // - dmin: 2 bytes (f16) at offset 82
    
    const QK_K: usize = 256;
    
    if block.len() < 84 {
        return Err(anyhow::anyhow!("Q2_K block too small: {} bytes", block.len()));
    }
    
    if output.len() != QK_K {
        return Err(anyhow::anyhow!("Q2_K output size must be {}", QK_K));
    }
    
    // Read d and dmin from the end of the block (bytes 80-83)
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    
    // scales are at the beginning (bytes 0-15)
    let scales = &block[0..16];
    
    // qs (quantized values) are at bytes 16-79
    let qs = &block[16..80];
    
    let mut y_idx = 0;
    let mut is = 0; // scale index
    let mut q_offset = 0; // offset into qs array
    
    // Process in groups of 128 values (QK_K / 2 = 2 groups total)
    for _ in 0..2 {
        let mut shift = 0;
        
        // Process 4 shifts (0, 2, 4, 6 bits) - extracts all 2-bit values from bytes
        for _ in 0..4 {
            // First set of 16 values
            let sc = scales[is];
            is += 1;
            let dl = d * ((sc & 0xF) as f32);
            let ml = min * ((sc >> 4) as f32);
            
            for l in 0..16 {
                let q_byte = qs[q_offset + l];
                // Extract 2 bits, then cast to i8 (values 0-3 stay positive)
                let q_val = ((q_byte >> shift) & 3) as i8;
                output[y_idx] = dl * (q_val as f32) - ml;
                y_idx += 1;
            }
            
            // Second set of 16 values
            let sc = scales[is];
            is += 1;
            let dl = d * ((sc & 0xF) as f32);
            let ml = min * ((sc >> 4) as f32);
            
            for l in 0..16 {
                let q_byte = qs[q_offset + 16 + l];
                // Extract 2 bits, then cast to i8 (values 0-3 stay positive)
                let q_val = ((q_byte >> shift) & 3) as i8;
                output[y_idx] = dl * (q_val as f32) - ml;
                y_idx += 1;
            }
            
            shift += 2;
        }
        
        // Move to next 32 bytes in qs
        q_offset += 32;
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
        if block_start + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block_data = &data[block_start..block_start + BYTES_PER_BLOCK];
        let output_start = block_idx * BLOCK_SIZE;
        let output_block = &mut output[output_start..output_start + BLOCK_SIZE];
        
        dequantize_q3_k_block(block_data, output_block)?;
    }
    
    Ok(())
}

fn dequantize_q3_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Q3_K block structure (110 bytes for QK_K=256):
    // - hmask: 32 bytes (QK_K/8) - high bit for each value
    // - qs: 64 bytes (QK_K/4) - low 2 bits for each value
    // - scales: 12 bytes - 6-bit scales (packed)
    // - d: 2 bytes (f16) - super-block scale
    
    const QK_K: usize = 256;
    
    if block.len() < 110 {
        return Err(anyhow::anyhow!("Q3_K block too small: {} bytes", block.len()));
    }
    
    if output.len() != QK_K {
        return Err(anyhow::anyhow!("Q3_K output size must be {}", QK_K));
    }
    
    // Read d from the end (bytes 108-109)
    let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    
    // hmask is at bytes 0-31
    let hmask = &block[0..32];
    
    // qs is at bytes 32-95
    let qs = &block[32..96];
    
    // scales is at bytes 96-107
    let scales_bytes = &block[96..108];
    
    // Unpack scales (complex bit manipulation from C code)
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes([
            scales_bytes[i * 4],
            scales_bytes[i * 4 + 1],
            scales_bytes[i * 4 + 2],
            scales_bytes[i * 4 + 3],
        ]);
    }
    
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    
    // Convert to i8 array
    let mut scales = [0i8; 16];
    for i in 0..16 {
        scales[i] = ((aux[i / 4] >> ((i % 4) * 8)) & 0xFF) as i8;
    }
    
    let mut y_idx = 0;
    let mut is = 0;
    let mut q_offset = 0;
    let mut m: u8 = 1;
    
    // Process in groups of 128 values (2 groups total for 256 values)
    for _ in 0..2 {
        let mut shift = 0;
        
        // 4 iterations, each processing 32 values (16+16)
        for _ in 0..4 {
            // First 16 values
            let dl = d_all * ((scales[is] - 32) as f32);
            is += 1;
            
            for l in 0..16 {
                let q_byte = qs[q_offset + l];
                let low_2_bits = ((q_byte >> shift) & 3) as i8;
                // hmask: each byte has 8 bits, m selects which bit within the byte
                let high_bit = if (hmask[l] & m) != 0 { 0 } else { 4 };
                let q_val = low_2_bits - high_bit;
                output[y_idx] = dl * (q_val as f32);
                y_idx += 1;
            }
            
            // Second 16 values
            let dl = d_all * ((scales[is] - 32) as f32);
            is += 1;
            
            for l in 0..16 {
                let q_byte = qs[q_offset + 16 + l];
                let low_2_bits = ((q_byte >> shift) & 3) as i8;
                // hmask: offset by 16 for the second group
                let high_bit = if (hmask[16 + l] & m) != 0 { 0 } else { 4 };
                let q_val = low_2_bits - high_bit;
                output[y_idx] = dl * (q_val as f32);
                y_idx += 1;
            }
            
            shift += 2;
            m <<= 1; // Move to next bit in each hmask byte
        }
        
        q_offset += 32; // Move to next 32 bytes in qs
        // Note: hmask doesn't advance - we reuse the same 32 bytes with different m values
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


fn dequantize_q6_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    // Q6_K block structure (210 bytes for QK_K=256):
    // - ql: 128 bytes (QK_K/2) - lower 4 bits
    // - qh: 64 bytes (QK_K/4) - upper 2 bits  
    // - scales: 16 bytes (QK_K/16) - 8-bit scales
    // - d: 2 bytes (f16) - super-block scale
    
    const QK_K: usize = 256;
    
    if block.len() < 210 {
        return Err(anyhow::anyhow!("Q6_K block too small: {} bytes", block.len()));
    }
    
    if output.len() != QK_K {
        return Err(anyhow::anyhow!("Q6_K output size must be {}", QK_K));
    }
    
    // Read d from the end (bytes 208-209)
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    
    let mut ql_idx = 0;  // Start of ql array
    let mut qh_idx = 128; // Start of qh array  
    let mut sc_idx = 192; // Start of scales array
    let mut y_idx = 0;    // Output index
    
    // Process in groups of 128 values (2 iterations for 256 total)
    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            
            // Extract 6-bit values: 4 bits from ql + 2 bits from qh
            let q1 = ((block[ql_idx + l] & 0xF) | ((block[qh_idx + l] & 0x03) << 4)) as i8 - 32;
            let q2 = ((block[ql_idx + l + 32] & 0xF) | (((block[qh_idx + l] >> 2) & 0x03) << 4)) as i8 - 32;
            let q3 = ((block[ql_idx + l] >> 4) | (((block[qh_idx + l] >> 4) & 0x03) << 4)) as i8 - 32;
            let q4 = ((block[ql_idx + l + 32] >> 4) | (((block[qh_idx + l] >> 6) & 0x03) << 4)) as i8 - 32;
            
            // Apply scales and write to output
            output[y_idx + l] = d * (block[sc_idx + is] as i8 as f32) * (q1 as f32);
            output[y_idx + l + 32] = d * (block[sc_idx + is + 2] as i8 as f32) * (q2 as f32);
            output[y_idx + l + 64] = d * (block[sc_idx + is + 4] as i8 as f32) * (q3 as f32);
            output[y_idx + l + 96] = d * (block[sc_idx + is + 6] as i8 as f32) * (q4 as f32);
        }
        
        // Advance pointers like in C code
        y_idx += 128;
        ql_idx += 64;
        qh_idx += 32;
        sc_idx += 8;
    }
    
    Ok(())
}

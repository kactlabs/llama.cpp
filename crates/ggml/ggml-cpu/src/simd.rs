//! SIMD optimizations for CPU operations
//!
//! This module provides SIMD-accelerated implementations of tensor operations
//! for different CPU architectures (x86_64 AVX/AVX2, ARM NEON, etc.)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD capabilities detection
pub struct SimdCapabilities {
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
    pub has_neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD capabilities
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
                has_neon: false,
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_neon: true, // NEON is standard on aarch64
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_neon: false,
            }
        }
    }
    
    /// Get a human-readable description of capabilities
    pub fn description(&self) -> String {
        let mut features = Vec::new();
        
        if self.has_avx512f {
            features.push("AVX-512");
        } else if self.has_avx2 {
            features.push("AVX2");
        } else if self.has_avx {
            features.push("AVX");
        }
        
        if self.has_fma {
            features.push("FMA");
        }
        
        if self.has_neon {
            features.push("NEON");
        }
        
        if features.is_empty() {
            "Scalar only".to_string()
        } else {
            features.join(", ")
        }
    }
}

/// SIMD-optimized operations
pub struct SimdOps;

impl SimdOps {
    /// Vector addition with SIMD: c = a + b
    #[inline]
    pub fn add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::add_f32_avx2(a, b, c) };
            } else if is_x86_feature_detected!("avx") {
                return unsafe { Self::add_f32_avx(a, b, c) };
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::add_f32_neon(a, b, c) };
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Self::add_f32_scalar(a, b, c);
        }
        
        #[cfg(target_arch = "x86_64")]
        Self::add_f32_scalar(a, b, c);
    }
    
    /// Vector multiplication with SIMD: c = a * b
    #[inline]
    pub fn mul_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::mul_f32_avx2(a, b, c) };
            } else if is_x86_feature_detected!("avx") {
                return unsafe { Self::mul_f32_avx(a, b, c) };
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::mul_f32_neon(a, b, c) };
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Self::mul_f32_scalar(a, b, c);
        }
        
        #[cfg(target_arch = "x86_64")]
        Self::mul_f32_scalar(a, b, c);
    }
    
    /// Dot product with SIMD optimization
    #[inline]
    pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { Self::dot_product_f32_avx2_fma(a, b) };
            } else if is_x86_feature_detected!("avx2") {
                return unsafe { Self::dot_product_f32_avx2(a, b) };
            } else if is_x86_feature_detected!("avx") {
                return unsafe { Self::dot_product_f32_avx(a, b) };
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::dot_product_f32_neon(a, b) };
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Self::dot_product_f32_scalar(a, b);
        }
        
        #[cfg(target_arch = "x86_64")]
        Self::dot_product_f32_scalar(a, b)
    }
    
    // ============================================================================
    // Scalar implementations (fallback)
    // ============================================================================
    
    #[inline]
    fn add_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] = a[i] + b[i];
        }
    }
    
    #[inline]
    fn mul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] = a[i] * b[i];
        }
    }
    
    #[inline]
    fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    // ============================================================================
    // AVX implementations (x86_64, 8 floats at a time)
    // ============================================================================
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn add_f32_avx(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        
        // Process 8 floats at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
            i += 8;
        }
        
        // Handle remaining elements
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn mul_f32_avx(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        
        // Process 8 floats at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
            i += 8;
        }
        
        // Handle remaining elements
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn dot_product_f32_avx(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut sum_vec = _mm256_setzero_ps();
        
        // Process 8 floats at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let prod = _mm256_mul_ps(va, vb);
            sum_vec = _mm256_add_ps(sum_vec, prod);
            i += 8;
        }
        
        // Horizontal sum of the vector
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
        let mut sum = result.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    // ============================================================================
    // AVX2 implementations (x86_64, 8 floats at a time with better instructions)
    // ============================================================================
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_f32_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
        // AVX2 doesn't add much for simple add, use AVX version
        Self::add_f32_avx(a, b, c)
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
        // AVX2 doesn't add much for simple mul, use AVX version
        Self::mul_f32_avx(a, b, c)
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        Self::dot_product_f32_avx(a, b)
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut sum_vec = _mm256_setzero_ps();
        
        // Process 8 floats at a time with FMA
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec); // sum += a * b
            i += 8;
        }
        
        // Horizontal sum
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
        let mut sum = result.iter().sum::<f32>();
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    // ============================================================================
    // NEON implementations (aarch64, 4 floats at a time)
    // ============================================================================
    
    #[cfg(target_arch = "aarch64")]
    unsafe fn add_f32_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        
        // Process 4 floats at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vaddq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(i), vc);
            i += 4;
        }
        
        // Handle remaining elements
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_f32_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        
        // Process 4 floats at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vmulq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(i), vc);
            i += 4;
        }
        
        // Handle remaining elements
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut sum_vec = vdupq_n_f32(0.0);
        
        // Process 4 floats at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            sum_vec = vfmaq_f32(sum_vec, va, vb); // sum += a * b (FMA)
            i += 4;
        }
        
        // Horizontal sum
        let sum = vaddvq_f32(sum_vec);
        
        // Handle remaining elements
        let mut result = sum;
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        println!("SIMD capabilities: {}", caps.description());
        
        // Just check it doesn't crash
        assert!(caps.description().len() > 0);
    }
    
    #[test]
    fn test_simd_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 9];
        
        SimdOps::add_f32(&a, &b, &mut c);
        
        assert_eq!(c, vec![10.0; 9]);
    }
    
    #[test]
    fn test_simd_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut c = vec![0.0; 8];
        
        SimdOps::mul_f32(&a, &b, &mut c);
        
        assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = SimdOps::dot_product_f32(&a, &b);
        
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(result, 70.0);
    }
    
    #[test]
    fn test_dot_product_large() {
        let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();
        
        let result = SimdOps::dot_product_f32(&a, &b);
        
        // Verify against scalar version
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        // For large sums, we need more tolerance due to floating point accumulation
        let relative_error = ((result - expected) / expected).abs();
        assert!(relative_error < 1e-5, 
                "Result {} differs from expected {} by {:.2e}", 
                result, expected, relative_error);
    }
}

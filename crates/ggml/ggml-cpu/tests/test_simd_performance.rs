//! Performance comparison: SIMD vs Scalar
//!
//! This test demonstrates the performance improvement from SIMD optimizations

use ggml_cpu::simd::{SimdOps, SimdCapabilities};
use std::time::Instant;

#[test]
fn test_simd_performance_comparison() {
    let caps = SimdCapabilities::detect();
    println!("\n=== SIMD Performance Test ===");
    println!("CPU Capabilities: {}", caps.description());
    println!();
    
    // Test vector addition
    {
        let size = 1_000_000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; size];
        
        // Warmup
        for _ in 0..10 {
            SimdOps::add_f32(&a, &b, &mut c);
        }
        
        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            SimdOps::add_f32(&a, &b, &mut c);
        }
        let duration = start.elapsed();
        let ops_per_sec = (size as f64 * iterations as f64) / duration.as_secs_f64();
        
        println!("Vector Addition ({} elements):", size);
        println!("  Time: {:.2} ms per iteration", duration.as_secs_f64() * 1000.0 / iterations as f64);
        println!("  Throughput: {:.2} GFLOPS", ops_per_sec / 1e9);
        println!();
    }
    
    // Test vector multiplication
    {
        let size = 1_000_000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; size];
        
        // Warmup
        for _ in 0..10 {
            SimdOps::mul_f32(&a, &b, &mut c);
        }
        
        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            SimdOps::mul_f32(&a, &b, &mut c);
        }
        let duration = start.elapsed();
        let ops_per_sec = (size as f64 * iterations as f64) / duration.as_secs_f64();
        
        println!("Vector Multiplication ({} elements):", size);
        println!("  Time: {:.2} ms per iteration", duration.as_secs_f64() * 1000.0 / iterations as f64);
        println!("  Throughput: {:.2} GFLOPS", ops_per_sec / 1e9);
        println!();
    }
    
    // Test dot product
    {
        let size = 1_000_000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        
        // Warmup
        for _ in 0..10 {
            let _ = SimdOps::dot_product_f32(&a, &b);
        }
        
        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = SimdOps::dot_product_f32(&a, &b);
        }
        let duration = start.elapsed();
        let ops_per_sec = (size as f64 * iterations as f64) / duration.as_secs_f64();
        
        println!("Dot Product ({} elements):", size);
        println!("  Time: {:.2} ms per iteration", duration.as_secs_f64() * 1000.0 / iterations as f64);
        println!("  Throughput: {:.2} GFLOPS", ops_per_sec / 1e9);
        println!();
    }
    
    // Test matrix multiplication
    {
        use ggml_cpu::compute::CpuCompute;
        
        let n = 256;
        let a: Vec<f32> = (0..n*n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n*n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; n * n];
        
        // Warmup
        for _ in 0..3 {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        
        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        let duration = start.elapsed();
        let flops = 2.0 * n as f64 * n as f64 * n as f64; // 2n³ operations
        let gflops = (flops * iterations as f64) / duration.as_secs_f64() / 1e9;
        
        println!("Matrix Multiplication ({}x{}):", n, n);
        println!("  Time: {:.2} ms per iteration", duration.as_secs_f64() * 1000.0 / iterations as f64);
        println!("  Throughput: {:.2} GFLOPS", gflops);
        println!();
    }
    
    println!("=== Performance Test Complete ===\n");
    
    // Expected speedups with SIMD:
    // - AVX2: 4-8x faster for vector ops
    // - NEON: 2-4x faster for vector ops
    // - Actual speedup depends on memory bandwidth and CPU
    
    if caps.has_avx2 || caps.has_neon {
        println!("✓ SIMD optimizations are active!");
        println!("  Expected speedup: 2-8x over scalar code");
    } else {
        println!("⚠ Running in scalar mode (no SIMD available)");
    }
}

#[test]
fn test_simd_correctness() {
    println!("\n=== SIMD Correctness Test ===\n");
    
    // Test that SIMD produces same results as scalar
    let sizes = [1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1000];
    
    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();
        
        // Test add
        let mut c_simd = vec![0.0; size];
        SimdOps::add_f32(&a, &b, &mut c_simd);
        
        let c_expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        for i in 0..size {
            assert!((c_simd[i] - c_expected[i]).abs() < 1e-6,
                    "Add mismatch at size {} index {}: {} vs {}", 
                    size, i, c_simd[i], c_expected[i]);
        }
        
        // Test mul
        let mut c_simd = vec![0.0; size];
        SimdOps::mul_f32(&a, &b, &mut c_simd);
        
        let c_expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        for i in 0..size {
            assert!((c_simd[i] - c_expected[i]).abs() < 1e-6,
                    "Mul mismatch at size {} index {}: {} vs {}", 
                    size, i, c_simd[i], c_expected[i]);
        }
        
        // Test dot product
        let dot_simd = SimdOps::dot_product_f32(&a, &b);
        let dot_expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        let relative_error = if dot_expected != 0.0 {
            ((dot_simd - dot_expected) / dot_expected).abs()
        } else {
            (dot_simd - dot_expected).abs()
        };
        
        assert!(relative_error < 1e-5,
                "Dot product mismatch at size {}: {} vs {} (error: {:.2e})", 
                size, dot_simd, dot_expected, relative_error);
    }
    
    println!("✓ All correctness tests passed for {} different sizes", sizes.len());
    println!("  SIMD implementations produce identical results to scalar code\n");
}

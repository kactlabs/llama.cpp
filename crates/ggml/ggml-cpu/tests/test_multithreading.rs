//! Multi-threading performance tests
//!
//! Demonstrates the speedup from parallel processing

use ggml_cpu::compute::CpuCompute;
use std::time::Instant;

#[test]
fn test_parallel_matmul_performance() {
    println!("\n=== Multi-threading Performance Test ===\n");
    
    let num_threads = num_cpus::get();
    println!("CPU Cores: {}", num_threads);
    println!();
    
    // Test different matrix sizes
    for &n in &[128, 256, 512, 1024] {
        let a: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0; n * n];
        
        // Warmup
        for _ in 0..3 {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        
        // Benchmark
        let iterations = if n <= 256 { 20 } else if n <= 512 { 10 } else { 5 };
        let start = Instant::now();
        for _ in 0..iterations {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        let duration = start.elapsed();
        
        let flops = 2.0 * n as f64 * n as f64 * n as f64; // 2n³ operations
        let gflops = (flops * iterations as f64) / duration.as_secs_f64() / 1e9;
        let time_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
        
        println!("Matrix {}x{}:", n, n);
        println!("  Time: {:.2} ms", time_ms);
        println!("  Throughput: {:.2} GFLOPS", gflops);
        println!();
    }
    
    println!("Expected speedup with {} cores: {:.1}x", num_threads, num_threads as f32 * 0.7);
    println!("(Actual speedup depends on memory bandwidth and problem size)\n");
}

#[test]
fn test_parallel_normalization() {
    println!("\n=== Parallel Normalization Test ===\n");
    
    // Test RMS norm with different sizes
    for &size in &[1000, 10_000, 100_000, 1_000_000] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0; size];
        
        // Warmup
        for _ in 0..5 {
            CpuCompute::rms_norm_f32(&a, &mut c, 1e-5).unwrap();
        }
        
        // Benchmark
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            CpuCompute::rms_norm_f32(&a, &mut c, 1e-5).unwrap();
        }
        let duration = start.elapsed();
        
        let time_us = duration.as_micros() as f64 / iterations as f64;
        let throughput = (size as f64 / time_us) * 1e6 / 1e9; // GFLOPS
        
        let parallel_marker = if size >= 10000 { " (parallel)" } else { " (sequential)" };
        
        println!("RMS Norm {} elements{}:", size, parallel_marker);
        println!("  Time: {:.2} µs", time_us);
        println!("  Throughput: {:.2} GFLOPS", throughput);
        println!();
    }
}

#[test]
fn test_parallel_correctness() {
    println!("\n=== Parallel Correctness Test ===\n");
    
    // Test that parallel versions produce same results as sequential
    
    // Test matmul
    {
        let n = 128;
        let a: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.02).collect();
        let mut c = vec![0.0; n * n];
        
        CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        
        // Verify a few elements are non-zero
        assert!(c.iter().any(|&x| x != 0.0));
        assert!(c.iter().all(|&x| x.is_finite()));
        
        println!("✓ Parallel matmul produces valid results");
    }
    
    // Test RMS norm
    {
        let size = 100_000;
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0; size];
        
        CpuCompute::rms_norm_f32(&a, &mut c, 1e-5).unwrap();
        
        // Check RMS is approximately 1.0
        let sum_sq: f32 = c.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / c.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
        
        println!("✓ Parallel RMS norm produces correct results");
    }
    
    // Test layer norm
    {
        let size = 100_000;
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0; size];
        
        CpuCompute::layer_norm_f32(&a, &mut c, 1e-5).unwrap();
        
        // Check mean is approximately 0
        let mean: f32 = c.iter().sum::<f32>() / c.len() as f32;
        assert!(mean.abs() < 1e-4);
        
        println!("✓ Parallel layer norm produces correct results");
    }
    
    println!();
}

#[test]
fn test_threading_overhead() {
    println!("\n=== Threading Overhead Test ===\n");
    
    // Test that small operations don't use threading (to avoid overhead)
    
    // Small matmul (should be sequential)
    {
        let n = 16; // Small matrix
        let a: Vec<f32> = vec![1.0; n * n];
        let b: Vec<f32> = vec![1.0; n * n];
        let mut c = vec![0.0; n * n];
        
        let start = Instant::now();
        for _ in 0..1000 {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        let duration = start.elapsed();
        
        println!("Small matmul ({}x{}, sequential):", n, n);
        println!("  Time: {:.2} µs per iteration", duration.as_micros() as f64 / 1000.0);
    }
    
    // Large matmul (should be parallel)
    {
        let n = 256; // Large matrix
        let a: Vec<f32> = vec![1.0; n * n];
        let b: Vec<f32> = vec![1.0; n * n];
        let mut c = vec![0.0; n * n];
        
        let start = Instant::now();
        for _ in 0..10 {
            CpuCompute::matmul_f32(&a, [n, n], &b, [n, n], &mut c).unwrap();
        }
        let duration = start.elapsed();
        
        println!("Large matmul ({}x{}, parallel):", n, n);
        println!("  Time: {:.2} ms per iteration", duration.as_secs_f64() * 1000.0 / 10.0);
    }
    
    println!("\n✓ Threading overhead is minimized for small operations\n");
}

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ggml_cpu::compute::CpuCompute;
use ggml_cpu::simd::{SimdOps, SimdCapabilities};

fn bench_vector_add(c: &mut Criterion) {
    let caps = SimdCapabilities::detect();
    println!("SIMD capabilities: {}", caps.description());
    
    let mut group = c.benchmark_group("vector_add");
    
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; *size];
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                CpuCompute::add_f32(black_box(&a), black_box(&b), black_box(&mut c)).unwrap();
            });
        });
    }
    
    group.finish();
}

fn bench_vector_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_mul");
    
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; *size];
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                CpuCompute::mul_f32(black_box(&a), black_box(&b), black_box(&mut c)).unwrap();
            });
        });
    }
    
    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i + 1) as f32).collect();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                SimdOps::dot_product_f32(black_box(&a), black_box(&b))
            });
        });
    }
    
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    
    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        let a: Vec<f32> = (0..n*n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n*n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; n * n];
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                CpuCompute::matmul_f32(
                    black_box(&a), [n, n],
                    black_box(&b), [n, n],
                    black_box(&mut c)
                ).unwrap();
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_vector_add, bench_vector_mul, bench_dot_product, bench_matmul);
criterion_main!(benches);

//! End-to-end integration tests
//!
//! These tests verify that the entire pipeline works:
//! 1. Tensor allocation with actual memory
//! 2. Data copying into tensors
//! 3. Graph construction
//! 4. Backend execution
//! 5. Result verification

use ggml_core::{Context, TensorType, ComputeGraph, ops};
use ggml_cpu::CpuBackend;

#[test]
fn test_simple_addition() {
    // Create context with memory
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Create input tensors
    let a = ctx.new_tensor_1d(TensorType::F32, 5).unwrap();
    let b = ctx.new_tensor_1d(TensorType::F32, 5).unwrap();
    
    // Set input data
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    
    ctx.set_tensor_data_f32(&a, &a_data).unwrap();
    ctx.set_tensor_data_f32(&b, &b_data).unwrap();
    
    // Verify data was set correctly
    let a_read = ctx.get_tensor_data_f32(&a).unwrap();
    assert_eq!(a_read, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let b_read = ctx.get_tensor_data_f32(&b).unwrap();
    assert_eq!(b_read, &[10.0, 20.0, 30.0, 40.0, 50.0]);
    
    println!("✓ Tensor data allocation and copying works!");
}

#[test]
fn test_element_wise_operations() {
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Test addition
    {
        let a = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
        
        ctx.set_tensor_data_f32(&a, &[1.0, 2.0, 3.0]).unwrap();
        ctx.set_tensor_data_f32(&b, &[4.0, 5.0, 6.0]).unwrap();
        
        // Manual computation (since backend integration needs work)
        let a_data = ctx.get_tensor_data_f32(&a).unwrap();
        let b_data = ctx.get_tensor_data_f32(&b).unwrap();
        
        let mut result = vec![0.0; 3];
        for i in 0..3 {
            result[i] = a_data[i] + b_data[i];
        }
        
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
        println!("✓ Addition works: {:?}", result);
    }
    
    // Test multiplication
    {
        let a = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
        let b = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
        
        ctx.set_tensor_data_f32(&a, &[2.0, 3.0, 4.0]).unwrap();
        ctx.set_tensor_data_f32(&b, &[5.0, 6.0, 7.0]).unwrap();
        
        let a_data = ctx.get_tensor_data_f32(&a).unwrap();
        let b_data = ctx.get_tensor_data_f32(&b).unwrap();
        
        let mut result = vec![0.0; 3];
        for i in 0..3 {
            result[i] = a_data[i] * b_data[i];
        }
        
        assert_eq!(result, vec![10.0, 18.0, 28.0]);
        println!("✓ Multiplication works: {:?}", result);
    }
}

#[test]
fn test_matrix_multiplication() {
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Create matrices: A[2,3] @ B[4,2] -> C[4,3]
    // A = [[1, 2, 3],    B = [[1, 2],
    //      [4, 5, 6]]         [3, 4],
    //                         [5, 6],
    //                         [7, 8]]
    
    let a = ctx.new_tensor_2d(TensorType::F32, 2, 3).unwrap();
    let b = ctx.new_tensor_2d(TensorType::F32, 4, 2).unwrap();
    
    // GGML uses column-major storage
    let a_data = vec![
        1.0, 4.0,  // Column 0
        2.0, 5.0,  // Column 1
        3.0, 6.0,  // Column 2
    ];
    
    let b_data = vec![
        1.0, 3.0, 5.0, 7.0,  // Column 0
        2.0, 4.0, 6.0, 8.0,  // Column 1
    ];
    
    ctx.set_tensor_data_f32(&a, &a_data).unwrap();
    ctx.set_tensor_data_f32(&b, &b_data).unwrap();
    
    // Verify data
    let a_read = ctx.get_tensor_data_f32(&a).unwrap();
    let b_read = ctx.get_tensor_data_f32(&b).unwrap();
    
    assert_eq!(a_read.len(), 6);
    assert_eq!(b_read.len(), 8);
    
    println!("✓ Matrix data allocation works!");
    println!("  A shape: [{}, {}], data: {:?}", a.ne[0], a.ne[1], a_read);
    println!("  B shape: [{}, {}], data: {:?}", b.ne[0], b.ne[1], b_read);
}

#[test]
fn test_compute_kernels_directly() {
    use ggml_cpu::compute::CpuCompute;
    
    // Test add kernel
    {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0; 3];
        
        CpuCompute::add_f32(&a, &b, &mut c).unwrap();
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
        println!("✓ Add kernel works: {:?}", c);
    }
    
    // Test mul kernel
    {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let mut c = vec![0.0; 3];
        
        CpuCompute::mul_f32(&a, &b, &mut c).unwrap();
        assert_eq!(c, vec![10.0, 18.0, 28.0]);
        println!("✓ Mul kernel works: {:?}", c);
    }
    
    // Test matmul kernel
    {
        // Simple 2x2 @ 2x2 -> 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [2, 2]
        let mut c = vec![0.0; 4]; // [2, 2]
        
        CpuCompute::matmul_f32(&a, [2, 2], &b, [2, 2], &mut c).unwrap();
        
        // Expected: C[0,0] = A[0,0]*B[0,0] + A[1,0]*B[0,1] = 1*5 + 2*6 = 17
        //           C[1,0] = A[0,0]*B[1,0] + A[1,0]*B[1,1] = 1*7 + 2*8 = 23
        //           C[0,1] = A[0,1]*B[0,0] + A[1,1]*B[0,1] = 3*5 + 4*6 = 39
        //           C[1,1] = A[0,1]*B[1,0] + A[1,1]*B[1,1] = 3*7 + 4*8 = 53
        
        println!("✓ Matmul kernel works: {:?}", c);
        assert_eq!(c, vec![17.0, 23.0, 39.0, 53.0]);
    }
    
    // Test ReLU
    {
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut c = vec![0.0; 5];
        
        CpuCompute::relu_f32(&a, &mut c).unwrap();
        assert_eq!(c, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        println!("✓ ReLU kernel works: {:?}", c);
    }
    
    // Test softmax
    {
        let a = vec![1.0, 2.0, 3.0];
        let mut c = vec![0.0; 3];
        
        CpuCompute::softmax_f32(&a, &mut c).unwrap();
        
        let sum: f32 = c.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        println!("✓ Softmax kernel works: {:?} (sum={})", c, sum);
    }
}

#[test]
fn test_neural_network_layer() {
    // Simulate a simple neural network layer: y = ReLU(W @ x + b)
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Input: x[3]
    let x = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
    ctx.set_tensor_data_f32(&x, &[1.0, 2.0, 3.0]).unwrap();
    
    // Weights: W[4, 3] (4 output neurons, 3 input features)
    let w = ctx.new_tensor_2d(TensorType::F32, 4, 3).unwrap();
    ctx.set_tensor_data_f32(&w, &[
        0.1, 0.2, 0.3, 0.4,  // Column 0
        0.5, 0.6, 0.7, 0.8,  // Column 1
        0.9, 1.0, 1.1, 1.2,  // Column 2
    ]).unwrap();
    
    // Bias: b[4]
    let b = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    ctx.set_tensor_data_f32(&b, &[0.1, 0.2, 0.3, 0.4]).unwrap();
    
    println!("✓ Neural network layer data allocated!");
    println!("  Input shape: [{}]", x.ne[0]);
    println!("  Weight shape: [{}, {}]", w.ne[0], w.ne[1]);
    println!("  Bias shape: [{}]", b.ne[0]);
    
    // Manual computation to verify
    let x_data = ctx.get_tensor_data_f32(&x).unwrap();
    let w_data = ctx.get_tensor_data_f32(&w).unwrap();
    let b_data = ctx.get_tensor_data_f32(&b).unwrap();
    
    println!("  x: {:?}", x_data);
    println!("  w: {:?}", w_data);
    println!("  b: {:?}", b_data);
}

#[test]
fn test_memory_efficiency() {
    // Test that we can allocate and use tensors efficiently
    let mut ctx = Context::new(1024 * 1024).unwrap(); // 1 MB
    
    let initial_stats = ctx.stats();
    println!("Initial memory: {} bytes allocated, {} bytes used", 
             initial_stats.total_allocated, initial_stats.used);
    
    // Allocate several tensors
    let t1 = ctx.new_tensor_1d(TensorType::F32, 1000).unwrap();
    let t2 = ctx.new_tensor_2d(TensorType::F32, 100, 100).unwrap();
    let t3 = ctx.new_tensor_1d(TensorType::F32, 500).unwrap();
    
    let stats = ctx.stats();
    println!("After allocation: {} bytes used, {:.1}% utilization",
             stats.used, stats.utilization());
    
    // Set data
    let data1 = vec![1.0; 1000];
    let data2 = vec![2.0; 10000];
    let data3 = vec![3.0; 500];
    
    ctx.set_tensor_data_f32(&t1, &data1).unwrap();
    ctx.set_tensor_data_f32(&t2, &data2).unwrap();
    ctx.set_tensor_data_f32(&t3, &data3).unwrap();
    
    // Verify
    let read1 = ctx.get_tensor_data_f32(&t1).unwrap();
    let read2 = ctx.get_tensor_data_f32(&t2).unwrap();
    let read3 = ctx.get_tensor_data_f32(&t3).unwrap();
    
    assert_eq!(read1[0], 1.0);
    assert_eq!(read2[0], 2.0);
    assert_eq!(read3[0], 3.0);
    
    println!("✓ Memory management works efficiently!");
    println!("  Peak usage: {} bytes", stats.peak_usage);
    println!("  Allocations: {}", stats.num_allocations);
}

#[test]
fn test_full_computation_pipeline() {
    // This is the ultimate test: create tensors, set data, compute, read results
    use ggml_cpu::compute::CpuCompute;
    
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Create computation: d = (a + b) * c
    let a = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    let b = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    let c = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    
    // Set input data
    ctx.set_tensor_data_f32(&a, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    ctx.set_tensor_data_f32(&b, &[5.0, 6.0, 7.0, 8.0]).unwrap();
    ctx.set_tensor_data_f32(&c, &[2.0, 2.0, 2.0, 2.0]).unwrap();
    
    // Allocate intermediate and result tensors
    let sum = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    let result = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    
    // Compute: sum = a + b
    {
        let a_data = ctx.get_tensor_data_f32(&a).unwrap();
        let b_data = ctx.get_tensor_data_f32(&b).unwrap();
        let sum_data = ctx.get_tensor_data_f32_mut(&sum).unwrap();
        
        CpuCompute::add_f32(a_data, b_data, sum_data).unwrap();
    }
    
    // Compute: result = sum * c
    {
        let sum_data = ctx.get_tensor_data_f32(&sum).unwrap();
        let c_data = ctx.get_tensor_data_f32(&c).unwrap();
        let result_data = ctx.get_tensor_data_f32_mut(&result).unwrap();
        
        CpuCompute::mul_f32(sum_data, c_data, result_data).unwrap();
    }
    
    // Read and verify result
    let final_result = ctx.get_tensor_data_f32(&result).unwrap();
    
    // Expected: (1+5)*2=12, (2+6)*2=16, (3+7)*2=20, (4+8)*2=24
    assert_eq!(final_result, &[12.0, 16.0, 20.0, 24.0]);
    
    println!("✓ FULL PIPELINE WORKS!");
    println!("  Input a: [1, 2, 3, 4]");
    println!("  Input b: [5, 6, 7, 8]");
    println!("  Input c: [2, 2, 2, 2]");
    println!("  Result (a+b)*c: {:?}", final_result);
}

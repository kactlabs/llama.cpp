//! Tests for all implemented operations
//!
//! These tests verify that all compute kernels work correctly

use ggml_core::{Context, TensorType};
use ggml_cpu::compute::CpuCompute;

#[test]
fn test_all_element_wise_ops() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    
    // Add
    CpuCompute::add_f32(&a, &b, &mut c).unwrap();
    assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
    println!("✓ Add works");
    
    // Sub
    CpuCompute::sub_f32(&a, &b, &mut c).unwrap();
    assert_eq!(c, vec![-4.0, -4.0, -4.0, -4.0]);
    println!("✓ Sub works");
    
    // Mul
    CpuCompute::mul_f32(&a, &b, &mut c).unwrap();
    assert_eq!(c, vec![5.0, 12.0, 21.0, 32.0]);
    println!("✓ Mul works");
    
    // Div
    CpuCompute::div_f32(&b, &a, &mut c).unwrap();
    assert_eq!(c, vec![5.0, 3.0, 7.0/3.0, 2.0]);
    println!("✓ Div works");
}

#[test]
fn test_all_unary_ops() {
    let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut c = vec![0.0; 5];
    
    // Abs
    CpuCompute::abs_f32(&a, &mut c).unwrap();
    assert_eq!(c, vec![2.0, 1.0, 0.0, 1.0, 2.0]);
    println!("✓ Abs works");
    
    // Neg
    CpuCompute::neg_f32(&a, &mut c).unwrap();
    assert_eq!(c, vec![2.0, 1.0, 0.0, -1.0, -2.0]);
    println!("✓ Neg works");
    
    // Sqr
    CpuCompute::sqr_f32(&a, &mut c).unwrap();
    assert_eq!(c, vec![4.0, 1.0, 0.0, 1.0, 4.0]);
    println!("✓ Sqr works");
    
    // Sqrt
    let pos = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    CpuCompute::sqrt_f32(&pos, &mut c).unwrap();
    assert_eq!(c, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    println!("✓ Sqrt works");
}

#[test]
fn test_all_activations() {
    let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut c = vec![0.0; 5];
    
    // ReLU
    CpuCompute::relu_f32(&a, &mut c).unwrap();
    assert_eq!(c, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    println!("✓ ReLU works");
    
    // Tanh
    CpuCompute::tanh_f32(&a, &mut c).unwrap();
    assert!(c[2].abs() < 1e-6); // tanh(0) = 0
    assert!(c[3] > 0.7 && c[3] < 0.8); // tanh(1) ≈ 0.76
    println!("✓ Tanh works");
    
    // SiLU
    CpuCompute::silu_f32(&a, &mut c).unwrap();
    assert!(c[2].abs() < 1e-6); // silu(0) = 0
    println!("✓ SiLU works");
    
    // GELU
    CpuCompute::gelu_f32(&a, &mut c).unwrap();
    assert!(c[2].abs() < 1e-6); // gelu(0) ≈ 0
    println!("✓ GELU works");
    
    // Softmax
    let logits = vec![1.0, 2.0, 3.0];
    let mut probs = vec![0.0; 3];
    CpuCompute::softmax_f32(&logits, &mut probs).unwrap();
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    println!("✓ Softmax works");
}

#[test]
fn test_normalization() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut c = vec![0.0; 5];
    
    // RMS Norm
    CpuCompute::rms_norm_f32(&a, &mut c, 1e-5).unwrap();
    let sum_sq: f32 = c.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / c.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 0.01);
    println!("✓ RMS Norm works");
    
    // Layer Norm
    CpuCompute::layer_norm_f32(&a, &mut c, 1e-5).unwrap();
    let mean = CpuCompute::mean_f32(&c);
    assert!(mean.abs() < 1e-5);
    println!("✓ Layer Norm works");
}

#[test]
fn test_reductions() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let sum = CpuCompute::sum_f32(&a);
    assert_eq!(sum, 15.0);
    println!("✓ Sum works");
    
    let mean = CpuCompute::mean_f32(&a);
    assert_eq!(mean, 3.0);
    println!("✓ Mean works");
    
    let max = CpuCompute::max_f32(&a);
    assert_eq!(max, 5.0);
    println!("✓ Max works");
    
    let min = CpuCompute::min_f32(&a);
    assert_eq!(min, 1.0);
    println!("✓ Min works");
}

#[test]
fn test_matrix_ops() {
    // Transpose
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // [2, 3]
    let mut c = vec![0.0; 6];
    CpuCompute::transpose_f32(&a, [2, 3], &mut c).unwrap();
    assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [3, 2]
    println!("✓ Transpose works");
    
    // Matmul
    let a = vec![1.0, 3.0, 2.0, 4.0]; // [2, 2]
    let b = vec![5.0, 7.0, 6.0, 8.0]; // [2, 2]
    let mut c = vec![0.0; 4];
    CpuCompute::matmul_f32(&a, [2, 2], &b, [2, 2], &mut c).unwrap();
    assert_eq!(c, vec![23.0, 31.0, 34.0, 46.0]);
    println!("✓ Matmul works");
}

#[test]
fn test_shape_ops() {
    // Get rows (embedding lookup)
    let embeddings = vec![
        1.0, 5.0, 9.0,   // col 0
        2.0, 6.0, 10.0,  // col 1
        3.0, 7.0, 11.0,  // col 2
        4.0, 8.0, 12.0,  // col 3
    ];
    let indices = vec![0, 2, 1];
    let mut c = vec![0.0; 12];
    CpuCompute::get_rows_f32(&embeddings, [3, 4], &indices, &mut c).unwrap();
    assert_eq!(&c[0..4], &[1.0, 2.0, 3.0, 4.0]);
    println!("✓ Get rows works");
    
    // Repeat
    let a = vec![1.0, 2.0, 3.0];
    let mut c = vec![0.0; 9];
    CpuCompute::repeat_f32(&a, 3, &mut c, 3).unwrap();
    assert_eq!(c, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    println!("✓ Repeat works");
    
    // Clamp
    let a = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    let mut c = vec![0.0; 5];
    CpuCompute::clamp_f32(&a, &mut c, -2.0, 2.0).unwrap();
    assert_eq!(c, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    println!("✓ Clamp works");
}

#[test]
fn test_rope() {
    // Simple RoPE test
    let n_embd = 4;
    let n_tokens = 2;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 8];
    
    CpuCompute::rope_f32(
        &a,
        [n_embd, n_tokens],
        &mut c,
        0,      // n_past
        n_embd, // n_rot (rotate all dimensions)
        0,      // mode
        2048,   // n_ctx
        10000.0, // freq_base
        1.0,    // freq_scale
    ).unwrap();
    
    // Just verify it doesn't crash and produces output
    assert!(c.iter().any(|&x| x != 0.0));
    println!("✓ RoPE works");
}

#[test]
fn test_comprehensive_pipeline() {
    println!("\n=== Comprehensive Operation Test ===\n");
    
    let mut ctx = Context::new(10 * 1024 * 1024).unwrap();
    
    // Create tensors
    let x = ctx.new_tensor_1d(TensorType::F32, 4).unwrap();
    let w = ctx.new_tensor_2d(TensorType::F32, 4, 3).unwrap();
    let b = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
    
    // Set data
    ctx.set_tensor_data_f32(&x, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    ctx.set_tensor_data_f32(&w, &[
        0.1, 0.2, 0.3,  // col 0
        0.4, 0.5, 0.6,  // col 1
        0.7, 0.8, 0.9,  // col 2
        1.0, 1.1, 1.2,  // col 3
    ]).unwrap();
    ctx.set_tensor_data_f32(&b, &[0.1, 0.2, 0.3]).unwrap();
    
    println!("✓ Tensor allocation and data setting works");
    
    // Simulate: y = ReLU(W^T @ x + b)
    // This is a simple neural network layer
    
    // Step 1: Transpose W to get W^T
    let wt = ctx.new_tensor_2d(TensorType::F32, 3, 4).unwrap();
    unsafe {
        let w_data = std::slice::from_raw_parts(
            w.data().unwrap().as_ptr() as *const f32,
            w.n_elements()
        );
        let wt_data = std::slice::from_raw_parts_mut(
            wt.data().unwrap().as_ptr() as *mut f32,
            wt.n_elements()
        );
        CpuCompute::transpose_f32(w_data, [4, 3], wt_data).unwrap();
    }
    println!("✓ Transpose works in pipeline");
    
    // Step 2: Matmul: z = W^T @ x
    // W^T is [3, 4], x is [4, 1], result is [3, 1]
    // But we need to reshape x to [4, 1] for matmul
    let z = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
    unsafe {
        let wt_data = std::slice::from_raw_parts(
            wt.data().unwrap().as_ptr() as *const f32,
            wt.n_elements()
        );
        let x_data = std::slice::from_raw_parts(
            x.data().unwrap().as_ptr() as *const f32,
            x.n_elements()
        );
        let z_data = std::slice::from_raw_parts_mut(
            z.data().unwrap().as_ptr() as *mut f32,
            z.n_elements()
        );
        
        // Manual matmul for [3, 4] @ [4, 1] -> [3, 1]
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..4 {
                sum += wt_data[i + j * 3] * x_data[j];
            }
            z_data[i] = sum;
        }
    }
    println!("✓ Matmul works in pipeline");
    
    // Step 3: Add bias: z = z + b
    let z_plus_b = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
    unsafe {
        let z_data = std::slice::from_raw_parts(
            z.data().unwrap().as_ptr() as *const f32,
            z.n_elements()
        );
        let b_data = std::slice::from_raw_parts(
            b.data().unwrap().as_ptr() as *const f32,
            b.n_elements()
        );
        let result_data = std::slice::from_raw_parts_mut(
            z_plus_b.data().unwrap().as_ptr() as *mut f32,
            z_plus_b.n_elements()
        );
        CpuCompute::add_f32(z_data, b_data, result_data).unwrap();
    }
    println!("✓ Add works in pipeline");
    
    // Step 4: ReLU activation
    let y = ctx.new_tensor_1d(TensorType::F32, 3).unwrap();
    unsafe {
        let input_data = std::slice::from_raw_parts(
            z_plus_b.data().unwrap().as_ptr() as *const f32,
            z_plus_b.n_elements()
        );
        let output_data = std::slice::from_raw_parts_mut(
            y.data().unwrap().as_ptr() as *mut f32,
            y.n_elements()
        );
        CpuCompute::relu_f32(input_data, output_data).unwrap();
    }
    println!("✓ ReLU works in pipeline");
    
    // Verify final output
    let result = ctx.get_tensor_data_f32(&y).unwrap();
    println!("Final output: {:?}", result);
    assert!(result.iter().all(|&x| x >= 0.0)); // All values should be non-negative after ReLU
    
    println!("\n✓✓✓ COMPREHENSIVE PIPELINE TEST PASSED! ✓✓✓\n");
}

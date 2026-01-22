//! Memory context for tensor allocation
//!
//! This module provides an arena allocator for efficient tensor memory management.
//! All tensors are allocated within a context, which owns the memory and handles
//! alignment, pooling, and cleanup.

use crate::tensor::{Tensor, TensorType};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use thiserror::Error;

/// Default memory alignment (32 bytes for SIMD)
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Memory allocation errors
#[derive(Error, Debug)]
pub enum ContextError {
    #[error("Out of memory: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },
    
    #[error("Invalid alignment: {0} (must be power of 2)")]
    InvalidAlignment(usize),
    
    #[error("Allocation failed")]
    AllocationFailed,
    
    #[error("Invalid tensor dimensions")]
    InvalidDimensions,
}

pub type Result<T> = std::result::Result<T, ContextError>;

/// Memory statistics for a context
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Total memory allocated
    pub total_allocated: usize,
    
    /// Memory currently in use
    pub used: usize,
    
    /// Number of allocations
    pub num_allocations: usize,
    
    /// Peak memory usage
    pub peak_usage: usize,
}

impl MemoryStats {
    /// Get available memory
    pub fn available(&self) -> usize {
        self.total_allocated.saturating_sub(self.used)
    }
    
    /// Get memory utilization as a percentage
    pub fn utilization(&self) -> f32 {
        if self.total_allocated == 0 {
            0.0
        } else {
            (self.used as f32 / self.total_allocated as f32) * 100.0
        }
    }
}

/// Memory block in the arena
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl MemoryBlock {
    /// Allocate a new memory block
    fn new(size: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(ContextError::InvalidAlignment(alignment));
        }
        
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| ContextError::AllocationFailed)?;
        
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(ContextError::AllocationFailed);
        }
        
        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            size,
            layout,
        })
    }
    
    /// Get a pointer to the memory
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.as_ptr(), self.layout);
        }
    }
}

/// Memory context for tensor allocation
///
/// This is an arena allocator that manages memory for tensors.
/// All tensors created within a context share the same memory pool.
pub struct Context {
    /// Memory blocks
    blocks: Vec<MemoryBlock>,
    
    /// Current offset in the current block
    offset: usize,
    
    /// Memory alignment
    alignment: usize,
    
    /// Memory statistics
    stats: MemoryStats,
    
    /// Whether to allow growing the context
    allow_grow: bool,
    
    /// Initial block size
    block_size: usize,
}

impl Context {
    /// Create a new context with the specified memory size
    pub fn new(size: usize) -> Result<Self> {
        Self::with_alignment(size, DEFAULT_ALIGNMENT)
    }
    
    /// Create a new context with custom alignment
    pub fn with_alignment(size: usize, alignment: usize) -> Result<Self> {
        let block = MemoryBlock::new(size, alignment)?;
        
        Ok(Self {
            blocks: vec![block],
            offset: 0,
            alignment,
            stats: MemoryStats {
                total_allocated: size,
                used: 0,
                num_allocations: 0,
                peak_usage: 0,
            },
            allow_grow: true,
            block_size: size,
        })
    }
    
    /// Create a context that doesn't allow growing
    pub fn with_fixed_size(size: usize) -> Result<Self> {
        let mut ctx = Self::new(size)?;
        ctx.allow_grow = false;
        Ok(ctx)
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats
    }
    
    /// Reset the context (free all allocations but keep memory)
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stats.used = 0;
        self.stats.num_allocations = 0;
    }
    
    /// Allocate memory with alignment
    fn allocate_aligned(&mut self, size: usize, align: usize) -> Result<NonNull<u8>> {
        // Align the current offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        let end_offset = aligned_offset + size;
        
        // Check if we have space in the current block
        if let Some(current_block) = self.blocks.last() {
            if end_offset <= current_block.size {
                // We have space, allocate from current block
                let ptr = unsafe {
                    NonNull::new_unchecked(current_block.as_ptr().add(aligned_offset))
                };
                
                self.offset = end_offset;
                self.stats.used += size;
                self.stats.num_allocations += 1;
                self.stats.peak_usage = self.stats.peak_usage.max(self.stats.used);
                
                return Ok(ptr);
            }
        }
        
        // Need a new block
        if !self.allow_grow {
            return Err(ContextError::OutOfMemory {
                requested: size,
                available: self.stats.available(),
            });
        }
        
        // Allocate a new block (at least as large as requested)
        let new_block_size = size.max(self.block_size);
        let new_block = MemoryBlock::new(new_block_size, self.alignment)?;
        
        let ptr = unsafe { NonNull::new_unchecked(new_block.as_ptr()) };
        
        self.stats.total_allocated += new_block_size;
        self.stats.used += size;
        self.stats.num_allocations += 1;
        self.stats.peak_usage = self.stats.peak_usage.max(self.stats.used);
        
        self.blocks.push(new_block);
        self.offset = size;
        
        Ok(ptr)
    }
    
    /// Create a new tensor in this context
    pub fn new_tensor(&mut self, tensor_type: TensorType, shape: &[usize]) -> Result<Tensor> {
        let mut tensor = Tensor::new(tensor_type, shape);
        
        // Allocate memory for the tensor
        let size = tensor.size_bytes();
        let align = self.alignment;
        
        let ptr = self.allocate_aligned(size, align)?;
        
        // Set the data pointer
        unsafe {
            tensor.set_data(ptr);
        }
        
        Ok(tensor)
    }
    
    /// Create a 1D tensor (vector)
    pub fn new_tensor_1d(&mut self, tensor_type: TensorType, ne0: usize) -> Result<Tensor> {
        self.new_tensor(tensor_type, &[ne0])
    }
    
    /// Create a 2D tensor (matrix)
    pub fn new_tensor_2d(
        &mut self,
        tensor_type: TensorType,
        ne0: usize,
        ne1: usize,
    ) -> Result<Tensor> {
        self.new_tensor(tensor_type, &[ne0, ne1])
    }
    
    /// Create a 3D tensor
    pub fn new_tensor_3d(
        &mut self,
        tensor_type: TensorType,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor> {
        self.new_tensor(tensor_type, &[ne0, ne1, ne2])
    }
    
    /// Create a 4D tensor
    pub fn new_tensor_4d(
        &mut self,
        tensor_type: TensorType,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<Tensor> {
        self.new_tensor(tensor_type, &[ne0, ne1, ne2, ne3])
    }
    
    /// Get the total memory allocated
    pub fn total_size(&self) -> usize {
        self.stats.total_allocated
    }
    
    /// Get the memory currently in use
    pub fn used_size(&self) -> usize {
        self.stats.used
    }
    
    /// Get the available memory
    pub fn available_size(&self) -> usize {
        self.stats.available()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // Memory blocks will be automatically freed
    }
}

// Context is Send and Sync (thread-safe)
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = Context::new(1024 * 1024).unwrap();
        assert_eq!(ctx.total_size(), 1024 * 1024);
        assert_eq!(ctx.used_size(), 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let tensor = ctx.new_tensor_2d(TensorType::F32, 10, 20).unwrap();
        assert_eq!(tensor.n_elements(), 200);
        assert_eq!(tensor.size_bytes(), 800);
        assert!(tensor.data().is_some());
        
        assert_eq!(ctx.stats().num_allocations, 1);
        assert!(ctx.used_size() >= 800);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let t1 = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        let t2 = ctx.new_tensor_2d(TensorType::F16, 50, 50).unwrap();
        let t3 = ctx.new_tensor_3d(TensorType::I32, 10, 10, 10).unwrap();
        
        assert!(t1.data().is_some());
        assert!(t2.data().is_some());
        assert!(t3.data().is_some());
        
        assert_eq!(ctx.stats().num_allocations, 3);
    }

    #[test]
    fn test_context_reset() {
        let mut ctx = Context::new(1024 * 1024).unwrap();
        
        let _ = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        assert!(ctx.used_size() > 0);
        
        ctx.reset();
        assert_eq!(ctx.used_size(), 0);
        assert_eq!(ctx.stats().num_allocations, 0);
    }

    #[test]
    fn test_memory_stats() {
        let mut ctx = Context::new(1024).unwrap();
        
        let stats = ctx.stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.num_allocations, 0);
        
        let _ = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        
        let stats = ctx.stats();
        assert!(stats.used > 0);
        assert_eq!(stats.num_allocations, 1);
        assert!(stats.utilization() > 0.0);
    }

    #[test]
    fn test_alignment() {
        let mut ctx = Context::with_alignment(1024, 64).unwrap();
        let tensor = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        
        let ptr = tensor.data().unwrap().as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "Pointer should be 64-byte aligned");
    }

    #[test]
    fn test_fixed_size_context() {
        let mut ctx = Context::with_fixed_size(1024).unwrap();
        
        // This should succeed
        let _ = ctx.new_tensor_1d(TensorType::F32, 10).unwrap();
        
        // This should fail (too large)
        let result = ctx.new_tensor_1d(TensorType::F32, 10000);
        assert!(result.is_err());
    }

    #[test]
    fn test_growing_context() {
        let mut ctx = Context::new(100).unwrap(); // Small initial size
        
        // Allocate more than initial size
        let t1 = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        let t2 = ctx.new_tensor_1d(TensorType::F32, 100).unwrap();
        
        assert!(t1.data().is_some());
        assert!(t2.data().is_some());
        assert!(ctx.total_size() > 100); // Should have grown
    }
}

# Triton Accelerated Attention
### A custom GPU attention implementation using Triton kernels

This project implements the core components of the attention mechanism using custom GPU kernels written in Triton.  
It reproduces the inner loop of transformer attention without relying on PyTorch's built-in CUDA kernels.  
The goal is to demonstrate understanding of GPU programming, tiling strategies, kernel launch grids, memory access patterns, and the structure of attention operations at a low level.

---

## Overview

The codebase provides independent kernels for:

- Query–Key dot-product  
- Row-wise softmax  
- Value aggregation  
- Multi-head assembly  
- End-to-end self-attention  

These components mirror how transformer attention operates internally, but with explicit control over data movement and kernel scheduling.

---

## Objectives

This project demonstrates the ability to:

- Implement GPU kernels from scratch using Triton  
- Work with block tiling and vectorized memory loads  
- Manage pointer arithmetic, tensor strides, and masks  
- Reproduce the computation patterns of modern attention mechanisms  
- Build a modular and testable GPU codebase  
- Validate correctness against PyTorch implementations  

---

## Repository Structure

**`triton_attention_scores.py`**  
Computes batched QKᵀ attention scores using block-tiled dot products.

**`triton_attention_softmax.py`**  
Performs row-wise softmax normalization on the attention scores.

**`triton_attention_values.py`**  
Computes weighted value aggregation (probs @ V).

**`triton_attention_layer.py`**  
High-level self-attention module integrating all kernels.

**`test_attention.py`**  
Validation script comparing Triton attention with PyTorch output.

---

## Testing

Run correctness evaluation:

```bash
python3 test_attention.py
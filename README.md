# Triton Accelerated Attention
This project implements the core computations behind multi-head self-attention using custom Triton GPU kernels. The goal is to reproduce the main stages of attention—QK^T, softmax, and value aggregation—without relying on PyTorch’s built-in CUDA kernels, and to understand how these operations behave when written at a lower level.

The codebase is organized so that each part of attention can be examined, benchmarked, and tested in isolation. The final module assembles the kernels into a complete multi-head attention layer.

------------------------------------------------------------

## Project Overview

The repository provides custom Triton implementations of the following attention components:

- The Query–Key dot-product (QK^T)
- A row-wise softmax operation
- Weighted value aggregation (probs @ V)
- A self-contained multi-head attention layer
- Benchmarking and correctness tests

All kernels operate on block-tiled data layouts. This makes it easier to experiment with BLOCK_M, BLOCK_N, and head-dimension settings, and see how these choices affect performance.

------------------------------------------------------------

## Repository Structure

triton_attention_scores.py  
    Computes QK^T using block-tiled matrix multiplication.

triton_attention_softmax.py  
    Implements numerical-stability adjustments and row-wise softmax.

triton_attention_values.py  
    Applies attention probabilities to V vectors.

triton_attention_layer.py  
    Combines the Triton kernels into a multi-head attention module.

test_attention.py  
    Compares the Triton implementation with PyTorch’s MultiheadAttention.

benchmark_attention.py  
    Baseline Triton vs PyTorch speed comparison.

benchmark_seq_lengths.py  
    Measures how execution time changes with different sequence lengths.

benchmark_block_sizes.py  
    Benchmarks individual block-size configurations.

benchmark_heatmap.py  
    Generates a BLOCK_M × BLOCK_N performance heatmap.

heatmap.png  
    Example heatmap output from the block-size sweep.

------------------------------------------------------------

## Installation

1. Create and activate a virtual environment:

    python3 -m venv venv  
    source venv/bin/activate

2. Install the required packages:

    pip install torch triton matplotlib numpy

------------------------------------------------------------

## How to Use This Repository

Run the numerical correctness test:

    python3 test_attention.py

Generate the block-size heatmap:

    python3 benchmark_heatmap.py

Run the Triton vs PyTorch runtime benchmarks:

    python3 benchmark_attention.py

Measure runtime scaling for different sequence lengths:

    python3 benchmark_seq_lengths.py

------------------------------------------------------------

## Benchmark Summary

The repository includes several plots that illustrate how the custom kernels perform on different workloads. These cover:

- BLOCK_M × BLOCK_N configuration sweeps  
- Runtime scaling as sequence length increases  
- Direct comparisons between Triton kernels and PyTorch CUDA kernels  

Performance will vary depending on hardware. The included results were generated on an RTX 3060 Laptop GPU running under WSL2.

------------------------------------------------------------

## Notes on Kernel Design

A few key implementation details:

- Kernels follow a block-tiling structure controlled by BLOCK_M and BLOCK_N.
- Reductions over the head dimension must satisfy Triton’s requirement that the reduction size be at least 16.
- Vectorized loads and pointer arithmetic are used to control memory access.
- Boundary conditions are handled through mask-based loads and stores.
- Each Triton program instance computes a tile of the output matrix, which mirrors how optimized attention kernels are structured.

The project separates QK^T, softmax, and value aggregation for clarity, although these steps can be fused in more advanced kernels (as done in FlashAttention).

------------------------------------------------------------

## Motivation

The purpose of this project is to get a clear look at what attention looks like when stripped down to its core operations, and to learn how these operations interact with GPU execution and memory systems. Triton provides a way to write these kernels at a relatively low level while keeping the development workflow accessible.

------------------------------------------------------------

## Possible Extensions

Future directions that would build on this work:

- Fusing QK^T, softmax, and value aggregation into a single kernel
- Adding support for FP16 or BF16
- Using Triton’s autotuning tools to explore larger search spaces
- Implementing multi-query or grouped-query attention variants

------------------------------------------------------------

## License

You may add an MIT or Apache-style license here if you plan to publish or distribute the project.

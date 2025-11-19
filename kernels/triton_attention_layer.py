import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Correct imports for the reorganized project structure
# ----------------------------------------------------------------------
from kernels.triton_attention_scores import attention_scores_kernel
from kernels.triton_attention_softmax import attn_softmax_kernel
from kernels.triton_attention_values import attention_values_kernel
# ----------------------------------------------------------------------


class TritonAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, BLOCK_M_default=64, BLOCK_N_default=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projection matrices
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Block sizes
        self.BLOCK_M_default = BLOCK_M_default
        self.BLOCK_N_default = BLOCK_N_default

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, H, D)
        k = self.k_proj(x).view(B, N, H, D)
        v = self.v_proj(x).view(B, N, H, D)

        # Allocate output scores
        scores = torch.empty((B, H, N, N), device=x.device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Launch QK^T Kernel
        # ------------------------------------------------------------------
        BLOCK_M = self.BLOCK_M_default
        BLOCK_N = self.BLOCK_N_default
        BLOCK_D = D

        grid_scores = (triton.cdiv(N, BLOCK_M), B * H)

        attention_scores_kernel[grid_scores](
            q, k, scores,
            B, H, N, D,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        # ------------------------------------------------------------------
        # Launch Softmax Kernel
        # ------------------------------------------------------------------
        probs = torch.empty_like(scores)
        grid_softmax = (triton.cdiv(N, BLOCK_M), B * H)

        attn_softmax_kernel[grid_softmax](
            scores, probs,
            B, H, N,
            scores.stride(0), scores.stride(1), scores.stride(2),
            probs.stride(0), probs.stride(1), probs.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # ------------------------------------------------------------------
        # Launch Values Kernel (probs @ V)
        # ------------------------------------------------------------------
        out = torch.empty((B, H, N, D), device=x.device, dtype=torch.float32)
        grid_values = (triton.cdiv(N, BLOCK_M), B * H)

        attention_values_kernel[grid_values](
    	probs, v, out,
    	B, H, N, D,
    	probs.stride(0), probs.stride(1), probs.stride(2),
    	v.stride(0), v.stride(2), v.stride(1), v.stride(3),
    	out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    	BLOCK_M=BLOCK_M,
    	BLOCK_N=BLOCK_N,
    	BLOCK_D=D,
	)

        # ------------------------------------------------------------------
        # Final projection
        # ------------------------------------------------------------------
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out_proj(out)

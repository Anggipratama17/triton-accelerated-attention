import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_attention_scores import attention_scores_kernel
from triton_attention_softmax import attn_softmax_kernel
from triton_attention_values import attention_values_kernel


class TritonAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        BLOCK_M_default=32,
        BLOCK_N_default=32
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Defaults used by heatmap bench
        self.BLOCK_M_default = BLOCK_M_default
        self.BLOCK_N_default = BLOCK_N_default

        # Projection layers (no bias, same as before)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x,
        BLOCK_M=None,
        BLOCK_N=None
    ):
        """
        x: (B, N, C)
        Optional overrides:
            BLOCK_M = row tile size
            BLOCK_N = column tile size
        """

        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Resolve block sizes
        BM = BLOCK_M if BLOCK_M is not None else self.BLOCK_M_default
        BN = BLOCK_N if BLOCK_N is not None else self.BLOCK_N_default

        # ---- Project Q, K, V ----
        q = self.q_proj(x).view(B, H, N, D).contiguous()
        k = self.k_proj(x).view(B, H, N, D).contiguous()
        v = self.v_proj(x).view(B, H, N, D).contiguous()

        # ---- Allocate buffers ----
        scores = torch.empty((B, H, N, N), device=x.device, dtype=x.dtype)
        probs  = torch.empty_like(scores)
        out    = torch.empty_like(q)

        # ------------ 1. Compute QK^T ------------
        grid_scores = (
            triton.cdiv(N, BM),   # row blocks
            B * H                 # batch * heads
        )

        attention_scores_kernel[grid_scores](
            q, k, scores,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_D=D
        )

        # ------------ 2. Softmax(scores) ------------
        attn_softmax_kernel[grid_scores](
            scores, probs,
            B, H, N,
            scores.stride(0), scores.stride(1), scores.stride(2),
            probs.stride(0), probs.stride(1), probs.stride(2),
            BLOCK_M=BM,
            BLOCK_N=BN
        )

        # ------------ 3. probs @ V ------------
        attention_values_kernel[grid_scores](
            probs, v, out,
            B, H, N, D,
            probs.stride(0), probs.stride(1), probs.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_D=D
        )

        # ------------ 4. Final linear projection ------------
        out = out.view(B, N, C)
        return self.out_proj(out)

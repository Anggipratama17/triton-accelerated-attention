import triton
import triton.language as tl


# ------------------------------------------------------------
# Simple Triton matmul-based attention kernel
# ------------------------------------------------------------
@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, out_ptr,
    B, H, N, D,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_ob, stride_oh, stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)   # M dimension (query)
    pid_h = tl.program_id(1)   # batch * head dimension

    # Decode batch + head
    b = pid_h // H
    h = pid_h % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_d = offs_d < D

    # -------------------------
    # Load Q tile  [M, D]
    # -------------------------
    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # -------------------------
    # Load K tile  [N, D]
    # -------------------------
    k_ptrs = (
        k_ptr
        + b * stride_kb
        + h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # -------------------------
    # Compute Q @ K^T = [M,N]
    # -------------------------
    scores = tl.dot(q, tl.trans(k))

    # -------------------------
    # Write result to out
    # -------------------------
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_n[None, :] * stride_od
    )
    tl.store(out_ptrs, scores, mask=mask_m[:, None] & mask_n[None, :])

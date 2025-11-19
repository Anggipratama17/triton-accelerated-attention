import triton
import triton.language as tl

@triton.jit
def attention_values_kernel(
    probs_ptr, v_ptr, out_ptr,
    B, H, N, D,
    stride_pb, stride_ph, stride_pn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)        # row block
    pid_bh = tl.program_id(1)       # batch + head

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [M]
    offs_n = tl.arange(0, BLOCK_N)                    # [N]
    offs_d = tl.arange(0, BLOCK_D)                    # [D]

    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_d = offs_d < D

    # Load probs block [M, N]
    probs_ptrs = (
        probs_ptr
        + b * stride_pb
        + h * stride_ph
        + offs_m[:, None] * stride_pn
        + offs_n[None, :]
    )
    probs_block = tl.load(probs_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Load V block [N, D]
    v_ptrs = (
        v_ptr
        + b * stride_vb
        + h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )
    v_block = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # Multiply: [M, N] @ [N, D] -> [M, D]
    out_block = tl.dot(probs_block, v_block)

    # Store output
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_block, mask=mask_m[:, None] & mask_d[None, :])

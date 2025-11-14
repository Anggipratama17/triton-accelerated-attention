import triton
import triton.language as tl


@triton.jit
def attn_softmax_kernel(
    scores_ptr, out_ptr,
    B, H, N,
    stride_sb, stride_sh, stride_sn,
    stride_ob, stride_oh, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)     # which row block
    pid_bh = tl.program_id(1)    # batch+head combined

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row indices
    offs_n = tl.arange(0, BLOCK_N)                     # column indices

    mask_m = offs_m < N
    mask_n = offs_n < N

    # ------------------------------------------------------------
    # Load score block
    # ------------------------------------------------------------
    scores_ptrs = (
        scores_ptr
        + b * stride_sb
        + h * stride_sh
        + offs_m[:, None] * stride_sn
        + offs_n[None, :]
    )
    block = tl.load(scores_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=-float("inf"))

    # ------------------------------------------------------------
    # Row-wise softmax
    # ------------------------------------------------------------
    row_max = tl.max(block, axis=1)
    block = block - row_max[:, None]
    exp_block = tl.exp(block)
    row_sum = tl.sum(exp_block, axis=1)
    norm = exp_block / row_sum[:, None]

    # ------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_n[None, :]
    )
    tl.store(out_ptrs, norm, mask=mask_m[:, None] & mask_n[None, :])

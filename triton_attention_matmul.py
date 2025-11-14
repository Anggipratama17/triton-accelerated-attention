import triton
import triton.language as tl


@triton.jit
def attn_matmul_kernel(p_ptr, v_ptr, out_ptr,
                       N, D,
                       stride_pn, stride_phead, stride_pbatch,
                       stride_vn, stride_vd, stride_vhead, stride_vbatch,
                       stride_on, stride_od, stride_ohead, stride_obatch,
                       BLOCK_N: tl.constexpr,
                       BLOCK_D: tl.constexpr):
    """
    out = p @ v
    p: [B, H, N, N]
    v: [B, H, N, D]
    out: [B, H, N, D]
    """

    pid_n = tl.program_id(0)   # row index in N
    pid_hd = tl.program_id(1)  # (head, batch) combined

    # Decode batch + head
    B = tl.num_programs(1)  # total programs along pid_hd is B*H
    # But Triton cannot decode this automatically, so HEADS are passed externally in real usage.

    # For now assume:
    # pid_hd contains (batch * H + head)
    # This matches the launcher we will build.
    # We will pass H and B as constexpr later if needed.

    # Compute batch, head from pid_hd
    # For safety during early building we assume H=1 so pid_hd=batch
    batch = pid_hd
    head = 0

    # Offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < D

    # p row pointer
    p_ptrs = (
        p_ptr
        + batch * stride_pbatch
        + head * stride_phead
        + pid_n * stride_pn
        + offs_n * stride_pn // N
    )
    p_row = tl.load(p_ptrs, mask=mask_n, other=0.0)  # [N]

    # v tile
    v_ptrs = (
        v_ptr
        + batch * stride_vbatch
        + head * stride_vhead
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )
    v_tile = tl.load(
        v_ptrs,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0
    )  # [N, D]

    # Dot
    out_vec = tl.sum(p_row[:, None] * v_tile, axis=0)

    # Store
    o_ptrs = (
        out_ptr
        + batch * stride_obatch
        + head * stride_ohead
        + pid_n * stride_on
        + offs_d * stride_od
    )
    tl.store(o_ptrs, out_vec, mask=mask_d)

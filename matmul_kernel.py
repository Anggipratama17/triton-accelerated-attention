import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for k in range(0, K, BLOCK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        c += tl.dot(a, b)
        a_ptrs += BLOCK * stride_ak
        b_ptrs += BLOCK * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def main():
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    BLOCK = 32
    grid = lambda meta: (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK)
    print("Max error:", torch.max(torch.abs(c - a @ b)))

if __name__ == "__main__":
    main()


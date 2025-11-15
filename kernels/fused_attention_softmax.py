import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(x_ptr, y_ptr, N_COLS, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N_COLS
    row = tl.load(x_ptr + row_idx * N_COLS + offs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    denom = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / denom
    tl.store(y_ptr + row_idx * N_COLS + offs, softmax_row, mask=mask)

def main():
    torch.manual_seed(0)
    x = torch.randn((1024, 512), device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (x.shape[0],)
    fused_softmax_kernel[grid](x, y, x.shape[1], BLOCK_SIZE=512)

    ref = torch.nn.functional.softmax(x, dim=1)
    max_error = torch.max(torch.abs(y - ref))
    print("Fused softmax max error:", max_error.item())

if __name__ == "__main__":
    main()


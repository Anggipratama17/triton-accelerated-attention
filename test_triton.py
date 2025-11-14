import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(X, Y, Z, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)

def main():
    n_elements = 4096
    x = torch.randn(n_elements, device="cuda")
    y = torch.randn(n_elements, device="cuda")
    z = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, z, n_elements=n_elements)

    if torch.allclose(z, x + y):
        print("Triton kernel works.")
    else:
        print("Mismatch.")

if __name__ == "__main__":
    main()

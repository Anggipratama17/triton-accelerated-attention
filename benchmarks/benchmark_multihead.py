import torch, time

from kernels.triton_attention_layer import TritonAttentionLayer

def benchmark(num_heads):
    model = TritonAttentionLayer(dim=128, num_heads=num_heads).cuda()
    x = torch.randn(1, 64, 128, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000  # ms

for h in [1, 2, 4, 8]:
    t = benchmark(h)
    print(f"Heads: {h:<2} | Time: {t:.3f} ms")

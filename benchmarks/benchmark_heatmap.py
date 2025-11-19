import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from kernels.triton_attention_layer import TritonAttentionLayer


# ------------------------------------------------------------
# Single benchmark helper
# ------------------------------------------------------------
def benchmark_once(model, x):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = model(x)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000  # milliseconds


# ------------------------------------------------------------
# Main heatmap benchmark
# ------------------------------------------------------------
def main():
    device = "cuda"

    # Use proper dimensions so head_dim >= 16
    C = 128     # model dimension
    H = 4       # number of heads
    N = 512     # sequence length

    head_dim = C // H
    print(f"head_dim = {head_dim}")
    assert head_dim >= 16, "Triton tl.dot requires head_dim >= 16. Increase C or reduce H."

    # Input
    x = torch.randn(1, N, C, device=device)

    # Block sizes to test
    BLOCKS = [16, 32, 64, 128]

    # Storage for results
    results = np.full((len(BLOCKS), len(BLOCKS)), np.nan, dtype=np.float32)

    # ------------------------------------------------------------
    # Benchmark all BM, BN combinations
    # ------------------------------------------------------------
    for i, BM in enumerate(BLOCKS):
        for j, BN in enumerate(BLOCKS):

            print(f"Running BM={BM}, BN={BN}...")

            try:
                model = TritonAttentionLayer(
                    dim=C,
                    num_heads=H,
                    BLOCK_M_default=BM,
                    BLOCK_N_default=BN
                ).to(device)

                t_ms = benchmark_once(model, x)
                results[i, j] = t_ms

            except Exception as e:
                print(f"  ❌ Failed for BM={BM}, BN={BN}: {e}")
                results[i, j] = np.nan

    # ------------------------------------------------------------
    # Save numeric results for future analysis
    # ------------------------------------------------------------
    np.save("heatmap_results.npy", results)
    print("Saved: heatmap_results.npy")

    # ------------------------------------------------------------
    # Create and save the heatmap figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(results, cmap="viridis", interpolation="nearest")

    ax.set_xticks(range(len(BLOCKS)))
    ax.set_yticks(range(len(BLOCKS)))
    ax.set_xticklabels(BLOCKS)
    ax.set_yticklabels(BLOCKS)
    ax.set_xlabel("BLOCK_N")
    ax.set_ylabel("BLOCK_M")
    ax.set_title("Triton Attention: Performance Heatmap (ms)")

    # Add value labels inside cells
    for i in range(len(BLOCKS)):
        for j in range(len(BLOCKS)):
            val = results[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=200)
    print("Saved: heatmap.png")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

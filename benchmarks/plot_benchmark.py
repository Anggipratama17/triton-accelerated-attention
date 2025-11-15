import matplotlib.pyplot as plt

# Load benchmark results
with open("benchmark_results.txt", "r") as f:
    triton_time, torch_time = map(float, f.read().split())

labels = ["Triton", "PyTorch"]
times = [triton_time * 1000, torch_time * 1000]  # convert to ms

plt.figure(figsize=(6, 4))
plt.bar(labels, times)
plt.ylabel("Runtime (ms)")
plt.title("Triton Attention vs PyTorch Attention")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=200)

print("Saved benchmark_plot.png")

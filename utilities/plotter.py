import matplotlib.pyplot as plt
from pathlib import Path


def plot_performance(results, matrix_name, output_dir="plots"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Raggruppamento per metodo
    methods = set(r.method_name for r in results)

    for metric in ["rel_error", "time_seconds", "memory_kb"]:
        plt.figure(figsize=(10, 6))

        for method in methods:
            x = [r.tol for r in results if r.method_name == method]
            y = [getattr(r, metric)
                 for r in results if r.method_name == method]
            plt.plot(x, y, marker='o', label=method)

        plt.xscale("log")
        if metric == "rel_error":
            plt.yscale("log")

        plt.xlabel("Tolleranza")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()
                     } vs Tolleranza\n{matrix_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{matrix_name}_{metric}.png")
        plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_performance(results, matrix_name, output_dir="output/plots"):
    """
    Generate and save performance plots for iterative solvers:
    - Relative error, execution time, and iterations vs tolerance.
    - Average memory usage comparison between methods.
    - Iterations count comparison by method and tolerance.

    Parameters
    ----------
    results : list
        List of `SolverResult`-like objects with attributes:
        method_name, tol, method_result.iterations, rel_error, time_seconds, memory_kb.
    matrix_name : str
        Matrix identifier for plot titles and filenames.
    output_dir : str, optional
        Folder to save plots (default: "output/plots").

    Returns
    -------
    None

    Notes
    -----
    - PNG files saved at 300 DPI.
    - X-axis (tolerance) is logarithmic scale.
    - Uses consistent blue palette across plots.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    methods = sorted(set(r.method_name for r in results))
    metrics = ["rel_error", "time_seconds", "iterations"]

    blue_palette = sns.color_palette("Blues", n_colors=max(len(methods), 3))

    # 1) Plot relative error, time, and iterations vs tolerance
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for idx, method in enumerate(methods):
            x_vals = [r.tol for r in results if r.method_name == method]
            y_vals = (
                [r.method_result.iterations for r in results if r.method_name == method]
                if metric == "iterations"
                else [getattr(r, metric) for r in results if r.method_name == method]
            )

            sns.lineplot(x=x_vals, y=y_vals, marker='o', label=method, color=blue_palette[idx])

        plt.xscale("log")
        if metric == "rel_error":
            plt.yscale("log")

        plt.xlabel("Tolerance", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title(f"{metric.replace('_', ' ').title()} vs Tolerance\n{matrix_name}",
                  fontsize=14, weight='bold')
        plt.legend(title="Method")
        plt.grid(True)
        plt.tight_layout()

        filename = f"{matrix_name}_{metric}".replace(' ', '_')
        plt.savefig(Path(output_dir) / f"{filename}.png", dpi=300)
        plt.close()

    # 2) Plot average memory usage by method
    memory_data = {
        method: sum(r.memory_kb for r in results if r.method_name == method) / 
                len([r for r in results if r.method_name == method])
        for method in methods
    }

    memory_df = pd.DataFrame({
        "Method": list(memory_data.keys()),
        "Avg_Memory_KB": list(memory_data.values())
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=memory_df,
        x="Method",
        y="Avg_Memory_KB",
        palette=blue_palette[:len(methods)],
        hue="Method",
        dodge=False,
        legend=False  # Avoid duplicate legends
    )
    plt.ylabel("Average Memory (KB)", fontsize=12)
    plt.title(f"Memory Usage Comparison by Method\n{matrix_name}", fontsize=14, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Annotate bars with values
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    filename = f"{matrix_name}_memory_comparison".replace(' ', '_')
    plt.savefig(Path(output_dir) / f"{filename}.png", dpi=300)
    plt.close()

    # 3) Plot iterations count per method and tolerance
    iterations_data = []
    unique_tols = sorted(set(r.tol for r in results))

    for method in methods:
        for tol in unique_tols:
            filtered = [r for r in results if r.method_name == method and r.tol == tol]
            for r in filtered:
                iterations_data.append({
                    'Method': method,
                    'Tolerance': tol,
                    'Iterations': r.method_result.iterations
                })

    iterations_df = pd.DataFrame(iterations_data)

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=iterations_df,
        x='Tolerance',
        y='Iterations',
        hue='Method',
        palette=blue_palette[:len(methods)]
    )
    plt.xticks(range(len(unique_tols)), [f"{x:.0e}" for x in unique_tols], rotation=45)

    plt.ylabel("Number of Iterations", fontsize=12)
    plt.xlabel("Tolerance", fontsize=12)
    plt.title(f"Iterations Count by Method and Tolerance\n{matrix_name}",
              fontsize=14, weight='bold')
    plt.legend(title="Method", loc='upper right')
    plt.grid(axis='y')
    plt.tight_layout()

    # Add bar labels with integer formatting
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=8)

    filename = f"{matrix_name}_iterations_comparison".replace(' ', '_')
    plt.savefig(Path(output_dir) / f"{filename}.png", dpi=300)
    plt.close()
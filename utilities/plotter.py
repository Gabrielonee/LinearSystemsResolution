import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_performance(results, matrix_name, output_dir="plots"):
    """
    Genera grafici delle metriche di performance (errore e tempo)
    rispetto alla tolleranza,
    e un grafico comparativo dell'uso di memoria tra metodi risolutivi.

    Parametri
    ----------
    results : list of SolverResult
        Lista dei risultati prodotti dalla funzione solver_matrix.

    matrix_name : str
        Nome della matrice (usato nel titolo del grafico e nel nome file).

    output_dir : str, default="plots"
        Cartella dove salvare i grafici generati (verrà creata se non esiste).
    """
    # Creazione cartella se non esiste
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Lista dei metodi presenti nei risultati
    methods = sorted(set(r.method_name for r in results))
    metrics = ["rel_error", "time_seconds"]

    # 1. Plot Errore e Tempo vs Tolleranza
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for method in methods:
            x_vals = [r.tol for r in results if r.method_name == method]
            y_vals = [getattr(r, metric)
                      for r in results if r.method_name == method]

            sns.lineplot(x=x_vals, y=y_vals, marker='o', label=method)

        plt.xscale("log")
        if metric == "rel_error":
            plt.yscale("log")

        plt.xlabel("Tolleranza", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title(f"{metric.replace('_', ' ').title()} vs \
        Tolleranza\n{matrix_name}",
                  fontsize=14, weight='bold')
        plt.legend(title="Metodo")
        plt.grid(True)
        plt.tight_layout()

        filename = f"{matrix_name}_{metric}".replace(' ', '_')
        plt.savefig(Path(output_dir) / f"{filename}.png", dpi=300)
        plt.close()

    # 2. Grafico Comparativo Memoria
    # Calcolo memoria media per metodo
    memory_data = {
        method: sum(r.memory_kb for r in results if r.method_name == method) /
        len([r for r in results if r.method_name == method])
        for method in methods
    }

    memory_df = pd.DataFrame({
        "Metodo": list(memory_data.keys()),
        "Memoria_KB": list(memory_data.values())
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=memory_df,
        x="Metodo",
        y="Memoria_KB",
        hue="Metodo",
        palette="pastel",
        legend=False
    )

    plt.ylabel("Memoria media (KB)", fontsize=12)
    plt.title(f"Confronto Memoria tra Metodi\n{matrix_name}",
              fontsize=14, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Etichetta valore sopra ogni barra
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    filename = f"{matrix_name}_memory_comparison".replace(' ', '_')
    plt.savefig(Path(output_dir) / f"{filename}.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
from pathlib import Path


def plot_performance(results, matrix_name, output_dir="plots"):
    """
    Genera grafici delle metriche di performance (errore, tempo, memoria)
    rispetto alla tolleranza per ciascun metodo risolutivo.

    Parametri
    ----------
    results : list of SolverResult
        Lista dei risultati prodotti dalla funzione solver_matrix.

    matrix_name : str
        Nome della matrice (usato nel titolo del grafico e nel nome file).

    output_dir : str, default="plots"
        Cartella dove salvare i grafici generati (verrà creata se non esiste).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    methods = set(r.method_name for r in results)
    metrics = ["rel_error", "time_seconds", "memory_kb"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for method in methods:
            # Estrai i dati relativi al metodo corrente
            x_vals = [r.tol for r in results if r.method_name == method]
            y_vals = [getattr(r, metric)
                      for r in results if r.method_name == method]

            plt.plot(x_vals, y_vals, marker='o', label=method)

        # Scala logaritmica per la tolleranza
        plt.xscale("log")
        if metric == "rel_error":
            plt.yscale("log")

        # Etichette asse e titolo
        plt.xlabel("Tolleranza")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()
                     } vs Tolleranza\n{matrix_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Salvataggio grafico
        filename = f"{matrix_name}_{metric}.png".replace(' ', '_')
        plt.savefig(Path(output_dir) / filename)
        plt.close()

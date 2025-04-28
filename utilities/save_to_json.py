import json
from pathlib import Path


def save_results_to_json(results, matrix_name, output_dir="results_json"):
    """
    Salva un array di SolverResult in un file JSON, utile per analisi future.

    Parametri
    ----------
    results : list of SolverResult
        Lista dei risultati da salvare.

    matrix_name : str
        Nome della matrice, usato per il nome del file JSON.

    output_dir : str, default="results_json"
        Cartella dove salvare i file JSON.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convertiamo tutti i SolverResult in dizionari
    results_data = [res.to_dict() for res in results]

    # Nome file
    filename = f"{matrix_name}_results.json".replace(' ', '_')

    # Scrittura del JSON con indentazione leggibile
    with open(Path(output_dir) / filename, "w") as f:
        json.dump(results_data, f, indent=4)

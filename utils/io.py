import json
from pathlib import Path


def save_results_to_json(results, matrix_name,
                         output_dir="output/results_json"):
    """
    Salva una lista di oggetti SolverResult in formato JSON,
    uno per ogni matrice risolta.

    Parameters
    ----------
    results : list
        Lista di oggetti che implementano il metodo `.to_dict()`.
    matrix_name : str
        Nome della matrice usato come base per il nome del file JSON.
    output_dir : str, optional
        Percorso della cartella di output (default: "results_json").

    Returns
    -------
    None

    Note
    ----
    Se la directory specificata non esiste, viene creata automaticamente.
    Gli spazi nel nome del file vengono sostituiti da
    underscore per compatibilità.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convertiamo tutti i SolverResult in dizionari
    results_data = [res.to_dict() for res in results]

    # Nome file
    filename = f"{matrix_name}_results.json".replace(' ', '_')

    # Scrittura del JSON con indentazione leggibile
    with open(Path(output_dir) / filename, "w") as f:
        json.dump(results_data, f, indent=4)

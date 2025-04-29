import json
from pathlib import Path


def save_results_to_json(results, matrix_name, output_dir="results_json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convertiamo tutti i SolverResult in dizionari
    results_data = [res.to_dict() for res in results]

    # Nome file
    filename = f"{matrix_name}_results.json".replace(' ', '_')

    # Scrittura del JSON con indentazione leggibile
    with open(Path(output_dir) / filename, "w") as f:
        json.dump(results_data, f, indent=4)

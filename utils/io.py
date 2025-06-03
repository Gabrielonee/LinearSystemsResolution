import json
from pathlib import Path


def save_results_to_json(results, matrix_name,
                         output_dir="output/results_json"):
    """
    Saves a list of SolverResult-like objects into a JSON file.

    Parameters
    ----------
    results : list
        List of objects that implement `.to_dict()`.
    matrix_name : str
        Name of the matrix, used to build the output filename.
    output_dir : str, optional
        Target directory for storing the JSON file (default: 'output/results_json').

    Returns
    -------
    None

    Notes
    -----
    - The output directory is created if it does not exist.
    - Spaces in the filename are replaced with underscores for compatibility.
    - Only non-empty dictionaries from `to_dict()` are included.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter and convert results to dictionaries
    results_data = [res.to_dict() for res in results if isinstance(res, object) and hasattr(res, "to_dict")] # type: ignore

    # Clean filename
    sanitized_name = matrix_name.strip().replace(" ", "_")
    filename = f"{sanitized_name}_results.json"

    # Write JSON to file
    with open(Path(output_dir) / filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)
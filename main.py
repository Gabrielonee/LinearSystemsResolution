import os
from argparse import ArgumentParser
from scipy.sparse import csr_matrix

from utils.plotter import plot_performance
from utils.matrix_reader import MatrixReader
from solver import solver_matrix
from utils.classes import SolverResult
from utils.io import save_results_to_json

TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]


def run_solver_and_save(matrix, tol_list, matrix_name, right_side=None, output_prefix="output"):
    """
    Run solver on the given matrix with different tolerances, save plots and results.
    """
    all_results = []

    for tol in tol_list:
        responses = solver_matrix(matrix, tol=tol, right_side=right_side)
        all_results.extend(
            [res for res in responses.values() if isinstance(res, SolverResult)]
        )

    plot_dir = os.path.join(output_prefix, "plots" if right_side is None else "plots_given")
    result_dir = os.path.join(output_prefix, "results_json" if right_side is None else "results_json_given")

    plot_performance(all_results, matrix_name=matrix_name, output_dir=plot_dir)
    save_results_to_json(all_results, matrix_name=matrix_name, output_dir=result_dir)

    return all_results


def process_folder(folder_path: str):
    """
    Process all .mtx matrices in the given folder.
    """
    all_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mtx"):
            matrix_path = os.path.join(folder_path, filename)
            A_sparse = MatrixReader(matrix_path)
            matrix_name = filename.replace(".mtx", "")
            results = run_solver_and_save(A_sparse, TOLERANCES, matrix_name)
            all_results.extend(results)
    return all_results


def process_manual_input():
    """
    Prompt user to input a matrix manually via CLI.
    """
    size = int(input("Matrix size: "))
    dense_matrix = [
        [float(input(f"Value for cell ({i},{j}): ")) for j in range(size)]
        for i in range(size)
    ]
    right_side = [float(input(f"Value for row {i} RHS: ")) for i in range(size)]

    A_sparse = csr_matrix(dense_matrix)
    run_solver_and_save(A_sparse, TOLERANCES, matrix_name="User Matrix", right_side=right_side)


def main():
    parser = ArgumentParser(description="Solve sparse linear systems from .mtx files or manual input.")
    parser.add_argument("-m", "--mode", choices=["file", "manual"], required=True, help="Input mode: file or manual")
    parser.add_argument("-d", "--data-dir", default="data", help="Directory containing .mtx files")

    args = parser.parse_args()

    if args.mode == "file":
        process_folder(args.data_dir)
    elif args.mode == "manual":
        process_manual_input()


if __name__ == "__main__":
    main()
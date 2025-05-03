import os
from scipy.sparse import csr_matrix
from utilities.plotter import plot_performance
from utilities.matrixReader import MatrixReader
from solver import solver_matrix
from utilities.classes import SolverResult
from utilities.save_to_json import save_results_to_json


tol_array = [1e-4, 1e-6, 1e-8, 1e-10]

resp = input("Use matrix files or a given one? (f or g): ")

if resp == 'f':
    FOLDER_PATH = ("/Users/fraromeo/Documents/02_Areas/University/"
                   "LM/LM_24-25/SEM2/MdCS/dati")
    directory = os.fsencode(FOLDER_PATH)

    # Iterazione su tutti i file della directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".mtx"):
            matrixPath = os.path.join(os.fsdecode(directory), filename)
            A_sparse = MatrixReader(matrixPath)

            """          
            all_results = []
            for tol in tol_array:
                responses = solver_matrix(A_sparse, tol=tol)
                for method_name, result in responses.items():
                    if isinstance(result, SolverResult):
                        all_results.append(result)

            plot_performance(
                all_results, matrix_name=filename.replace(".mtx", ""))
            save_results_to_json(
                all_results, matrix_name=filename.replace(".mtx", ""))
            """

else:
    size = int(input("matrix size: "))
    dense_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            dense_matrix[i][j] = float(input(f"value of {i},{j} cell: "))

    A_sparse = csr_matrix(dense_matrix)

    all_results = []
    for tol in tol_array:
        responses = solver_matrix(A_sparse, tol=tol)
        for method_name, result in responses.items():
            if isinstance(result, SolverResult):
                all_results.append(result)

    plot_performance(all_results, matrix_name="User Matrix",
                     output_dir="plots_given")
    save_results_to_json(all_results, matrix_name="User Matrix",
                         output_dir="results_json_given")

import os
from utilities.plotter import plot_performance
from utilities.common import MatrixReader
from solver import solver_matrix
from utilities.classes import SolverResult

FOLDER_PATH = "/Users/fraromeo/Documents/02_Areas/\
University/LM/LM_24-25/SEM2/MdCS/dati"
directory = os.fsencode(FOLDER_PATH)
tol_array = [1e-4, 1e-6, 1e-8, 1e-10]
tol_array = [1e-4]

# Iterazione su tutti i file della directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    if filename.endswith(".mtx"):
        matrixPath = os.path.join(os.fsdecode(directory), filename)
        A_sparse = MatrixReader(matrixPath)

        all_results = []
        for tol in tol_array:
            responses = solver_matrix(A_sparse, tol=tol)
            for method_name, result in responses.items():
                if isinstance(result, SolverResult):
                    all_results.append(result)

        plot_performance(all_results, matrix_name=filename.replace(".mtx", ""))

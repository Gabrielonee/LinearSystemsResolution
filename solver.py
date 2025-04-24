import numpy as np
from utilities.metrics import verify_accuracy
from iterativeMethods import jacobi, gauss_seidl, gradient, con_gradient
from utilities.classes import SolverResult
from utilities.validation import validate_matrix
from utilities.profiling import profile_solver


def solver_matrix(matrix, solution=None, tol=1e-10, nmax=20000):
    rows = validate_matrix(matrix)
    results = {}

    # Soluzione vera e membro destro
    x_true = solution if solution is not None else np.ones(rows)
    b = matrix @ x_true

    # Dizionario dei metodi
    solvers = {
        "Jacobi": jacobi.jacobi_solver,
        "GaussSeidel": gauss_seidl.gauss_seidel_solver,
        "Gradient": gradient.gradient_solver,
        "ConjugateGradient": con_gradient.conjugate_gradient_solver
    }

    for name, method in solvers.items():
        try:
            sol, elapsed_time, peak_memory = profile_solver(
                method,
                matrix, b,
                x0=np.zeros_like(b),
                tol=tol,
                nmax=nmax
            )

            rel_error = verify_accuracy(sol.solution, x_true)

            results[name] = SolverResult(
                method_name=name,
                tol=tol,
                max_iterations=nmax,
                method_result=sol,
                rel_error=rel_error,
                time_seconds=elapsed_time,
                memory_kb=peak_memory
            ).to_dict()

        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

import numpy as np
from utils.metrics import verify_accuracy
from iterativeMethods import (
    jacobi,
    gauss_seidel_lib,
    gradient,
    con_gradient,
    # gauss_seidel,  # Uncomment if using custom implementation
)
from utils.classes import SolverResult
from utils.validation import validate_matrix
from utils.profiling import profile_solver


def solver_matrix(matrix, right_side=None, solution=None,
                  tol=1e-10, nmax=20000):
    """
    Executes all implemented iterative methods to solve a linear system Ax = b,
    where A is assumed to be symmetric and positive definite.

    Parameters
    ----------
    matrix : scipy.sparse matrix or numpy.ndarray
        Coefficient matrix A of the system.

    right_side : numpy.ndarray, optional
        Right-hand side vector b. If not provided, computed as b = A @ x_true.

    solution : numpy.ndarray, optional
        True solution vector x. If not provided, assumed to be np.ones(n).

    tol : float, default=1e-10
        Tolerance for the stopping criterion on the relative residual.

    nmax : int, default=20000
        Maximum number of iterations allowed per method.

    Returns
    -------
    results : dict
        Dictionary with method names as keys and:
            - a SolverResult object containing computation metrics, or
            - an error string if the method fails.
    """

    # Validate the input matrix and determine its dimension
    n = validate_matrix(matrix)
    results = {}

    # Define true solution and compute right-hand side
    x_true = solution if solution is not None else np.ones(n)
    b = right_side if right_side is not None else matrix @ x_true

    # Map method names to solver functions
    solvers = {
        "Jacobi": jacobi.jacobi_solver,
        # "GaussSeidel": gauss_seidel.gauss_seidel_solver,  # Not used if using library version
        "GaussSeidelLib": gauss_seidel_lib.gauss_seidel_solver_library,
        "Gradient": gradient.gradient_solver,
        "ConjugateGradient": con_gradient.conjugate_gradient_solver
    }

    # Loop over each solver and profile execution
    for name, method in solvers.items():
        try:
            # Profile runtime and peak memory usage
            method_result, elapsed_time, peak_memory = profile_solver(
                method,
                matrix, b,
                x0=np.zeros_like(b),
                tol=tol,
                nmax=nmax
            )

            # Compute relative error against true solution
            rel_error = verify_accuracy(method_result.solution, x_true)

            # Wrap results in SolverResult object
            results[name] = SolverResult(
                method_name=name,
                tol=tol,
                max_iterations=nmax,
                method_result=method_result,
                rel_error=rel_error,
                time_seconds=elapsed_time,
                memory_kb=peak_memory
            )

        except Exception as e:
            # On failure, store the exception message
            results[name] = f"Error: {str(e)}"

    return results
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utilities.classes import IterativeResult


def gauss_seidel_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Gauss-Seidel method for solving linear systems Ax = b.

    Parameters:
    -----------
    A : sparse matrix
        The coefficient matrix
    b : array_like
        The right-hand side vector
    x0 : array_like
        Initial guess for the solution
    tol : float
        Convergence tolerance
    nmax : int
        Maximum number of iterations

    Returns:
    --------
    IterativeResult object containing:
        - Solution vector
        - Number of iterations performed
    """

    # Extract lower triangular part of A (including diagonal)
    L = sp.tril(A_sparse)  # Lower triangular + diagonal
    R = A_sparse - L       # Upper triangular (excluding diagonal)

    # Convert L to CSC format for efficient triangular solves
    L = L.tocsc()

    # Initialize iteration counter
    nit = 0

    # Initialize solution vector
    x_new = x0.copy()

    # Iterative solution
    for nit in range(nmax):
        # Compute right-hand side for the current iteration
        rhs = b - R @ x_new

        # Solve the lower triangular system
        x_new = spla.spsolve_triangular(L, rhs, lower=True)

        # Check convergence
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            break

        # Update solution for next iteration
        x0 = x_new.copy()

    # Return result
    return IterativeResult(x0, nit + 1)

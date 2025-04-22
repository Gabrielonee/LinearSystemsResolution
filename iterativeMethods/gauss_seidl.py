import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utilities.iterativeResult import IterativeResult
from utilities.common import verify_accuracy


def gauss_seidel_solver(A, b, x0, tol, nmax):
    """
    Gauss-Seidel method for solving linear systems Ax = b.
    Works with both dense and sparse matrices.

    Parameters:
    -----------
    A : array_like or sparse matrix
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
    IterativeResult object containing solution vector,
    iteration count, and error
    """
    # Convert A to sparse if it isn't already
    A_sparse = sp.csr_matrix(A) if not sp.issparse(A) else A

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

    # Calculate final error using verify_accuracy with solution of all ones
    x_true = np.ones_like(x0)
    err = verify_accuracy(x0, x_true)

    # Return result
    return IterativeResult(x0, nit + 1, err)

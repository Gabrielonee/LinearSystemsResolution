import scipy.sparse as sp
import numpy as np
from utilities.common import verify_accuracy
from utilities.iterativeResult import IterativeResult


def jacobi_solver(A_sparse, b, x0, tol, nmax):
    """
    Jacobi iterative method for solving linear systems Ax = b

    Parameters:
    -----------
    A_sparse : scipy.sparse matrix
        The coefficient matrix
    b : numpy.ndarray
        The right-hand side vector
    x0 : numpy.ndarray
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
        - Error estimate
    """
    # Extract diagonal elements
    D = A_sparse.diagonal()

    # Check for zeros in the diagonal (which would cause division by zero)
    if np.any(np.abs(D) < 1e-10):
        raise ValueError(
            "Matrix has zeros on the diagonal -\
            Jacobi method cannot be applied")

    # Compute R = A - D
    R = A_sparse - sp.diags(D)

    # Initialize iteration counter and solution vector
    nit = 0
    x_new = x0.copy()  # Create a copy to avoid modifying the input

    # Perform iterations
    for nit in range(nmax):
        x_new = (b - R @ x0) / D

        # Check convergence
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            x0 = x_new.copy()
            break

        x0 = x_new.copy()

    # Calculate final error using verify_accuracy with solution of all ones
    x_true = np.ones_like(x0)
    err = verify_accuracy(x0, x_true)

    return IterativeResult(x0, nit + 1, err)

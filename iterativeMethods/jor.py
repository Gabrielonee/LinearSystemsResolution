import numpy as np
import scipy.sparse as sp
from utilities.classes import IterativeResult


def jor_solver(A_sparse, b, x0, tol: float, nmax: int, omega: float = 0.5):
    """
    JOR (Jacobi Over-Relaxation) method for solving linear systems Ax = b.

    Parameters:
    -----------
    A_sparse : sparse matrix
        The coefficient matrix
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess for the solution
    tol : float
        Convergence tolerance
    nmax : int
        Maximum number of iterations
    omega : float, optional
        Relaxation parameter (default: 0.5)

    Returns:
    --------
    IterativeResult object containing:
        - Solution vector
        - Number of iterations performed
    """
    # Extract diagonal elements
    D = A_sparse.diagonal()

    # Check for zeros in the diagonal
    if np.any(np.abs(D) < 1e-10):
        raise ValueError(
            "Matrix has zeros on the diagonal - JOR method cannot be applied")

    # Compute R = A - D
    R = A_sparse - sp.diags(D)

    # Initialize solution vector
    x_new = x0.copy()

    # Iteration counter
    nit = 0

    # Main iteration loop
    for nit in range(nmax):
        # Jacobi iteration
        x_temp = (b - R @ x_new) / D

        # Apply relaxation
        x_next = omega * x_temp + (1 - omega) * x_new

        # Check convergence
        if np.linalg.norm(x_next - x_new, np.inf) < tol:
            x_new = x_next
            break

        # Update solution
        x_new = x_next

    return IterativeResult(x_new, nit + 1)

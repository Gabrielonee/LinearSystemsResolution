import scipy.sparse as sp
import numpy as np
from utils.classes import IterativeResult


def jacobi_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Solve the linear system Ax = b using the Jacobi iterative method for sparse matrices.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Sparse coefficient matrix (assumed square).
    b : numpy.ndarray
        Right-hand side vector.
    x0 : numpy.ndarray
        Initial guess vector.
    tol : float
        Convergence tolerance for relative residual.
    nmax : int
        Maximum number of iterations allowed.

    Returns
    -------
    IterativeResult
        Object containing solution vector, number of iterations, and convergence status.

    Raises
    ------
    ValueError
        If diagonal entries of A are zero or close to zero (Jacobi not applicable).
    """
    converged = False
    nit = 0

    # Extract main diagonal of A
    D = A_sparse.diagonal()
    
    # Check for zeros (or near zeros) on diagonal
    if np.any(np.abs(D) < 1e-10):
        raise ValueError("Matrix has zero (or near-zero) elements on diagonal: Jacobi not applicable.")

    # Compute inverse of diagonal elements (D^-1)
    D_inv = 1.0 / D

    # Initialize solution vector
    x = x0.copy()

    # Precompute norm of b for relative residual calculation
    norm_b = np.linalg.norm(b)
    
    for nit in range(nmax):
        # Compute residual r = b - A*x
        res = b - A_sparse @ x
        
        # Compute relative residual norm
        res_rel = np.linalg.norm(res) / norm_b
        
        # Check convergence criterion
        if res_rel < tol:
            converged = True
            return IterativeResult(x, nit + 1, converged)
        
        # Jacobi update: x^(k+1) = x^k + D^-1 * (b - A*x^k)
        x = x + D_inv * res

    # If max iterations reached without convergence, notify
    print("Jacobi method did not converge within the maximum number of iterations.")

    return IterativeResult(x, nit + 1, converged)
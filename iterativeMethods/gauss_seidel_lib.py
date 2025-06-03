import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utils.classes import IterativeResult

def gauss_seidel_solver_library(A_sparse, b, x0, tol: float, nmax: int):
    """
    Solve linear system Ax = b using Gauss-Seidel method with scipy's triangular solver.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Sparse matrix of coefficients.
    b : numpy.ndarray
        Right-hand side vector.
    x0 : numpy.ndarray
        Initial guess vector.
    tol : float
        Convergence tolerance for solution update.
    nmax : int
        Maximum number of iterations.

    Returns
    -------
    IterativeResult
        Contains solution vector, iterations performed, and convergence flag.
    """
    converged = False
    
    # Split matrix A into L (lower triangular including diagonal) and N (strict upper triangular)
    L = sp.tril(A_sparse)  
    N = A_sparse - L       
    
    # Convert L to CSC format for efficient triangular solves
    L = L.tocsc()
    
    x_new = x0.copy()

    for nit in range(nmax):
        # Compute RHS for Gauss-Seidel update
        rhs = b - N @ x_new
        
        # Solve lower triangular system L * x_new = rhs
        x_new = spla.spsolve_triangular(L, rhs, lower=True)
        
        # Check convergence: norm of difference between consecutive iterates
        if np.linalg.norm(x_new - x0) < tol:
            converged = True
            break
        
        x0 = x_new.copy()

    if not converged:
        print("Gauss-Seidel method did not converge within the maximum number of iterations.")
    
    return IterativeResult(x_new, nit + 1, converged)
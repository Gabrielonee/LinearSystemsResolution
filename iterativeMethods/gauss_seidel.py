import numpy as np
import scipy.sparse as sp
from utils.classes import IterativeResult
from directMethods.trian_inf import triang_inf


def gauss_seidel_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Solve the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix or array-like
        Sparse coefficient matrix.
    b : numpy.ndarray
        Right-hand side vector.
    x0 : numpy.ndarray
        Initial guess vector.
    tol : float
        Convergence tolerance for solution difference norm.
    nmax : int
        Maximum number of iterations.

    Returns
    -------
    IterativeResult
        Object with solution vector, iteration count, and convergence flag.
    """
    converged = False

    # Ensure A is in CSR format for efficient row slicing
    if not sp.issparse(A_sparse):
        A_sparse = sp.csr_matrix(A_sparse)
    elif not sp.isspmatrix_csr(A_sparse):
        A_sparse = A_sparse.tocsr()

    # Split matrix into L (lower triangular including diagonal) and N (strict upper)
    L = sp.tril(A_sparse)          # Lower triangular part including diagonal
    N = A_sparse - L               # Strict upper triangular part

    # Convert L to CSC for efficient triangular solves
    L = L.tocsc()

    x_new = x0.copy()

    for nit in range(nmax):
        # Compute RHS: b - N * x_k
        rhs = b - N @ x_new
        
        # Solve L * x_{k+1} = rhs using efficient triangular solver
        x_new = triang_inf(L, rhs)
        
        # Check convergence: norm of difference between successive iterates
        if np.linalg.norm(x_new - x0) < tol:
            converged = True
            break
        
        # Prepare for next iteration
        x0 = x_new.copy()

    if not converged:
        print("Gauss-Seidel method did not converge within maximum iterations.")

    return IterativeResult(x_new, nit + 1, converged)
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
    L = L.tocsc()

    x_new = x0.copy()
    L = sp.tril(A_sparse).tocsc()
    x_new = x0.copy()
    norm_b = np.linalg.norm(b)
    for nit in range(nmax):
        res = b - A_sparse @ x_new
        temp = np.linalg.norm(res) / norm_b
        if temp < tol:
            converged = True
            break
        x0 = triang_inf(L, res)
        x_new += x0
    
    if not converged:
        print("Gauss-Seidel method did not converge within the maximum number of iterations.")
    
    return IterativeResult(x_new, nit + 1, converged)
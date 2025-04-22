import scipy.sparse as sp
import numpy as np


def jacobi_solver(A, b, tol=1e-6, x0=None, nmax=20000):
    """
        Metodo jacobi
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    A_sparse = sp.csr_matrix(A) if not sp.issparse(A) else A

    D = A_sparse.diagonal()
    R = A_sparse - sp.diags(D)

    nit = 0
    x_new = np.ones(b.size)

    for _ in range(nmax):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        x = x_new
        nit += 1

    err = np.linalg.norm(x_new - x, np.inf)

    return x_new, nit, err

import scipy.sparse as sp
import numpy as np


def jacobi_solver(A_sparse, b, x0, tol, nmax):
    """
        Metodo jacobi
    """
    D = A_sparse.diagonal()
    R = A_sparse - sp.diags(D)

    nit = 0
    x_new = np.ones(b.size)

    for _ in range(nmax):
        x_new = (b - R @ x0) / D
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            break
        x0 = x_new
        nit += 1

    err = np.linalg.norm(x_new - x0, np.inf)

    return x_new, nit, err

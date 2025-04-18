import numpy as np
import scipy.sparse as sp


def jor_solver(A, b, tol=1e-6, x0=None, nmax=10000, omega=0.5):
    """
    Metodo JOR (Jacobi Over-Relaxation) per risolvere Ax = b.
    Supporta sia matrici dense che sparse.
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    A_sparse = sp.csr_matrix(A) if not sp.issparse(A) else A

    D = A_sparse.diagonal()
    R = A_sparse - sp.diags(D)

    nit = 0
    x_new = np.ones(b)

    for _ in range(nmax):
        x_new = (b - R @ x) / D
        x_new = omega * x_new + (1 - omega) * x
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        x = x_new
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x_new) / np.linalg.norm(x_new)

    return x_new, nit, err

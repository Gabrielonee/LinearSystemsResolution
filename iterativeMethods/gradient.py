import numpy as np
import scipy.sparse as sp


def gradient_sparse(A_sparse, b, x0=None, tol=1e-6, nmax=1000):
    if not sp.isspmatrix(A_sparse):
        raise ValueError("A must be a sparse matrix")

    # Verifica simmetria
    if not (A_sparse != A_sparse.T).nnz == 0:
        raise ValueError("Matrix A must be symmetric")

    # Inizializza vettore x0
    if x0 is None:
        x0 = np.zeros_like(b)
    r = b - A_sparse @ x0
    nit = 0

    while np.linalg.norm(r) > tol and nit < nmax:
        Ar = A_sparse @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x0 = x0 + alpha * r
        r = b - A_sparse @ x0
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x0) / np.linalg.norm(x0)
    return x0, nit, err

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utilities import iterativeResult as IR


def gauss_seidel_solver(A, b, x0, tol, nmax):
    """
    Metodo di Gauss-Seidel per risolvere Ax = b.
    Supporta sia matrici dense che sparse.
    """

    A_sparse = sp.csr_matrix(A) if not sp.issparse(A) else A

    # Estrai parte triangolare inferiore di A (inclusa diagonale)
    L = sp.tril(A_sparse)  # Lower triangular + diagonal
    R = A_sparse - L       # Parte superiore
    L = L.tocsc()

    nit = 0
    x_new = np.ones(b)

    for _ in range(nmax):
        rhs = b - R @ x0
        # Risoluzione del sistema L x_new = rhs
        if sp.issparse(L):
            x_new = spla.spsolve_triangular(L, rhs, lower=True)
        else:
            x_new = spla.spsolve_triangular(L, rhs, lower=True)

        if np.linalg.norm(x_new - x0, np.inf) < tol:
            break

        x0 = x_new
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x_new) / np.linalg.norm(x_new)

    return IR.IterativeResult(x0, nit, err)

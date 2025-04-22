import numpy as np
import scipy.sparse as sp
from utilities import iterativeResult as IR


def jor_solver(A_sparse, b, x0, tol, nmax, omega=0.5):
    """
    Metodo JOR (Jacobi Over-Relaxation) per risolvere Ax = b.
    Supporta sia matrici dense che sparse.
    """

    D = A_sparse.diagonal()
    R = A_sparse - sp.diags(D)

    nit = 0
    x_new = np.ones(b)

    for _ in range(nmax):
        x_new = (b - R @ x0) / D
        x_new = omega * x_new + (1 - omega) * x0
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            break
        x0 = x_new
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x_new) / np.linalg.norm(x_new)

    return IR.IterativeResult(x0, nit, err)

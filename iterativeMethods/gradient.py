import numpy as np
from utilities import iterativeResult as IR


def gradient_sparse(A_sparse, b, x0, tol, nmax):
    r = b - A_sparse @ x0
    nit = 0

    while np.linalg.norm(r) > tol and nit < nmax:
        Ar = A_sparse @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x0 = x0 + alpha * r
        r = b - A_sparse @ x0
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x0) / np.linalg.norm(x0)

    return IR.IterativeResult(x0, nit, err)

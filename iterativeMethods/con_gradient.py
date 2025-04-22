import numpy as np


def con_gradient_sparse(A_sparse, b, x0=None, tol=1e-6, nmax=1000):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - A_sparse @ x
    p = r.copy()
    nit = 0
    norm_b = np.linalg.norm(b)
    err = np.linalg.norm(r) / norm_b if norm_b != 0 else np.linalg.norm(r)

    while nit < nmax and err > tol:
        Ap = A_sparse @ p
        alpha = np.dot(r, r)
        denom = np.dot(p, Ap)
        if denom == 0:
            break  # evita divisione per zero
        alpha /= denom

        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        nit += 1
        err = np.linalg.norm(r) / np.linalg.norm(r)

    return x, nit, err

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gauss_seidel_solver(A, b, x0=None, tol=1e-6, nmax=10000):
    """
    Metodo di Gauss-Seidel per risolvere Ax = b.
    Supporta sia matrici dense che sparse.
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    A_sparse = sp.csr_matrix(A) if not sp.issparse(A) else A

    # Estrai parte triangolare inferiore di A (inclusa diagonale)
    L = sp.tril(A_sparse)  # Lower triangular + diagonal
    R = A_sparse - L       # Parte superiore
    L = L.tocsc()

    nit = 0
    x_new = np.ones(b)

    for _ in range(nmax):
        rhs = b - R @ x
        # Risoluzione del sistema L x_new = rhs
        if sp.issparse(L):
            x_new = spla.spsolve_triangular(L, rhs, lower=True)
        else:
            x_new = spla.spsolve_triangular(L, rhs, lower=True)

        if np.linalg.norm(x_new - x, np.inf) < tol:
            break

        x = x_new
        nit += 1

    err = np.linalg.norm(b - A_sparse @ x_new) / np.linalg.norm(x_new)

    return x_new, nit,  err


A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 2, 3])
x0 = np.zeros_like(b)

x, nit, err = gauss_seidel_solver(A, b)

print(f"Solution: {x}")
print(f"Iteration: {nit}")
print(f"Relative residual error: {err:.2e}")

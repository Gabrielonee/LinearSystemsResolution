import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utilities.classes import IterativeResult

def gauss_seidel_solver_library(A_sparse, b, x0, tol: float, nmax: int):
    """
    Risolve il sistema lineare Ax = b usando il metodo iterativo di
    Gauss-Seidel per matrici sparse.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Matrice dei coefficienti A in formato sparso, preferibilmente quadrata
        e a diagonale dominante.
    b : ndarray
        Vettore dei termini noti.
    x0 : ndarray
        Vettore iniziale della soluzione.
    tol : float
        Tolleranza per il criterio di arresto basato sulla norma infinito
        della differenza tra iterazioni successive.
    nmax : int
        Numero massimo di iterazioni consentite.

    Returns
    -------
    IterativeResult
        Oggetto contenente:
            - x : soluzione approssimata del sistema.
            - nit : numero di iterazioni eseguite (1-based).

    Note
    ----
    La matrice A dovrebbe essere almeno debolmente diagonalmente dominante per
    garantire convergenza.
    Il metodo usa `spsolve_triangular` per risolvere il sistema triangolare
    inferiore in modo efficiente.
    """
    # Same factorization as Jacobi
    # Extract lower triangular part of A (including diagonal)
    L = sp.tril(A_sparse)  # Lower triangular + diagonal
    N = A_sparse - L       # Upper triangular (excluding diagonal)

    # Convert L to CSC format for efficient triangular solves
    L = L.tocsc()

    # Initialize iteration counter
    nit = 0
    # Initialize solution vector
    x_new = x0.copy()
    # Iterative solution
    for nit in range(nmax):
        # Compute right-hand side for the current iteration
        rhs = b - N @ x_new
        # Solve the lower triangular system
        x_new = spla.spsolve_triangular(L, rhs, lower = True)
        # Check convergence
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            break
        # Update solution for next iteration
        x0 = x_new.copy()

    # Return result
    return IterativeResult(x0, nit + 1)


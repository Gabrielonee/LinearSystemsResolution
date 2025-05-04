import scipy.sparse as sp
import numpy as np
from utilities.classes import IterativeResult


def jacobi_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Risolve il sistema lineare Ax = b usando il metodo iterativo di Jacobi.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Matrice dei coefficienti A in formato sparso. Deve essere quadrata e
        preferibilmente a diagonale dominante.
    b : ndarray
        Vettore dei termini noti.
    x0 : ndarray
        Vettore iniziale della soluzione.
    tol : float
        Tolleranza per il criterio di arresto basato sulla norma infinito della
        differenza tra iterazioni successive.
    nmax : int
        Numero massimo di iterazioni permesse.

    Returns
    -------
    IterativeResult
        Oggetto contenente:
            - x : soluzione approssimata del sistema.
            - nit : numero di iterazioni eseguite (1-based).

    Raises
    ------
    ValueError
        Se la matrice ha zeri (o quasi zeri) sulla diagonale principale,
        il metodo di Jacobi non è applicabile.

    Note
    ----
    Il metodo è semplice ma converge lentamente. È garantita la convergenza
    solo se A è diagonalmente dominante o simmetrica definita positiva.
    """
    # Estrai gli elementi diagonali di A (matrice P)
    P = A_sparse.diagonal()

    # Verifica zeri nella diagonale
    if np.any(np.abs(P) < 1e-10):
        raise ValueError(
            "Matrix has zeros on the diagonal"
            "- Jacobi method cannot be applied"
        )

    # Calcola N = A - D (L + U)
    N = A_sparse - sp.diags(P)

    nit = 0
    x_new = x0.copy()

    for nit in range(nmax):
        x_new = (b - N @ x0) / P
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            x0 = x_new.copy()
            break
        x0 = x_new.copy()

    return IterativeResult(x0, nit + 1)

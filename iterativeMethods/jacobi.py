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
    D = A_sparse.diagonal()
    if np.any(np.abs(D) < 1e-10):
        raise ValueError("Matrice con zeri (o quasi) sulla diagonale: Jacobi non applicabile.")

    D_inv = 1.0 / D
    x = x0.copy()
    norm_b = np.linalg.norm(b)
    
    for nit in range(nmax):
        res = b - A_sparse @ x
        res_rel = np.linalg.norm(res) / norm_b
        if res_rel < tol:
            break
        x = x + D_inv * res
    else:
        print("Metodo non converge entro il numero massimo di iterazioni.")
    return IterativeResult(x, nit + 1)

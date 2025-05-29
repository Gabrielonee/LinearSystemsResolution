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
    # Estrai gli elementi diagonali di A (matrice D)
    D = A_sparse.diagonal()
    # Controlla se ci sono zeri (o quasi zeri) sulla diagonale
    if np.any(np.abs(D) < 1e-10):
        raise ValueError("Matrice con zeri (o quasi) sulla diagonale: Jacobi non applicabile.")

    #Calcolo dell'inversa di ogni elemento della diagonale
    D_inv = 1.0 / D
    #Inizalizzazione del vettore x con una copia del vettore iniziale
    x = x0.copy()
    #Calcolo della norma 2 del vettore b
    norm_b = np.linalg.norm(b)
    
    for nit in range(nmax):
        #Calcolo del residuo --> quanto la soluzione corrente x non soddisfa ancora il sistema
        res = b - A_sparse @ x
        #Calcolo del residuo relativo
        res_rel = np.linalg.norm(res) / norm_b
        if res_rel < tol:
            break
        #Aggiornamento della soluzione corrente x secondo il metodo di Jacobi: D^-1(b-Ax) = D^-1 * res
        x = x + D_inv * res
    else:
        print("Metodo non converge entro il numero massimo di iterazioni.")
    # Restituisce il risultato come oggetto IterativeResult
    return IterativeResult(x, nit + 1)

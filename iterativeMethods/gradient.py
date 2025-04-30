import numpy as np
from utilities.classes import IterativeResult


def gradient_solver(A_sparse, b, x0, tol, nmax):
    """
    Risolve il sistema lineare Ax = b con il metodo del gradiente
    (steepest descent), adatto per matrici simmetriche definite positive.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix or ndarray
        Matrice dei coefficienti A,
        preferibilmente simmetrica definita positiva.
    b : ndarray
        Vettore dei termini noti.
    x0 : ndarray
        Vettore iniziale della soluzione.
    tol : float
        Tolleranza per il criterio di arresto,
        applicata alla norma del residuo relativo.
    nmax : int
        Numero massimo di iterazioni consentite.

    Returns
    -------
    IterativeResult
        Oggetto contenente:
            - x : soluzione approssimata.
            - nit : numero di iterazioni eseguite.

    Note
    ----
    Il metodo del gradiente semplice converge più lentamente rispetto al
    gradiente coniugato, ma è utile per scopi didattici o come baseline.
    L'arresto si basa sulla norma del residuo: ||r_k|| <= tol * ||b||.
    """
    # Calcolo residuo iniziale
    r = b - A_sparse @ x0

    # Norme iniziali
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    rel_tol = tol * norm_b if norm_b > 0 else tol

    nit = 0

    while norm_r > rel_tol and nit < nmax:
        Ar = A_sparse @ r
        r_dot_r = np.dot(r, r)
        r_dot_Ar = np.dot(r, Ar)

        if abs(r_dot_Ar) < 1e-14:
            break  # evita divisione per zero

        alpha = r_dot_r / r_dot_Ar
        x0 = x0 + alpha * r

        r = b - A_sparse @ x0
        norm_r = np.linalg.norm(r)

        nit += 1

    return IterativeResult(x0, nit)

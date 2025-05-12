import numpy as np
import scipy.sparse as sp

def solve_lower_triangular(L_sparse, b):
    """
    Risolve L x = b dove L è triangolare inferiore sparsa (formato CSC o CSR).
    Usa forward substitution esplicita su formato sparse.

    Parametri:
    -----------
    L_sparse : scipy.sparse.spmatrix
        Matrice triangolare inferiore sparsa (idealmente in formato CSC).
    b : numpy.ndarray
        Vettore termine noto.

    Ritorna:
    --------
    x : numpy.ndarray
        Vettore soluzione.
    """
    if not sp.isspmatrix_csc(L_sparse):
        L = L_sparse.tocsc()
    else:
        L = L_sparse

    n = L.shape[0]
    x = np.zeros_like(b, dtype=float)

    for i in range(n):
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]
        sum_ax = 0.0
        diag = None
        for idx in range(row_start, row_end):
            j = L.indices[idx]
            val = L.data[idx]
            if j < i:
                sum_ax += val * x[j]
            elif j == i:
                diag = val
        if diag is None or abs(diag) < 1e-14:
            raise ValueError(f"Elemento diagonale nullo in riga {i}")
        x[i] = (b[i] - sum_ax) / diag
    return x

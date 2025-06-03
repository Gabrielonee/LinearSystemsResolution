
import numpy as np
import scipy.sparse as sp

def triang_inf(L_sparse, b):
    """
    Risolve il sistema triangolare inferiore L x = b via forward substitution,
    ottimizzato per matrici sparse.

    Parametri
    ---------
    L_sparse : scipy.sparse matrix
        Matrice triangolare inferiore sparsa, con diagonale non nulla.
    b : (n,) numpy.ndarray
        Vettore termine noto.

    Ritorna
    -------
    x : (n,) numpy.ndarray
        Soluzione del sistema.
    """
    
    if not sp.issparse(L_sparse):
        raise ValueError("La matrice L deve essere in formato sparso")
    n = L_sparse.shape[0]
    if not sp.isspmatrix_csr(L_sparse):
        L_sparse = L_sparse.tocsr()   
    diag = L_sparse.diagonal()
    if np.any(np.abs(diag) < 1e-14):
        raise ValueError("La matrice L ha elementi diagonali nulli o prossimi a zero")
    x = np.zeros_like(b)

    for i in range(n):
        row = L_sparse[i, :i]  #Ottiene gli elementi della riga i fino alla colonna i-1
        x[i] = (b[i] - row.dot(x[:i])) / diag[i]

    return x
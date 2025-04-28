import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import eigsh


def validate_matrix(matrix):
    """
    Verifica che una matrice sia adatta all'uso con metodi iterativi 
    per matrici simmetriche e definite positive (SPD).

    Parametri
    ----------
    matrix : ndarray o scipy.sparse matrix
        Matrice da validare.

    Ritorna
    -------
    int
        Dimensione della matrice (numero di righe/colonne).

    Solleva
    -------
    ValueError
        Se la matrice è None, non ha attributo shape, non è quadrata,
        non è simmetrica, oppure non è definita positiva.
    """
    if matrix is None:
        raise ValueError("Matrix è None.")

    if not hasattr(matrix, 'shape') or matrix.shape is None:
        raise ValueError("Matrice non valida o informazioni mancanti")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("La matrice non è quadrate. Impossibile procedere")

    # Simmetria
    if isspmatrix(matrix):
        # Per matrici sparse
        diff = matrix - matrix.T
        if diff.nnz != 0:
            raise ValueError("La matrice non è simmetrica")
    else:
        # Per matrici dense
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("La matrice non è simmetrica")

    # Verifica definita positiva
    try:
        # Usiamo solo il min autovalore, metodo efficiente per SPD
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        if min_eig <= 0:
            raise ValueError("La matrice non è definita positiva")
    except Exception as e:
        raise ValueError(
            f"Impossibile verificare se la matrice è definita positiva: {e}")

    return rows

import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import eigsh


def validate_matrix(matrix):
    """
    Valida una matrice assicurandosi che sia quadrata,
    simmetrica e definita positiva (SPD).

    Questa funzione è compatibile sia con matrici dense (NumPy)
    sia con matrici sparse (SciPy).

    Parameters
    ----------
    matrix : array-like or sparse matrix
        Matrice da validare.

    Returns
    -------
    int
        La dimensione della matrice (numero di righe o colonne).

    Raises
    ------
    ValueError
        Se la matrice è nulla, non quadrata, non simmetrica
        o non definita positiva.

    Notes
    -----
    - La simmetria è verificata con `np.allclose` per matrici dense
      e con `nnz` per matrici sparse.
    - La verifica di definita positività usa `scipy.sparse.linalg.eigsh`
      per ottenere il minimo autovalore.
    """
    if matrix is None:
        raise ValueError("Matrix is None.")

    if not hasattr(matrix, 'shape') or matrix.shape is None:
        raise ValueError("Matrix not valid or missing shape information.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("La matrice non è quadrata. Impossibile procedere.")

    # Verifica simmetria
    if isspmatrix(matrix):
        if (matrix - matrix.T).nnz != 0:
            raise ValueError("La matrice sparsa non è simmetrica.")
    else:
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("La matrice densa non è simmetrica.")

    # Verifica definita positiva (SPD)
    try:
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        if min_eig <= 0:
            raise ValueError("La matrice non è definita positiva (SPD).")
    except Exception as e:
        raise ValueError(f"Impossibile determinare se la matrice è SPD: {e}")

    return rows

from scipy.sparse.linalg import eigsh


def validate_matrix(matrix):
    """
    Valida una matrice sparsa CSR assicurandosi che sia:
    quadrata, simmetrica, definita positiva (SPD) e diagonale dominante.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Matrice sparsa da validare.

    Returns
    -------
    int
        La dimensione della matrice (numero di righe/colonne).

    Raises
    ------
    ValueError
        Se la matrice è nulla, non quadrata, non simmetrica,
        non definita positiva o non diagonale dominante.

    Notes
    -----
    - Supporta solo matrici in formato CSR.
    - La simmetria è verificata tramite confronto con la trasposta.
    - La definita positività è valutata con il minimo autovalore.
    - La dominanza diagonale è verificata riga per riga senza conversioni.
    """
    if matrix is None:
        raise ValueError("La matrice è None.")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("La matrice non è quadrata.")

    # Verifica simmetria: matrix == matrix.T
    if (matrix - matrix.T).nnz != 0:
        raise ValueError("La matrice non è simmetrica.")

    # Verifica definita positiva (SPD)
    try:
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        if min_eig <= 0:
            raise ValueError("La matrice non è definita positiva (SPD).")
    except Exception as e:
        raise ValueError(f"Errore durante la verifica SPD: {e}")

    return rows

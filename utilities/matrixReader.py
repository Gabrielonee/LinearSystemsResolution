import scipy.io

def MatrixReader(filePath=None, debug=False):
    """
    Legge una matrice da file nel formato Matrix Market (.mtx),
    la converte in formato CSR e stampa informazioni utili.

    Parameters
    ----------
    filePath : str
        Percorso del file .mtx da leggere.
    debug : bool
        Se True, stampa le informazioni sulla matrice letta.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Matrice letta e convertita in formato CSR.

    Raises
    ------
    ValueError
        Se il percorso non è fornito o è `None`.
    IOError
        Se si verifica un errore durante la lettura del file.

    Notes
    -----
    Utilizza `scipy.io.mmread`, che restituisce inizialmente una matrice COO,
    poi convertita in CSR, ideale per metodi iterativi e
    prodotti matrice-vettore.
    """
    if filePath is None:
        raise ValueError("Percorso non valido: `filePath` è None.")

    try:
        A = scipy.io.mmread(filePath)

        shape = A.shape
        n_elements = shape[0] * shape[1]
        n_nonzeros = A.nnz
        sparsity = 100 * (1 - n_nonzeros / n_elements)

        if debug:
            print(f"Nome file: {filePath}")
            print(f"Dimensione matrice: {shape[0]} x {shape[1]}")
            print(f"Elementi non nulli: {n_nonzeros}")
            print(f"Sparsità: {sparsity:.2f}%")

        return A

    except Exception as e:
        raise IOError(f"Errore durante la lettura della matrice: {str(e)}")

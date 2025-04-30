import scipy.io


def MatrixReader(filePath=None):
    """
    Legge una matrice da file nel formato Matrix Market (.mtx).

    Parameters
    ----------
    filePath : str
        Percorso del file .mtx da leggere.

    Returns
    -------
    A : scipy sparse matrix
        Matrice letta dal file nel formato sparso di SciPy.

    Raises
    ------
    ValueError
        Se il percorso non è fornito o è `None`.
    IOError
        Se si verifica un errore durante la lettura del file.

    Notes
    -----
    Utilizza `scipy.io.mmread`, che restituisce una matrice sparsa
    (`scipy.sparse`), ideale per problemi numerici su larga scala.
    """
    if filePath is None:
        raise ValueError("Percorso non valido: `filePath` è None.")

    try:
        A = scipy.io.mmread(filePath)
        return A
    except Exception as e:
        raise IOError(f"Errore durante la lettura della matrice: {str(e)}")

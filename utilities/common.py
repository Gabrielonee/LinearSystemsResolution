import scipy.io


def MatrixReader(filePath=None):
    """
    Legge una matrice da un file in formato Matrix Market (.mtx).

    Parametri
    ----------
    filePath : str
        Percorso del file contenente la matrice da leggere.

    Ritorna
    -------
    scipy.sparse.csr_matrix o numpy.ndarray
        La matrice letta dal file.

    Solleva
    -------
    ValueError
        Se il percorso del file non è stato specificato.

    IOError
        Se si verificano errori durante la lettura del file.
    """
    if filePath is None:
        raise ValueError(
            "Percorso del file non specificato. Fornire un path valido.")

    try:
        # Lettura della matrice in formato Matrix Market
        A = scipy.io.mmread(filePath)
        return A

    except Exception as e:
        raise IOError(f"Errore nella lettura del file della matrice: {str(e)}")

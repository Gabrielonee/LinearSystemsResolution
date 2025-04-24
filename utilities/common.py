import scipy.io


def MatrixReader(filePath=None):
    """
    Reads a matrix from a Matrix Market (.mtx) file.

    Parameters:
    ----------
    filePath : str
    Path to the file containing the matrix to read

    Returns:
    --------
    scipy.sparse.csr_matrix or numpy.ndarray
    The matrix read from the file

    Raises:
    -------
    ValueError
    If the file path is not specified
    IOError
    If errors occur while reading the file
    """
    if filePath is None:
        raise ValueError("Path file not provided! Use a valid path")

    try:
        # Lettura matrice in formato Matrix Market
        A = scipy.io.mmread(filePath)

        return A

    except Exception as e:
        raise IOError(f"Error reading matrix file: {str(e)}")

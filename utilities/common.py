import numpy as np
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


def verify_accuracy(x_computed, x_true):
    """
    Calculates the relative error between a computed solution
    and the exact solution.

    Parameters:
    -----------
    x_computed : numpy.ndarray
    The computed solution
    x_true : numpy.ndarray
    The exact solution

    Returns:
    --------
    float
    The relative error in the infinity norm
    """
    # Verifica che i vettori abbiano la stessa dimensione
    if x_computed.shape != x_true.shape:
        raise ValueError(f"Dimension mismatch: x_computed has shape {
                         x_computed.shape}, x_true has shape {x_true.shape}")

    # Calcolo dell'errore relativo nella norma infinito
    norm_true = np.linalg.norm(x_true, ord=np.inf)

    # Evita divisione per zero
    if norm_true < 1e-14:
        return np.linalg.norm(x_computed - x_true, ord=np.inf)

    return np.linalg.norm(x_computed - x_true, ord=np.inf) / norm_true

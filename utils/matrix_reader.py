import scipy.io
from scipy.sparse import csr_matrix


def MatrixReader(file_path: str = None, debug: bool = False) -> csr_matrix:
    """
    Reads a matrix from a Matrix Market (.mtx) file and converts it to CSR format.

    Parameters
    ----------
    file_path : str
        Path to the .mtx file.
    debug : bool, optional
        If True, prints matrix information (default: False).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The matrix converted to CSR format.

    Raises
    ------
    ValueError
        If no file path is provided.
    IOError
        If the file cannot be read or parsed.

    Notes
    -----
    Uses `scipy.io.mmread`, which returns a COO matrix.
    This is converted to CSR format, which is optimal for
    iterative solvers and matrix-vector operations.
    """
    if file_path is None:
        raise ValueError("Invalid input: `file_path` is None.")

    try:
        A = scipy.io.mmread(file_path).tocsr()  # Convert explicitly to CSR

        if debug:
            shape = A.shape
            n_elements = shape[0] * shape[1]
            n_nonzeros = A.nnz
            sparsity = 100 * (1 - n_nonzeros / n_elements)

            print(f"[MatrixReader] File: {file_path}")
            print(f"[MatrixReader] Shape: {shape[0]} x {shape[1]}")
            print(f"[MatrixReader] Non-zero entries: {n_nonzeros}")
            print(f"[MatrixReader] Sparsity: {sparsity:.2f}%")

        return A

    except Exception as e:
        raise IOError(f"Error while reading matrix: {str(e)}")
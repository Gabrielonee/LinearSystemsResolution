import scipy.io

def MatrixReader(filePath=None):
    if filePath is None:
        raise ValueError(
            "Unvalid path.")
    try:
        # Reading matrix
        A = scipy.io.mmread(filePath)
        return A
    except Exception as e:
        raise IOError(f"Error reading matrix: {str(e)}")

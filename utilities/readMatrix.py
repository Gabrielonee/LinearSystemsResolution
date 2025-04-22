import numpy as np
import scipy.io


def MatrixReader(filePath=None):
    if filePath is None:
        raise ValueError("Path file not provided! Use a valid path")

    A = scipy.io.mmread(filePath)  # Lettura matrice sparse
    return A


# Funzione per verificare l'accuratezza
def verify_accuracy(x_computed, x_true):
    error = np.linalg.norm(x_computed - x_true, ord=np.inf)  # Calcola l'errore
    return error

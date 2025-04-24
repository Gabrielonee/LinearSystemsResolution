import numpy as np


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

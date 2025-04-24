import numpy as np


def verify_accuracy(x_computed, x_true):
    """
    Calcola l'errore relativo (in norma infinito) tra la soluzione
    computata e la soluzione esatta.

    Parametri
    ----------
    x_computed : numpy.ndarray
        Soluzione calcolata dal metodo iterativo.

    x_true : numpy.ndarray
        Soluzione esatta del sistema.

    Ritorna
    -------
    float
        Errore relativo nella norma infinito.

    Solleva
    -------
    ValueError
        Se le dimensioni di x_computed e x_true non corrispondono.
    """
    if x_computed.shape != x_true.shape:
        raise ValueError(
            f"Mismatch delle dimensioni: x_computed ha una forma di: {
                x_computed.shape}, "
            f"mentre x_true di: {x_true.shape}"
        )

    norm_true = np.linalg.norm(x_true, ord=np.inf)

    if norm_true < 1e-14:
        # Caso borderline: norma della soluzione vera ≈ 0
        return np.linalg.norm(x_computed - x_true, ord=np.inf)

    rel_error = np.linalg.norm(x_computed - x_true, ord=np.inf) / norm_true
    return rel_error

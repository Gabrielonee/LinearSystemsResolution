import numpy as np
from numpy import ndarray


def verify_accuracy(x_computed: ndarray, x_true: ndarray) -> float:
    """
    Compute the relative L2-norm error between a computed solution and the reference.

    Parameters
    ----------
    x_computed : np.ndarray
        Computed solution vector from a numerical algorithm.
    x_true : np.ndarray
        Reference or true solution vector.

    Returns
    -------
    float
        Relative error in the L2 norm. If the norm of `x_true` is very small (< 1e-14),
        returns the absolute error instead.

    Raises
    ------
    ValueError
        If the two vectors do not have matching shapes.

    Notes
    -----
    Relative Error:
        ||x_computed - x_true||_2 / ||x_true||_2
    Falls back to absolute error if ||x_true||_2 < 1e-14.
    """
    if not isinstance(x_computed, np.ndarray) or not isinstance(x_true, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays.")

    if x_computed.shape != x_true.shape:
        raise ValueError(
            f"Shape mismatch: x_computed {x_computed.shape} vs x_true {x_true.shape}"
        )

    diff = x_computed - x_true
    norm_diff = np.linalg.norm(diff, ord=2)
    norm_true = np.linalg.norm(x_true, ord=2)

    if norm_true < 1e-14:
        return norm_diff

    return norm_diff / norm_true
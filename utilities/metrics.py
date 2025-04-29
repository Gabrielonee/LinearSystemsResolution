import numpy as np


def verify_accuracy(x_computed, x_true):
    #relative error (infinite norm) between solution computated and the real one
    if x_computed.shape != x_true.shape:
        raise ValueError(
            f"Mismatch of sizes, x_computed: {
                x_computed.shape}, "
            f"x_true: {x_true.shape}"
        )
    norm_true = np.linalg.norm(x_true, ord=np.inf)
    if norm_true < 1e-14:
        return np.linalg.norm(x_computed - x_true, ord=np.inf)
    rel_error = np.linalg.norm(x_computed - x_true, ord=np.inf) / norm_true
    return rel_error

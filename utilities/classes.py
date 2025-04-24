from dataclasses import dataclass
import numpy as np


@dataclass
class IterativeResult:
    solution: np.ndarray
    iterations: int


@dataclass
class SolverResult:
    method_name: str
    tol: float
    max_iterations: int
    method_result: IterativeResult
    rel_error: np.floating
    time_seconds: float
    memory_kb: float

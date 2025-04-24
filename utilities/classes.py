from dataclasses import dataclass
import numpy as np


@dataclass
class IterativeResult:
    solution: np.ndarray
    iterations: int


@dataclass
class SolverResult:
    result: IterativeResult
    time_seconds: float
    memory_kb: float

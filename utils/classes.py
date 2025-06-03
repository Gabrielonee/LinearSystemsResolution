from dataclasses import dataclass
import numpy as np


@dataclass
class IterativeResult:
    solution: np.ndarray
    iterations: int
    converged: bool


@dataclass
class SolverResult:
    method_name: str
    tol: float
    max_iterations: int
    method_result: "IterativeResult"
    rel_error: np.floating
    time_seconds: float
    memory_kb: float

    def __str__(self):
        if not self.method_result.converged:
            return (
                f"Method: {self.method_name}\n"
                f"Tolerance: {self.tol:.1e}, Max Iterations: {self.max_iterations}\n"
                f"Iterations Performed: {self.method_result.iterations}\n"
                f"Convergence Failed, Relative Error: {self.rel_error:.2e}\n"
                f"Time: {self.time_seconds:.4f} s, Memory: {self.memory_kb:.2f} KB\n"
            )
        else:
            return (
                f"Method: {self.method_name}\n"
                f"Tolerance: {self.tol:.1e}, Max Iterations: {
                    self.max_iterations}\n"
                f"Iterations Performed: {self.method_result.iterations}\n"
                f"Relative Error: {self.rel_error:.2e}\n"
                f"Time: {self.time_seconds:.4f} s, Memory: {
                    self.memory_kb:.2f} KB\n"
            )

    def to_dict(self):
        if self.method_result.converged:
            return {
                "method": self.method_name,
                "tolerance": self.tol,
                "max_iterations": self.max_iterations,
                "performed_iterations": self.method_result.iterations,
                "rel_error": float(self.rel_error),
                "time_seconds": self.time_seconds,
                "memory_kb": self.memory_kb,
            }
        else:
            return {}

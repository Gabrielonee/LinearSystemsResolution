from dataclasses import dataclass
import numpy as np


@dataclass
class IterativeResult:
    """
    Stores the output of an iterative solver.
    """
    solution: np.ndarray       # Final solution vector
    iterations: int            # Number of iterations performed
    converged: bool            # Whether the solver met the convergence criteria


@dataclass
class SolverResult:
    """
    Stores metadata and results for a specific solver execution.
    """
    method_name: str           # Name of the solver used
    tol: float                 # Convergence tolerance
    max_iterations: int        # Max number of iterations allowed
    method_result: "IterativeResult"  # Output of the solver
    rel_error: np.floating     # Relative error vs ground truth
    time_seconds: float        # Execution time in seconds
    memory_kb: float           # Peak memory usage in kilobytes

    def __str__(self):
        """
        Custom string representation of the result, for human-readable output.
        """
        base_info = (
            f"Method: {self.method_name}\n"
            f"Tolerance: {self.tol:.1e}, Max Iterations: {self.max_iterations}\n"
            f"Iterations Performed: {self.method_result.iterations}\n"
            f"Time: {self.time_seconds:.4f} s, Memory: {self.memory_kb:.2f} KB\n"
        )

        if not self.method_result.converged:
            return base_info + f"Convergence Failed, Relative Error: {self.rel_error:.2e}\n"
        else:
            return base_info + f"Relative Error: {self.rel_error:.2e}\n"

    def to_dict(self):
        """
        Converts the result to a dictionary (for JSON export).
        Returns an empty dict if the solver did not converge.
        """
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
import time
import tracemalloc


def profile_solver(solver_func, *args, **kwargs):
    """
    Runs a linear system solver while profiling execution time and peak memory usage.

    Parameters
    ----------
    solver_func : callable
        The solver function to profile (e.g., iterative method).
    *args : tuple
        Positional arguments to pass to solver_func.
    **kwargs : dict
        Keyword arguments to pass to solver_func.

    Returns
    -------
    tuple
        (result, elapsed_time, peak_memory_kb):
        - result: output returned by solver_func, or None if an exception occurs
        - elapsed_time: float, total execution time in seconds
        - peak_memory_kb: float, peak memory usage during execution in kilobytes

    Notes
    -----
    - Uses tracemalloc to track Python heap memory peak usage.
    - Uses time.perf_counter() for high-resolution timing.
    - The function guarantees timing and memory measurement even if solver_func raises an exception.
    """
    tracemalloc.start()
    start_time = time.perf_counter()
    result = None

    try:
        # Call the solver function with all given arguments
        result = solver_func(*args, **kwargs)
    finally:
        # Measure elapsed time regardless of exceptions
        elapsed_time = time.perf_counter() - start_time
        # Get current and peak memory usage, then stop tracing
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # Convert peak memory from bytes to kilobytes
    peak_memory_kb = peak_memory / 1024
    return result, elapsed_time, peak_memory_kb
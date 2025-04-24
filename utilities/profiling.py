def profile_solver(solver_func, *args, **kwargs):
    import tracemalloc
    import time

    tracemalloc.start()
    start_time = time.perf_counter()
    result = solver_func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_time, peak_memory / 1024  # in KB

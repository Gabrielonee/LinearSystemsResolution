import time
import tracemalloc


def profile_solver(solver_func, *args, **kwargs):
    #use a method to solve a linear system computing time and memory 
    
    # Avvia tracciamento memoria e tempo
    tracemalloc.start()
    start_time = time.perf_counter()

    # Esecuzione del solver
    result = solver_func(*args, **kwargs)

    # Tempo totale e memoria usata
    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed_time, peak_memory / 1024  # Convertito in KB

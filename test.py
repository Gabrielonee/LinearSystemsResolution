import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import
from directMethods.factorization import factorization
from directMethods.LU_Pivoting import factorization_LU_pivoting
from directMethods.triang_inf import triang_inf
from iterativeMethods.jacobi import jacobi
from iterativeMethods.gauss_seidl import gauss_seidl
from iterativeMethods.gradient import gradient
from iterativeMethods.con_gradient import con_gradient
from iterativeMethods.jor import jor

def direct_solve_LU(A, b):
    try:
        L, U = factorization(A)
        y = triang_inf(L, b)
        x = np.linalg.solve(U, y)  #Solve upper triangular
        return x, 0, 0, np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    except ValueError as e:
        print(f"Error in LU factorization: {e}")
        return None, None, None, None

def direct_solve_LU_pivot(A, b):
    try:
        P, L, U = factorization_LU_pivoting(A)
        #Solve Ly = Pb
        y = triang_inf(L, P @ b)
        #Solve Ux = y
        x = np.linalg.solve(U, y)  #Solve upper triangular
        return x, 0, 0, np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    except Exception as e:
        print(f"Error in LU pivoting: {e}")
        return None, None, None, None

# DA TOGLIERE PER INSERIRE MATRICI DI TEST
def generate_test_matrices(sizes):
    matrices = {}
    for n in sizes:
        #Diagonally dominant tridiagonal matrix (well-conditioned)
        diag = np.ones(n) * 4
        off_diag = np.ones(n-1) * -1
        A_tridiag = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        matrices[f"Tridiagonal_{n}"] = (A_tridiag, f"Tridiagonal Matrix (n={n})")
        #Random SPD matrix (medium condition number)
        A_random = np.random.rand(n, n)
        A_random = A_random + A_random.T + n * np.eye(n)  # Make it SPD
        matrices[f"Random_SPD_{n}"] = (A_random, f"Random SPD Matrix (n={n})")        
    return matrices

def compare_methods(matrices, tolerances, max_iterations):
    results = []
    
    for matrix_name, (A, matrix_desc) in matrices.items():
        n = A.shape[0]
        condition_number = np.linalg.cond(A)
        b = np.ones(n)
        x0 = np.zeros(n)
        
        print(f"\nTesting on {matrix_desc}")
        print(f"Condition number: {condition_number:.2e}")
        
        for tol in tolerances:
            print(f"\n  Tolerance: {tol}")
            
            methods = {
                "LU Factorization": lambda: direct_solve_LU(A, b),
                "LU with Pivoting": lambda: direct_solve_LU_pivot(A, b),
                "Jacobi": lambda: jacobi(A, b, x0, tol, max_iterations),
                "Gauss-Seidel": lambda: gauss_seidl(A, b, x0, tol, max_iterations),
                "Gradient": lambda: gradient(A, b, x0, tol, max_iterations),
                "Conjugate Gradient": lambda: con_gradient(A, b, x0, tol, max_iterations),
                "JOR (ω=0.8)": lambda: jor(A, b, x0, tol, max_iterations, 0.8),
                "JOR (ω=1.2)": lambda: jor(A, b, x0, tol, max_iterations, 1.2)
            }
            
            for method_name, method_func in methods.items():
                try:
                    start_time = time.time()
                    x, iterations, method_time, error = method_func()
                    total_time = time.time() - start_time
                    if method_time == 0:
                        method_time = total_time
                    
                    if x is not None:
                        results.append({
                            "Matrix": matrix_desc,
                            "Size": n,
                            "Condition": condition_number,
                            "Method": method_name,
                            "Tolerance": tol,
                            "Iterations": iterations,
                            "Time": method_time,
                            "Error": error
                        })
                        
                        print(f"    {method_name}: {'converged' if error < tol * 10 else 'did not converge'}, "
                              f"iterations: {iterations}, time: {method_time:.6f}s, error: {error:.2e}")
                    else:
                        print(f"    {method_name}: failed to converge or not applicable")
                        
                except Exception as e:
                    print(f"    {method_name}: Error - {str(e)}")
    
    return results

def plot_results(results):
    #Convert results to numpy arrays for easier filtering
    methods = list(set([r["Method"] for r in results]))
    matrices = list(set([r["Matrix"] for r in results]))
    tolerances = list(set([r["Tolerance"] for r in results]))
    tolerances.sort()
    
    #Iterations vs Condition Number
    plt.figure(figsize=(12, 8))
    for method in methods:
        method_results = [r for r in results if r["Method"] == method and r["Tolerance"] == tolerances[0] and r["Iterations"] is not None]
        if method_results:
            conditions = [r["Condition"] for r in method_results]
            iterations = [r["Iterations"] for r in method_results]
            plt.scatter(conditions, iterations, label=method, alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Condition Number')
    plt.ylabel('Iterations')
    plt.title(f'Iterations vs Condition Number (tol={tolerances[0]})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('iterations_vs_condition.png')
    
    #Time vs Matrix size by method
    plt.figure(figsize=(12, 8))
    for method in methods:
        method_results = [r for r in results if r["Method"] == method and r["Tolerance"] == tolerances[0]]
        if method_results:
            sizes = [r["Size"] for r in method_results]
            times = [r["Time"] for r in method_results]
            #Size and average
            unique_sizes = list(set(sizes))
            unique_sizes.sort()
            avg_times = []
            for size in unique_sizes:
                avg_time = np.mean([r["Time"] for r in method_results if r["Size"] == size])
                avg_times.append(avg_time)
            plt.plot(unique_sizes, avg_times, 'o-', label=method)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Computation Time vs Matrix Size (tol={tolerances[0]})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('time_vs_size.png')
    
    #Error vs Tolerance for each method
    plt.figure(figsize=(12, 8))
    for method in methods:
        for matrix_type in matrices[:3]:
            method_matrix_results = [r for r in results if r["Method"] == method and r["Matrix"] == matrix_type]
            if method_matrix_results:
                tols = [r["Tolerance"] for r in method_matrix_results]
                errors = [r["Error"] for r in method_matrix_results]
                if len(tols) > 0:  #plot if we have data
                    plt.loglog(tols, errors, 'o-', label=f"{method} - {matrix_type.split(' ')[0]}")
    
    plt.xlabel('Tolerance')
    plt.ylabel('Final Error')
    plt.title('Error vs Tolerance for Different Methods and Matrices')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('error_vs_tolerance.png')
    
    return ["iterations_vs_condition.png", "time_vs_size.png", "error_vs_tolerance.png"]

def generate_report(results, plot_files):
    report = "# Comparison of Numerical Methods for Linear Systems\n\n"
    
    #Summary of methods tested
    methods = list(set([r["Method"] for r in results]))
    report += "## Methods Compared\n\n"
    report += "This report compares the following numerical methods for solving linear systems Ax = b:\n\n"
    for method in methods:
        report += f"- {method}\n"
    
    #Summary of matrices tested --> DA CAMBIARE: FILE MATRICI
    matrices = list(set([r["Matrix"] for r in results]))
    sizes = list(set([r["Size"] for r in results]))
    report += "\n## Test Matrices\n\n"
    report += "The methods were tested on the following matrix types:\n\n"
    for matrix in matrices:
        cond = np.mean([r["Condition"] for r in results if r["Matrix"] == matrix])
        report += f"- {matrix} (avg. condition number: {cond:.2e})\n"
    
    #Performance analysis
    report += "\n## Performance Analysis\n\n"
    
    #Convergence properties
    report += "### Convergence Properties\n\n"
    for matrix in matrices:
        report += f"#### {matrix}\n\n"
        report += "| Method | Tolerance | Iterations | Time (s) | Final Error |\n"
        report += "|--------|-----------|------------|----------|-------------|\n"
        
        for method in methods:
            for tol in sorted(set([r["Tolerance"] for r in results])):
                filtered = [r for r in results if r["Method"] == method and r["Matrix"] == matrix and r["Tolerance"] == tol]
                if filtered:
                    r = filtered[0]
                    iterations = r["Iterations"] if r["Iterations"] is not None else "N/A"
                    report += f"| {method} | {tol:.1e} | {iterations} | {r['Time']:.6f} | {r['Error']:.2e} |\n"
        
        report += "\n"
    
    #Effect of matrix condition number
    report += "### Effect of Matrix Condition Number\n\n"
    report += "For iterative methods, the number of iterations required typically increases with the condition number of the matrix.\n\n"
    
    #Computational efficiency
    report += "### Computational Efficiency\n\n"
    report += "Methods ranked by average computation time (fastest to slowest):\n\n"
    
    method_avg_times = {}
    for method in methods:
        times = [r["Time"] for r in results if r["Method"] == method]
        if times:
            method_avg_times[method] = np.mean(times)
    
    for method, avg_time in sorted(method_avg_times.items(), key=lambda x: x[1]):
        report += f"- {method}: {avg_time:.6f} seconds\n"
    
    #Accuracy comparison
    report += "\n### Accuracy Comparison\n\n"
    report += "Methods ranked by average error (most accurate to least accurate):\n\n"
    
    method_avg_errors = {}
    for method in methods:
        errors = [r["Error"] for r in results if r["Method"] == method]
        if errors:
            method_avg_errors[method] = np.mean(errors)
    
    for method, avg_error in sorted(method_avg_errors.items(), key=lambda x: x[1]):
        report += f"- {method}: {avg_error:.2e}\n"
    
    report += "\n## Conclusions\n\n"
    
    #Find best method for well-conditioned matrices
    well_cond = [r for r in results if "Tridiagonal" in r["Matrix"]]
    if well_cond:
        best_method = min(well_cond, key=lambda x: x["Time"])["Method"]
        report += f"- For well-conditioned matrices (like tridiagonal systems), {best_method} performs best in terms of speed.\n"
    
    #Find best method for ill-conditioned matrices
    ill_cond = [r for r in results if "Hilbert" in r["Matrix"]]
    if ill_cond:
        best_method = min(ill_cond, key=lambda x: x["Error"])["Method"]
        report += f"- For ill-conditioned matrices (like Hilbert matrices), {best_method} provides the most accurate results.\n"
    
    #General recommendations
    report += "- Direct methods (LU decomposition) are generally more reliable but can be slower for large systems.\n"
    report += "- Iterative methods can be more efficient for large, sparse systems but may struggle with ill-conditioned problems.\n"
    report += "- The Conjugate Gradient method typically converges faster than the standard Gradient method for SPD matrices.\n"
    report += "- The choice of relaxation parameter in JOR can significantly affect convergence.\n"
    
    #Add plots to the report
    report += "\n## Visual Analysis\n\n"
    
    for plot_file in plot_files:
        report += f"![{plot_file.split('.')[0].replace('_', ' ').title()}]({plot_file})\n\n"
    
    return report

def main():
    #TEST SU MATRICI IMPORTATE
    
    tolerances = [1e-6, 1e-8, 1e-10, 1e-14]  #Convergence tolerances
    max_iterations = 10000  #Maximum iterations for iterative methods
    
    #matrices = generate_test_matrices(sizes) 
    #results = compare_methods(matrices, tolerances, max_iterations)
    #plot_files = plot_results(results)
    #report = generate_report(results, plot_files)
    
    with open("numerical_methods_comparison.md", "w") as f:
        #f.write(report)
    
    #print("Comparison complete! Results saved to 'numerical_methods_comparison.md'")

#if __name__ == "__main__":
    #main()
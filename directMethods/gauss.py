import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, solve

def GaussMethod(A, b):
    return np.linalg.solve(A, b)

# Error,Cond vectors
err_vec = []
cond_vec = []

for n in range(5, 55, 5):
    A = hilbert(n)  
    b = np.ones(n)  

    #Random perturbation
    pert_factor = 1e-9
    pert_mat = (np.random.rand(n, n) - 0.5) * 2
    A_pert = A + pert_factor * pert_mat  
    pert_vec = (np.random.rand(n) - 0.5) * 2  
    b_pert = b + pert_factor * pert_vec  

    
    x = GaussMethod(A, b)
    x_pert = GaussMethod(A_pert, b_pert)

    
    err_vec.append(np.linalg.norm(x - x_pert) / np.linalg.norm(x))
    cond_vec.append(np.linalg.cond(A)) 


print("Relative errors:", err_vec)
print("Conditioning:", cond_vec)

plt.plot(range(5, 55, 5), err_vec, marker='o', label="Relative error")
plt.xlabel("System size")
plt.ylabel("Relative error")
plt.title("Perturbation effect on the system")
plt.legend()
plt.grid()
plt.show()
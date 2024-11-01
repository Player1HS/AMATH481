# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
# from scipy.sparse import diags
# from scipy.sparse.linalg import eigs

# def shoot(phi_primes, x, epsilon):
#     return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

# tol = 1e-4  # define a tolerance level 
# epsilon_0 = 0.1; initial = [1, 4]
# xspan =  np.arange(-4, 4.1, 0.1)

# eigenvals=[]
# A1 = np.zeros((len(xspan), 5))

# for modes in range(1, 6):  # begin mode loop
#     epsilon = epsilon_0  
#     depsilon = epsilon_0 / 100
#     for _ in range(1000):
#         phi = odeint(shoot, initial, xspan, args=(epsilon,)) 
#         boundcheck = phi[-1, 1] + np.sqrt(16 - epsilon)*phi[-1, 0]

#         if abs(phi[-1, 0]) < tol:  # check for convergence
#             break  # get out of convergence loop

#         if (-1) ** (modes + 1) * boundcheck > 0:
#             epsilon += depsilon
#         else:
#             epsilon -= depsilon / 2
#             depsilon /= 2
    
#     eigenvals.append(epsilon)
#     epsilon_0 = epsilon + 0.1  # after finding eigenvalue, pick new start
#     norm = np.trapz(phi[:, 0] * phi[:, 0], xspan)  # calculate the normalization
#     A1[:, modes - 1] = np.abs(phi[:, 0] / np.sqrt(norm))

# eigenvals = np.array(eigenvals)
# A2 = eigenvals

# system_dim = len(xspan)-2
# # Construct A
# e1 = np.ones(system_dim)
# multiplier = -1/(0.1**2)
# main_diag = multiplier*-2*e1
# main_diag += xspan[1:system_dim+1]**2
# diagonals = [multiplier*e1, main_diag, multiplier*e1]
# offsets = [-1, 0, 1]
# A = diags(diagonals, offsets, shape=(system_dim, system_dim), format='csr')

# # left boundary conditions
# A[0, 0] = 2/3
# A[0, 1] = -2/3

# # right boundary conditions
# A[system_dim-1, system_dim-1] = 2/3
# A[system_dim-1, system_dim-2] = -2/3

# A4, A3 = eigs(A, k=5, which='SR')
# print(A4)

import numpy as np
from scipy.sparse import diags
from scipy.integrate import odeint
from scipy.sparse.linalg import eigs
import scipy.sparse as sp

xspan =  np.arange(-4, 4.1, 0.1)
system_dim = len(xspan)

e1 = np.ones(system_dim)
multiplier = -1/(0.1**2)
main_diag = multiplier*-2*e1
main_diag += xspan**2
diagonals = [multiplier*e1, main_diag, multiplier*e1]
offsets = [-1, 0, 1]
A = diags(diagonals, offsets, shape=(system_dim, system_dim), format='csr')

# left boundary conditions
A[0, 0] = -3
A[0, 1] = 4
A[0, 2] = -1

# right boundary conditions
A[system_dim-1, system_dim-1] = -3
A[system_dim-1, system_dim-2] = 4
A[system_dim-1, system_dim-3] = -1

A4, A3 = eigs(A, k=10, which='SI')
print(A4)

system_dim = len(xspan)-2
# Construct A
e1 = np.ones(system_dim)
multiplier = -1/(0.1**2)
main_diag = multiplier*-2*e1
main_diag += xspan[1:system_dim+1]**2
diagonals = [multiplier*e1, main_diag, multiplier*e1]
offsets = [-1, 0, 1]
A = diags(diagonals, offsets, shape=(system_dim, system_dim), format='csr')

# left boundary conditions
A[0, 0] = 2/3
A[0, 1] = -2/3

# right boundary conditions
A[system_dim-1, system_dim-1] = 2/3
A[system_dim-1, system_dim-2] = -2/3

A4, A3 = eigs(A, k=10, which='SI')
print(A4)
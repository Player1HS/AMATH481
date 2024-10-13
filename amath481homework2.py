import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot(phi_primes, x, epsilon):
    return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

tol = 1e-6  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
epsilon_0 = 1; A = 1; initial = [0, A]
xspan =  np.arange(-4, 4, 0.1)

eigenvals=[]
eigenfuncs=[]

for modes in range(1, 6):  # begin mode loop
    epsilon = epsilon_0  
    depsilon = epsilon_0 / 100
    for _ in range(1000):
        phi = odeint(shoot, initial, xspan, args=(epsilon,)) 

        if abs(phi[-1, 0]) < tol:  # check for convergence
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * phi[-1, 0] > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    eigenvals.append(epsilon)
    epsilon_0 = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(phi[:, 0] * phi[:, 0], xspan)  # calculate the normalization
    plt.plot(xspan, phi[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
    eigenfuncs.append(phi[:, 0] / np.sqrt(norm))

eigenvals = np.array(eigenvals)
eigenfuncs = np.array(eigenfuncs)
A1 = np.transpose(eigenfuncs)
A2 = eigenvals
plt.show()
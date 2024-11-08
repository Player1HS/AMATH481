import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def shoot(x, phi_primes, epsilon):
    return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

tol = 1e-4  # define a tolerance level 
epsilon_0 = 0.1
xspan =  np.arange(-4, 4.1, 0.1)
col = ['r', 'b', 'g', 'c', 'm']

eigenvals=[]
A1 = np.zeros((len(xspan), 5))

for modes in range(1, 6):  # begin mode loop
    epsilon = epsilon_0  
    depsilon = epsilon_0 / 100
    for _ in range(1000):
        initial = [1, np.sqrt(16-epsilon)]
        phi = solve_ivp(shoot, (xspan[0], xspan[-1]), initial, args=(epsilon,), t_eval=xspan) 
        boundcheck = phi.y[1, -1] + np.sqrt(16 - epsilon)*phi.y[0, -1]

        if abs(boundcheck) < tol:  # check for convergence
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * boundcheck > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2
    
    eigenvals.append(epsilon)
    epsilon_0 = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(phi.y[0, :] * phi.y[0, :], xspan)  # calculate the normalization
    plt.plot(xspan, phi.y[0, :] / np.sqrt(norm), col[modes - 1])
    A1[:, modes - 1] = np.abs(phi.y[0, :] / np.sqrt(norm))

eigenvals = np.array(eigenvals)
A2 = eigenvals
plt.show()
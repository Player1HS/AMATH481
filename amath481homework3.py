import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

## PART A

def shoot(phi_primes, x, epsilon):
    return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

tol = 1e-4  # define a tolerance level 
epsilon_0 = 0.1; initial = [1, 4]
xspan =  np.arange(-4, 4.1, 0.1)

eigenvals=[]
A1 = np.zeros((len(xspan), 5))

for modes in range(1, 6):  # begin mode loop
    epsilon = epsilon_0  
    depsilon = epsilon_0 / 100
    for _ in range(1000):
        phi = odeint(shoot, initial, xspan, args=(epsilon,)) 
        boundcheck = phi[-1, 1] + np.sqrt(16 - epsilon)*phi[-1, 0]

        if abs(phi[-1, 0]) < tol:  # check for convergence
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * boundcheck > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2
    
    eigenvals.append(epsilon)
    epsilon_0 = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(phi[:, 0] * phi[:, 0], xspan)  # calculate the normalization
    A1[:, modes - 1] = np.abs(phi[:, 0] / np.sqrt(norm))

eigenvals = np.array(eigenvals)
A2 = eigenvals

#=======================================================================
## PART B

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
A[0, 0] = -2/3
A[0, 1] = 2/3

# right boundary conditions
A[system_dim-1, system_dim-1] = -2/3
A[system_dim-1, system_dim-2] = 2/3

A4, A3 = eigs(A, k=8, which='SI')
print(A4)

#=======================================================================
## PART C

def shoot(phi_primes, x, epsilon):
    return [phi_primes[1], (0.05*np.abs(phi_primes[0])**2+x**2 - epsilon) * phi_primes[0]]

tol = 1e-4  # define a tolerance level 
epsilon_0 = 1; initial = [0.1, 2]
xspan =  np.arange(-2, 2.1, 0.1)

eigenvals=[]
A5 = np.zeros((len(xspan), 2))

for modes in range(1, 3):  # begin mode loop
    epsilon = epsilon_0  
    depsilon = epsilon_0 / 100
    for _ in range(1000):
        phi = odeint(shoot, initial, xspan, args=(epsilon,))
        boundcheck = phi[-1, 1] + np.sqrt(0.05*np.abs(phi[-1, 0])**2+4 - epsilon)*phi[-1, 0]

        if abs(phi[-1, 0]) < tol:  # check for convergence
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * boundcheck > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2
    
    eigenvals.append(epsilon)
    epsilon_0 = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(phi[:, 0] * phi[:, 0], xspan)  # calculate the normalization
    A5[:, modes - 1] = np.abs(phi[:, 0] / np.sqrt(norm))

eigenvals = np.array(eigenvals)
A6 = eigenvals
print(A6)

#=======================================================================
## PART D

def hw1_rhs_a(x, phi_primes, epsilon):
    return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

E = 1; y0 = [1, np.sqrt(3)]
x_span =  (-2,2)

TOL = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
average_step_sizes_rk45 = []
average_step_sizes_rk23 = []
average_step_sizes_radau = []
average_step_sizes_bdf = []

for tol in TOL:
    options = {'rtol': tol, 'atol': tol}
    
    sol_rk45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(E,), **options)
    average_step_sizes_rk45.append(np.mean(np.diff(sol_rk45.t)))
    
    sol_rk23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(E,), **options)
    average_step_sizes_rk23.append(np.mean(np.diff(sol_rk23.t)))

    sol_radau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(E,), **options)
    average_step_sizes_radau.append(np.mean(np.diff(sol_radau.t)))
    
    sol_bdf = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(E,), **options)
    average_step_sizes_bdf.append(np.mean(np.diff(sol_bdf.t)))

log_tol = np.log10(TOL)
log_step_sizes_rk45 = np.log10(average_step_sizes_rk45)
log_step_sizes_rk23 = np.log10(average_step_sizes_rk23)
log_step_sizes_radau = np.log10(average_step_sizes_radau)
log_step_sizes_bdf = np.log10(average_step_sizes_bdf)

slope_rk45, _ = np.polyfit(log_step_sizes_rk45, log_tol, 1)
slope_rk23, _ = np.polyfit(log_step_sizes_rk23, log_tol, 1)
slope_radau, _ = np.polyfit(log_step_sizes_radau, log_tol, 1)
slope_bdf, _ = np.polyfit(log_step_sizes_bdf, log_tol, 1)

A9 = np.array([slope_rk45, slope_rk23, slope_radau, slope_bdf])

plt.figure(figsize=(10, 6))
plt.loglog(average_step_sizes_rk45, TOL, marker='o', label='RK45')
plt.loglog(average_step_sizes_rk23, TOL, marker='x', label='RK23')
plt.loglog(average_step_sizes_radau, TOL, marker='.', label='Radau')
plt.loglog(average_step_sizes_bdf, TOL, marker='^', label='BDF')
plt.ylabel('Tolerance')
plt.xlabel('Average Step Size')
plt.legend()
plt.grid()
plt.show()
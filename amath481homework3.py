import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy import special

## PART A

def shoot(x, phi_primes, epsilon):
    return [phi_primes[1], (x**2 - epsilon) * phi_primes[0]]

tol = 1e-4  # define a tolerance level 
epsilon_0 = 0.1
xspan =  np.arange(-4, 4.1, 0.1)

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
    A1[:, modes - 1] = np.abs(phi.y[0, :] / np.sqrt(norm))

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
A[0, 0] = multiplier*(-2/3) +  + ((-3.9)**2)
A[0, 1] = multiplier*(2/3)

# right boundary conditions
A[system_dim-1, system_dim-1] = multiplier*(-2/3) + ((3.9)**2)
A[system_dim-1, system_dim-2] = multiplier*(2/3)

A4, A3 = eigs(A, k=5, which='SI')
A4 = np.real(A4)
A3 = np.real(A3)
temp = np.zeros((81,5))
temp[1:80,:] = A3
temp[0,:] = ((4/3)*temp[1,:]) - ((1/3)*temp[2,:])
temp[80,:] = ((4/3)*temp[79,:]) - ((1/3)*temp[78,:])
A3 = temp
for i in range(5):
    A3[:, i] = np.abs(A3[:, i]) / np.sqrt(np.trapz(np.abs(A3[:, i])**2, xspan))

#=======================================================================
## PART C
def shootc(x, phi_primes, epsilon, gamma):
    return [phi_primes[1], (gamma*phi_primes[0]**2+x**2 - epsilon) * phi_primes[0]]

L=2
xp=[-L,L]
x_evals=np.linspace(-L,L,20*L+1)
alleigv=[]
alleigf=[]

for gamma in [0.05,-0.05]:
    epsilon0=0.1
    for modes in range(2):
        epsilon=epsilon0
        deps=0.2
        A=1e-3
        for j in range(1000):
            y0=np.array([A,np.sqrt(L**2-epsilon)*A])
            ys=solve_ivp(shootc,xp,y0,t_eval=x_evals,args=(epsilon,gamma))
            norm=np.trapz(ys.y[0,:]*ys.y[0,:],x_evals)
            boundcheck=ys.y[1,-1]+np.sqrt(L**2-epsilon)*ys.y[0,-1]

            if (np.abs(boundcheck)<tol) and (np.abs(1-norm)<tol):
                break
            else:
                A=A/np.sqrt(norm)
            
            y0=np.array([A,np.sqrt(L**2-epsilon)*A])
            ys=solve_ivp(shootc,xp,y0,t_eval=x_evals,args=(epsilon,gamma))
            norm=np.trapz(ys.y[0,:]*ys.y[0,:],x_evals)
            boundcheck=ys.y[1,-1]+np.sqrt(L**2-epsilon)*ys.y[0,-1]

            if (np.abs(boundcheck)<tol) and (np.abs(1-norm)<tol):
                break
            if (-1)**modes*boundcheck>0:
                epsilon+=deps
            else:
                epsilon-=deps/2
                deps/=2
        alleigv.append(epsilon)
        alleigf.append(np.abs(ys.y[0,:]))
        epsilon0=epsilon+0.1

A6=np.array(alleigv)[:2]
A8=np.array(alleigv)[2:]
A5=np.array(alleigf).T[:,:2]
A7=np.array(alleigf).T[:,2:]

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

# plt.figure(figsize=(10, 6))
# plt.loglog(average_step_sizes_rk45, TOL, marker='o', label='RK45')
# plt.loglog(average_step_sizes_rk23, TOL, marker='x', label='RK23')
# plt.loglog(average_step_sizes_radau, TOL, marker='.', label='Radau')
# plt.loglog(average_step_sizes_bdf, TOL, marker='^', label='BDF')
# plt.ylabel('Tolerance')
# plt.xlabel('Average Step Size')
# plt.legend()
# plt.grid()
# plt.show()

#=======================================================================
## PART E
exact_eigvals = np.array([1,3,5,7,9])
A11 = 100 * np.abs(A2 - exact_eigvals) / exact_eigvals
A13 = 100 * np.abs(A4 - exact_eigvals) / exact_eigvals
H = np.array([np.ones_like(xspan), 2*xspan, 4*xspan**2-2, 8*xspan**3-12*xspan,16*xspan**4-48*xspan**2+12])

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

phi_exact=np.zeros((81,5))
for j in range(5):
    phi_exact[:,j]=(np.exp(-xspan**2/2)*H[j,:]/np.sqrt(factorial(j)*(2**j)*np.sqrt(np.pi))).T

A10= []
A12 = []
for n in range(5):
    partaevec = A1[:,n]
    partbevec = A3[:,n]
    A10.append(np.trapz((np.abs(partaevec) - np.abs(phi_exact[:,n]))**2, xspan))
    A12.append(np.trapz((np.abs(partbevec) - np.abs(phi_exact[:,n]))**2, xspan))

A10 = np.array(A10)
A12 = np.array(A12)

print(A11)
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import bicgstab, gmres
import time
import imageio.v2 as imageio
import os

n = 64   # N value in x and y directions
system_dim = n * n  # total size of matrix
step_size=20/64 # L/n

# creating A
e0 = np.zeros((system_dim, 1))  # vector of zeros
e1 = np.ones((system_dim, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, n+1):
    e2[n*j-1] = 0  # overwrite every nth value with zero
    e4[n*j-1] = 1  # overwirte every nth value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:system_dim] = e2[0:system_dim-1]
e3[0] = e2[system_dim-1]

e5 = np.zeros_like(e4)
e5[1:system_dim] = e4[0:system_dim-1]
e5[0] = e4[system_dim-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(system_dim-n), -n, -n+1, -1, 0, 1, n-1, n, (system_dim-n)]

matA = spdiags(diagonals, offsets, system_dim, system_dim).toarray()
matA = (1/(step_size**2)) * matA

#=====================================================================================
# creating B (first x derivative)

e1 = np.ones((system_dim, 1))
e1 = e1.flatten()
diagonals_B = [e1, -e1, e1, -e1]
offsets_B = [-(system_dim - n), -n, n, (system_dim-n)]

matB = spdiags(diagonals_B, offsets_B, system_dim, system_dim).toarray()
matB = (1 / (2 * step_size)) * matB

#=====================================================================================
# creating C (first y derivative)

e0 = np.zeros((system_dim, 1))  # vector of zeros
e1 = np.ones((system_dim, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, n+1):
    e2[n*j-1] = 0  # overwrite every nth value with zero
    e4[n*j-1] = 1  # overwirte every nth value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:system_dim] = e2[0:system_dim-1]
e3[0] = e2[system_dim-1]

e5 = np.zeros_like(e4)
e5[1:system_dim] = e4[0:system_dim-1]
e5[0] = e4[system_dim-1]

# Place diagonal elements
diagonals = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [-(n-1), -1, 1, (n-1)]

matC = spdiags(diagonals, offsets, system_dim, system_dim).toarray()
matC = (1 / (2 * step_size)) * matC

#=====================================================================================
# homework 5 part a

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w = np.exp(-X**2 - Y**2/20).flatten()

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 10e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 10e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2


# Define the ODE system
def spc_rhs(t, w, nu):
    wt = fft2(w.reshape(nx,ny))

    psit = -wt / K
    psi = np.real(ifft2(psit)).reshape(N)

    rhs = nu*np.dot(matA,w)-((np.dot(matB,psi)*np.dot(matC,w))-(np.dot(matC,psi)*np.dot(matB,w)))
    return rhs

start_time = time.time() # Record the start time

# Solve the ODE
wtsol = solve_ivp(
    spc_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"FFT Elapsed time: {elapsed_time:.2f} seconds")
A1 = wtsol.y

# # Plot the results
# for j, t in enumerate(tspan):
#     w = wtsol.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
#     plt.subplot(3, 3, j + 1)
#     plt.pcolor(x, y, w)
#     plt.title(f'Time: {t}')
#     plt.colorbar()

# plt.tight_layout()
# plt.show()

# #=====================================================================================
# # homework 5 part b

n = 64   # N value in x and y directions
system_dim = n * n  # total size of matrix
step_size=20/64 # L/n

# creating A
e0 = np.zeros((system_dim, 1))  # vector of zeros
e1 = np.ones((system_dim, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, n+1):
    e2[n*j-1] = 0  # overwrite every nth value with zero
    e4[n*j-1] = 1  # overwirte every nth value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:system_dim] = e2[0:system_dim-1]
e3[0] = e2[system_dim-1]

e5 = np.zeros_like(e4)
e5[1:system_dim] = e4[0:system_dim-1]
e5[0] = e4[system_dim-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(system_dim-n), -n, -n+1, -1, 0, 1, n-1, n, (system_dim-n)]

matA = spdiags(diagonals, offsets, system_dim, system_dim).toarray()
matA[0,0] = 2
matA = (1/(step_size**2)) * matA

w = np.exp(-X**2 - Y**2/20).flatten()

# Define the ODE system
def basic_rhs(t, w, nu):
    psi = np.linalg.solve(matA, w)
    wt = nu*np.dot(matA,w)-((np.dot(matB,psi)*np.dot(matC,w))-(np.dot(matC,psi)*np.dot(matB,w)))
    return wt

start_time = time.time() # Record the start time

# Solve the ODE
wtsol = solve_ivp(
    basic_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"GE Elapsed time: {elapsed_time:.2f} seconds")
A2 = wtsol.y

# Using LU
P, L, U = lu(matA)
w = np.exp(-X**2 - Y**2/20).flatten()

start_time = time.time() # Record the start time

# Define the ODE system
def lu_rhs(t, w, nu):
    Pb = np.dot(P, w)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    wt = nu*np.dot(matA,w)-((np.dot(matB,psi)*np.dot(matC,w))-(np.dot(matC,psi)*np.dot(matB,w)))
    return wt

# Solve the ODE
wtsol = solve_ivp(
    lu_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"LU Elapsed time: {elapsed_time:.2f} seconds")
A3 = wtsol.y

# Using BICGSTAB
w = np.exp(-X**2 - Y**2/20).flatten()

residuals_bicgstab = []
def bicgstab_callback(residual_norm):
    residuals_bicgstab.append(residual_norm)

# Define the ODE system
def bicgstab_rhs(t, w, nu):
    psi, exitcode = bicgstab(csr_matrix(matA), w, rtol = 1e-4, callback = bicgstab_callback)
    wt = nu*np.dot(matA,w)-((np.dot(matB,psi)*np.dot(matC,w))-(np.dot(matC,psi)*np.dot(matB,w)))
    return wt

start_time = time.time() # Record the start time

# Solve the ODE
wtsol = solve_ivp(
    bicgstab_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"BICGSTAB Elapsed time: {elapsed_time:.2f} seconds")
print(f"BICGSTAB iterations: {len(residuals_bicgstab)}")

# Using GMRES
w = np.exp(-X**2 - Y**2/20).flatten()

residuals_gm = []
def gm_callback(residual_norm):
    residuals_gm.append(residual_norm)

# Define the ODE system
def gmres_rhs(t, w, nu):
    psi, exitcode = gmres(csr_matrix(matA), w, rtol = 1e-4, callback = gm_callback)
    wt = nu*np.dot(matA,w)-((np.dot(matB,psi)*np.dot(matC,w))-(np.dot(matC,psi)*np.dot(matB,w)))
    return wt

start_time = time.time() # Record the start time

# Solve the ODE
wtsol = solve_ivp(
    gmres_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"GMRES Elapsed time: {elapsed_time:.2f} seconds")
print(f"GMRES iterations: {len(residuals_gm)}")

# #=====================================================================================
# homework 5 parts c & d (use FFT since fastest)

# creating A
e0 = np.zeros((system_dim, 1))  # vector of zeros
e1 = np.ones((system_dim, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, n+1):
    e2[n*j-1] = 0  # overwrite every nth value with zero
    e4[n*j-1] = 1  # overwirte every nth value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:system_dim] = e2[0:system_dim-1]
e3[0] = e2[system_dim-1]

e5 = np.zeros_like(e4)
e5[1:system_dim] = e4[0:system_dim-1]
e5[0] = e4[system_dim-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(system_dim-n), -n, -n+1, -1, 0, 1, n-1, n, (system_dim-n)]

matA = spdiags(diagonals, offsets, system_dim, system_dim).toarray()
matA = (1/(step_size**2)) * matA

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w = (np.exp(-(X+1)**2 - Y**2/20)-np.exp(-(X-1)**2 - Y**2/20)).flatten()

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 10e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 10e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Solve the ODE
wtsol = solve_ivp(
    spc_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

frames = []
# make animation
for j, t in enumerate(tspan):
    w = wtsol.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

    frame_filename = f"frame_{j}.png"
    plt.savefig(frame_filename)
    plt.close()
    frames.append(imageio.imread(frame_filename))
    # os.remove(frame_filename)

gif_filename = "oppositely_charged_gaussian.gif"
imageio.mimsave(gif_filename, frames, fps=3)

# 2 same charged gaussian
w = (np.exp(-(X+1)**2 - Y**2/20)+np.exp(-(X-1)**2 - Y**2/20)).flatten()
wtsol = solve_ivp(
    spc_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

frames = []
# make animation
for j, t in enumerate(tspan):
    w = wtsol.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

    frame_filename = f"frame_{j}.png"
    plt.savefig(frame_filename)
    plt.close()
    frames.append(imageio.imread(frame_filename))
    # os.remove(frame_filename)

gif_filename = "same_charged_gaussian.gif"
imageio.mimsave(gif_filename, frames, fps=3)

# 2 opposite pairs
w = (np.exp(-(X+1)**2 - (Y+3.25)**2/4)-np.exp(-(X-1)**2 - (Y-3.25)**2/4)+np.exp(-(X+1)**2 - (Y-1)**2/4)-np.exp(-(X-1)**2 - (Y+1)**2/4)).flatten()
wtsol = solve_ivp(
    spc_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

frames = []
# make animation
for j, t in enumerate(tspan):
    w = wtsol.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

    frame_filename = f"frame_{j}.png"
    plt.savefig(frame_filename)
    plt.close()
    frames.append(imageio.imread(frame_filename))
    # os.remove(frame_filename)

gif_filename = "opposite_pairs.gif"
imageio.mimsave(gif_filename, frames, fps=3)


# random assortment
def gen_random_w():
    w=0
    for i in range(13):
        charge = np.random.choice([1, -1])
        w+=charge*np.exp(-(X+np.random.uniform(-10, 10))**2 - (Y+np.random.uniform(-10, 10))**2/np.random.uniform(4, 20)).flatten()
    return w
w = gen_random_w()
wtsol = solve_ivp(
    spc_rhs,
    [tspan[0], tspan[-1]],
    w,
    t_eval=tspan,
    args=(nu,),
    method="RK45",
)

frames = []
# make animation
for j, t in enumerate(tspan):
    w = wtsol.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

    frame_filename = f"frame_{j}.png"
    plt.savefig(frame_filename)
    plt.close()
    frames.append(imageio.imread(frame_filename))
    # os.remove(frame_filename)

gif_filename = "random_assortment.gif"
imageio.mimsave(gif_filename, frames, fps=3)
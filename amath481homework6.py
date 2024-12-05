import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import kron

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
m = 1 # number of spirals
D1 = 0.1
D2 = 0.1
beta = 1

Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

r = np.sqrt(X**2 + Y**2)
theta = np.angle(X + 1j*Y)
U = np.tanh(r) * np.cos(m*theta - r)
V = np.tanh(r) * np.sin(m*theta - r)

#=====================================================================================
# PART A

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Define the ODE system
def spc_rhs(t, transformed_values):
    ut = transformed_values[:2*N] # "ut" & "vt" are fourier transformed u & v
    vt = transformed_values[2*N:]

    utc = ut[0:N] + 1j*ut[N:]
    ut = utc.reshape((nx, ny))
    vtc = vt[0:N] + 1j*vt[N:]
    vt = vtc.reshape((nx, ny))

    laplacian_ut = -K*ut
    laplacian_vt = -K*vt

    u = np.real(ifft2(ut))
    v = np.real(ifft2(vt))

    A2 = u**2 + v**2
    lambd = 1 - A2
    omega = -beta*(A2)
    
    dut_dt = (fft2(lambd*u - omega*v) + D1*laplacian_ut).reshape(N)
    dut_dt = np.hstack([np.real(dut_dt),np.imag(dut_dt)])

    dvt_dt = (fft2(omega*u + lambd*v) + D2*laplacian_vt).reshape(N)
    dvt_dt = np.hstack([np.real(dvt_dt),np.imag(dvt_dt)])

    return np.hstack([dut_dt, dvt_dt])

# Solve the ODE
ut0 = np.hstack([np.real(fft2(U).reshape(N)),np.imag(fft2(U).reshape(N))])
vt0 = np.hstack([np.real(fft2(V).reshape(N)),np.imag(fft2(V).reshape(N))])
transformed_initialconds = np.hstack([ut0, vt0])
transformed_sol = solve_ivp(spc_rhs, (tspan[0], tspan[-1]), transformed_initialconds, t_eval=tspan, method='RK45')

utsol = transformed_sol.y[:2*N, :]
vtsol = transformed_sol.y[2*N:, :]
utcsol = utsol[:N, :] + 1j*utsol[:N, :]
vtcsol = vtsol[:N, :] + 1j * vtsol[N:, :]

A1 = np.vstack([utcsol, vtcsol])

# # Plotting
# u_all = np.real(ifft2(utcsol.reshape(N, -1)))  # Get all values of u over time
# v_all = np.real(ifft2(vtcsol.reshape(N, -1)))  # Get all values of v over time

# u_min = np.min(u_all)
# u_max = np.max(u_all)

# v_min = np.min(v_all)
# v_max = np.max(v_all)
# for step, t in enumerate(tspan):
#     u = np.real(ifft2(utcsol[:, step].reshape((nx, ny))))
#     v = np.real(ifft2(vtcsol[:, step].reshape((nx, ny))))

#     plt.subplot(3, 3, step + 1)
    
#     # Plot u with one colormap
#     c1 = plt.pcolor(x, y, u, cmap='viridis', vmin=u_min, vmax=u_max)
    
#     # Plot v with another colormap, semi-transparent
#     c2 = plt.pcolor(x, y, v, cmap='plasma', vmin=v_min, vmax=v_max, alpha=0.6)
    
#     plt.title(f'Time: {t}')
#     plt.colorbar(c1)  # Colorbar for u
#     plt.colorbar(c2)  # Colorbar for v

# plt.tight_layout()
# plt.show()

#=====================================================================================
# PART B

def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	return D, x.reshape(N+1)

N = 30
D, x = cheb(N)

# no flux boundaries
D[N,:] = 0
D[0,:] = 0

Dxx = np.dot(D,D)/((Lx/2)**2)
n2 = (N+1)**2
I = np.eye(len(Dxx))
L = kron(I,Dxx)+kron(Dxx,I) # 2D laplacian

y = x
X,Y = np.meshgrid(x,y)
X = X*(Lx/2)
Y = Y*(Ly/2)

r = np.sqrt(X**2 + Y**2)
theta = np.angle(X + 1j*Y)
U = np.tanh(r) * np.cos(m*theta - r)
V = np.tanh(r) * np.sin(m*theta - r)

uv0 = np.hstack([U.reshape(n2), V.reshape(n2)])

def RD_2D(t,uv):
    u = uv[0:n2]
    v = uv[n2:]

    A2 = u**2 + v**2
    lambd = 1 - A2
    omega = -beta*A2

    rhs_u = D1*np.dot(L,u) + lambd*u - omega*v
    rhs_v = D2*np.dot(L,v) + omega*u + lambd*v
    rhs = np.hstack([rhs_u, rhs_v])
    return rhs

sol = solve_ivp(RD_2D, [0, 4], uv0, t_eval=tspan, method='RK45')
A2 = sol.y
print(A2)
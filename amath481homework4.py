import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

n = 8    # N value in x and y directions
system_dim = n * n  # total size of matrix
step_size=2.5 # 20/8

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
A1 = (1/(step_size**2)) * matA

#=====================================================================================
# creating B (first x derivative)

e1 = np.ones((system_dim, 1))
e1 = e1.flatten()
diagonals_B = [e1, -e1, e1, -e1]
offsets_B = [-(system_dim - n), -n, n, (system_dim-n)]

matB = spdiags(diagonals_B, offsets_B, system_dim, system_dim).toarray()
A2 = (1 / (2 * step_size)) * matB

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
A3 = (1 / (2 * step_size)) * matC
# plt.imshow(A3)
# plt.colorbar()
# plt.show()

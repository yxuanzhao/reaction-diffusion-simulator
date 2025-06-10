import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve, factorized
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operators.discrete_operators import laplacian_neumann

'''
We solve for the heat equation

u_t = D*(∇^2u)

with initial condition a dirac function at center.

'''

# -------------------Spatial parameters-------------------
D = 0.1
L = 25.0
N = 101
h = L/N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# -------------------Time parameters-------------------
dt = 0.01
T = 60.0
steps = int(T/dt)

# -------------------Initial values-------------------
u_initial = np.zeros((N, N))
u_initial[N//2, N//2] = 100.0

# -------------------Discrete operators-------------------
Lap = laplacian_neumann(N, h, 'csc')
I = identity(N**2, format='csc')

# -------------------Flatten initial values-------------------
u = u_initial.flatten()
A_u = I - dt*D*Lap
solve_A = factorized(A_u)
# -------------------Time evolution-------------------
for step in range(steps):
    u_new = u.copy()
    u_new = solve_A(u)
    u = u_new


# -------------------Plot the result-------------------
u_final = u.reshape(N, N)
plt.imshow(u_final, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.colorbar(label='$u(x, y)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(f'Heat distribution at $T=$ {T}')
plt.show()
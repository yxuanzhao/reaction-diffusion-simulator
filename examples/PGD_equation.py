import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve, factorized
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operators.discrete_operators import laplacian_neumann, derivative_x_neumann, derivative_y_neumann

'''
In this code we take

    A = phi
    B = phi
    C = 0
'''

# -------------------Spatial parameters-------------------
L = 4
N = 50
h = L/N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)


# -------------------Time parameters-------------------
T_step = 100
dt = 1/T_step
T = 1000.0
steps = int(T/dt)


# -------------------Equation parameters-------------------
r_u = 5.0
r_v = 6.0
c = 1.0
gamma = 1.0
mu_u = 2.0
mu_v = 3.7
kappa = 1.0
delta = 0.001
w_u, w_v, w_phi = 30, 10, 10
D_u, D_v, D_phi = 0.01, 0.005, 0.1

s = c*(r_u - r_v) / (kappa*(c + mu_u - mu_v))
u_0 = (c*r_v/(gamma*kappa*s**2) + delta/(kappa*s) - mu_v/(gamma*s))
v_0 = (c*r_v/(gamma*kappa*s) - c*r_v/(gamma*kappa*s**2) - mu_v/gamma + mu_v/(gamma*s) - delta/(kappa*s))
phi_0 = (c + mu_u - mu_v)/(r_u - r_v)


# -------------------Initial values-------------------
A = 0.1
A_u = 0.01
A_v = 0.01
A_phi = 0.01
ku_x, ku_y = 3*np.pi/4, np.pi
kv_x, kv_y = 3*np.pi/4, np.pi
kphi_x, kphi_y = 3*np.pi/4, np.pi

du = A*A_u*np.cos(ku_x*X + ku_y*Y)
dv = A*A_v*np.cos(kv_x*X + kv_y*Y)
dphi = A*A_phi*np.cos(kphi_x*X + kphi_y*Y)

u_initial = u_0 + du
v_initial = v_0 + dv
phi_initial = phi_0 + dphi


# -------------------Discrete operators-------------------
Lap = laplacian_neumann(N, h, 'csc')
Grad_x = derivative_x_neumann(N, h, 'csc')
Grad_y = derivative_y_neumann(N, h, 'csc')
I = identity(N**2, format='csc')


# -------------------Flatten initial values-------------------
u = u_initial.flatten()
v = v_initial.flatten()
phi = phi_initial.flatten()


# -------------------Time evolution-------------------
A_u = I - dt*D_u*Lap
A_v = I - dt*D_v*Lap
A_phi = I - dt*D_phi*Lap

solve_Au = factorized(A_u)
solve_Av = factorized(A_v)
solve_Aphi = factorized(A_phi)

# -------------------time snapshot-------------------
save_steps = [steps]

for step in range(steps + 1):
    u_new = u.copy()
    v_new = v.copy()
    phi_new = phi.copy()

    grad_u_x = Grad_x @ u
    grad_u_y = Grad_y @ u
    grad_v_x = Grad_x @ v
    grad_v_y = Grad_y @ v
    grad_phi_x = Grad_x @ phi
    grad_phi_y = Grad_y @ phi
    lap_phi = Lap @ phi

    reaction_u = -2*w_u*D_u*(grad_u_x*grad_phi_x + grad_u_y*grad_phi_y) - 2*w_u*D_u*u*lap_phi + u*(r_u*phi - c - gamma*(u + v) - mu_u)
    reaction_v = -2*w_v*D_v*(grad_v_x*grad_phi_x + grad_v_y*grad_phi_y) - 2*w_v*D_v*v*lap_phi + v*(r_v*phi - gamma*(u + v) - mu_v)
    reaction_phi = c*u - (kappa*(u + v) + delta)*phi

    u_new = solve_Au(reaction_u * dt + u)
    v_new = solve_Av(reaction_v * dt + v)
    phi_new = solve_Aphi(reaction_phi * dt + phi)

    u = u_new
    v = v_new
    phi = phi_new

    u = u.reshape(N, N)
    v = v.reshape(N, N)
    phi = phi.reshape(N, N)


    # -------------------save the figures-------------------
    if step in save_steps:
        t_now = step * dt
        u_now = u.copy().reshape(N, N)
        v_now = v.copy().reshape(N, N)
        phi_now = phi.copy().reshape(N, N)

        # Plot u
        plt.imshow(u_now, cmap='viridis', origin='lower', extent=[0, L, 0, L])
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        #plt.title(f"$u$ at $T$ = {t_now:.2f}")
        plt.savefig(f"results/u_{int(step*dt)}.pdf", dpi=300)
        plt.close()

        # Plot v
        plt.imshow(v_now, cmap='plasma', origin='lower', extent=[0, L, 0, L])
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        #plt.title(f"$v$ at $T$ = {t_now:.2f}")
        plt.savefig(f"results/v_{int(step*dt)}.pdf", dpi=300)
        plt.close()

        # Plot phi
        plt.imshow(phi_now, cmap='inferno', origin='lower', extent=[0, L, 0, L])
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        #plt.title(f"$\\phi$ at $T$ = {t_now:.2f}")
        plt.savefig(f"results/phi_{int(step*dt)}.pdf", dpi=300)
        plt.close()

    u = u.flatten()
    v = v.flatten()
    phi = phi.flatten()

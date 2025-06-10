# reaction-diffusion-simulator
Finite-difference implementation of a reaction-diffusion PDE system with Neumann boundary conditions in 2D.

### PDE

We implement `Python` codes to solve for the following PDE systems we construct for a Public Goods Dilemma model, we call it a PGD equation:
$$
\begin{equation}
    \begin{aligned}
        \frac{\partial u}{\partial t}&=D_u\nabla\cdot\left[\nabla u-2u\nabla \ln f(w_u,A)\right]+u\left[r_u\phi-c-\gamma(u+v)-\mu_u\right],\\
        \frac{\partial v}{\partial t}&=D_v\nabla\cdot\left[\nabla v-2v\nabla\ln f(w_v,B)\right]+v\left[r_v\phi-\gamma(u+v)-\mu_v\right],\\
        \frac{\partial \phi}{\partial t}&=D_\phi\nabla\cdot\left[\nabla \phi-2\phi\nabla\ln f(w_\phi,C)\right]+cu-\left[\kappa(u+v)+\delta\right]\phi.
    \end{aligned}
    \nonumber
\end{equation}
$$

#### `discrete_operators.py`

Provides reusable discrete differential operators for 2D PDE solvers, including:

- `laplacian_neumann()`: 2D Laplace operator with Neumann boundary conditions
- `derivative_x_neumann()`: $x$-directional derivative operators
- `derivative_y_neumann()`: $y$-directional derivative operators
- Implemented using SciPy sparse matrices for efficient numerical simulation

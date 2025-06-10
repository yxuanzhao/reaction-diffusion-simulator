import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operators.discrete_operators import laplacian_neumann

def test_laplacian_convergence():
    ns = [20, 40, 80, 160]
    hs = []
    errors = []

    for n in ns:
        h = 1.0/(n - 1)
        hs.append(h)

        # grid
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # function u(x, y)
        u = np.cos(np.pi*X)*np.cos(np.pi*Y)

        # exact value of laplacian
        lap_u_exact = -2*(np.pi**2)*u

        # approximating discrete value of laplacian
        u_flat = u.flatten()
        L = laplacian_neumann(n, h)
        lap_u_num = L @ u_flat

        # error computing
        error = np.abs(lap_u_num - lap_u_exact.flatten())
        max_error = np.max(error)
        errors.append(max_error)

        print(f"n = {n}, h = {h:.4e}, max error = {max_error:.2e}")

    log_h = np.log10(hs)
    log_err = np.log10(errors)

    # visualize the error curve
    plt.figure(figsize=(6, 5))
    plt.plot(log_h, log_err, 'o-', label='Laplacian error')
    plt.xlabel(r'$\log_{10}(h)$')
    plt.ylabel(r'$\log_{10}(\mathrm{max\ error})$')
    plt.title('Convergence of Laplacian (Neumann BC)')
    plt.grid(True)
    plt.legend()

    slope, _ = np.polyfit(log_h, log_err, 1)
    print(f"Estimated order: {slope:.2f}")

    plt.show()


# run test
test_laplacian_convergence()

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, diags

def laplacian_neumann(n, h, matrix_type = 'csr'):
    """
    Construct a 2D Laplacian operator with Neumann boundary conditions on [0,1]x[0,1] with n^2 grids.

    Parameters:
    ----------
    n : int
        Number of grid points in the x- and y- direction.
    h : float
        Step length of the domain in the x-direction.
    matrix_type : string
        Determine matrix type. Default value 'csr'

    Returns:
    ----------
    scipy.sparse.spmatrix
        Laplacian matrix in the specified sparse format.
    """
    laplace = np.zeros((n**2, n**2))
    neighbour = 4
    for i in range(n):
        for j in range(n):
            if i != 0 and i != n - 1:
                if j != 0 and j != n - 1:
                    laplace[i*n + j, (i - 1)*n + j] = 1/h**2 
                    laplace[i*n + j, (i + 1)*n + j] = 1/h**2 
                    laplace[i*n + j, i*n + j + 1] = 1/h**2
                    laplace[i*n + j, i*n + j - 1] = 1/h**2 
                    laplace[i*n + j, i*n + j] = -neighbour/h**2
                elif j == 0:
                    laplace[i*n, i*n + 1] = 2/h**2
                    laplace[i*n, (i + 1)*n] = 1/h**2
                    laplace[i*n, (i - 1)*n] = 1/h**2
                    laplace[i*n, i*n] = -neighbour/h**2
                else:
                    laplace[i*n + n - 1, i*n + n - 2] = 2/h**2
                    laplace[i*n + n - 1, (i + 1)*n + n - 1] = 1/h**2
                    laplace[i*n + n - 1, (i - 1)*n + n - 1] = 1/h**2
                    laplace[i*n + n - 1, i*n + n - 1] = -neighbour/h**2
            elif i == 0:
                if j != 0 and j != n - 1:
                    laplace[j, n + j] = 2/h**2
                    laplace[j, j + 1] = 1/h**2
                    laplace[j, j - 1] = 1/h**2
                    laplace[j, j] = -neighbour/h**2
                elif j == 0:
                    laplace[0, n] = 2/h**2
                    laplace[0, 1] = 2/h**2
                    laplace[0, 0] = -neighbour/h**2
                else:
                    laplace[n - 1, 2*n - 1] = 2/h**2
                    laplace[n - 1, n - 2] = 2/h**2
                    laplace[n - 1, n - 1] = -neighbour/h**2
            else:
                if j != 0 and j != n - 1:
                    laplace[n*(n - 1) + j, n*(n - 2) + j] = 2/h**2
                    laplace[n*(n - 1) + j, n*(n - 1) + j - 1] = 1/h**2
                    laplace[n*(n - 1) + j, n*(n - 1) + j + 1] = 1/h**2
                    laplace[n*(n - 1) + j, n*(n - 1) + j] = -neighbour/h**2
                elif j == 0:
                    laplace[n*(n - 1), n*(n - 2)] = 2/h**2
                    laplace[n*(n - 1), n*(n - 1) + 1] = 2/h**2
                    laplace[n*(n - 1), n*(n - 1)] = -neighbour/h**2
                else:
                    laplace[n*(n - 1) + n - 1, n*(n - 2) + n - 1] = 2/h**2
                    laplace[n*(n - 1) + n - 1, n*(n - 1) + n - 2] = 2/h**2
                    laplace[n*(n - 1) + n - 1, n*(n - 1) + n - 1] = -neighbour/h**2
    
    if matrix_type == 'csr':
        laplace_sparse = csr_matrix(laplace)
    elif matrix_type == 'csc':
        laplace_sparse = csc_matrix(laplace)
    else:
        raise ValueError("matrix_format must be 'csr' or 'csc'")

    return laplace_sparse


def derivative_x_neumann(n, h, matrix_type = 'csr'):
    """
    Construct a 2D x-differential operator with Neumann boundary conditions on [0,1]x[0,1] with n^2 grids.

    Parameters:
    ----------
    n : int
        Number of grid points in the x- and y- direction.
    h : float
        Step length of the domain in the x-direction.
    matrix_type : string
        Determine matrix type. Default value 'csr'

    Returns:
    ----------
    scipy.sparse.spmatrix
        Laplacian matrix in the specified sparse format.
    """
    D_x = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            if i != 0 and i != n - 1:
                D_x[i*n + j, (i + 1)*n + j] = 1/(2*h)
                D_x[i*n + j, (i - 1)*n + j] = -1/(2*h)
    
    if matrix_type == 'csr':
        D_x_sparse = csr_matrix(D_x)
    elif matrix_type == 'csc':
        D_x_sparse = csc_matrix(D_x)
    else:
        raise ValueError("matrix_format must be 'csr' or 'csc'")

    return D_x_sparse


def derivative_y_neumann(n, h, matrix_type = 'csr'):
    """
    Construct a 2D y-differential operator with Neumann boundary conditions on [0,1]x[0,1] with n^2 grids.

    Parameters:
    ----------
    n : int
        Number of grid points in the x- and y- direction.
    h : float
        Step length of the domain in the x-direction.
    matrix_type : string
        Determine matrix type. Default value 'csr'

    Returns:
    ----------
    scipy.sparse.spmatrix
        Laplacian matrix in the specified sparse format.
    """
    D_y = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            if j != 0 and j != n - 1:
                D_y[i*n + j, i*n + j + 1] = 1/(2*h)
                D_y[i*n + j, i*n + j - 1] = -1/(2*h)
    
    if matrix_type == 'csr':
        D_y_sparse = csr_matrix(D_y)
    elif matrix_type == 'csc':
        D_y_sparse = csc_matrix(D_y)
    else:
        raise ValueError("matrix_format must be 'csr' or 'csc'")

    return D_y_sparse
import numpy as np

def noisy_meshgrid_2d_regression(bounds: float, n_sqrt_mesh: int, input_noise: float):
    """
    Generate a 2D meshgrid for regression tasks.
    """
    
    # Create grid
    x = np.linspace(-bounds, bounds, n_sqrt_mesh)
    y = np.linspace(-bounds, bounds, n_sqrt_mesh)
    X, Y = np.meshgrid(x, y)

    # Noise levels
    input_noise = input_noise

    # Add noise to inputs
    X_noisy = X + input_noise * np.random.randn(*X.shape)
    Y_noisy = Y + input_noise * np.random.randn(*Y.shape)

    return X_noisy, Y_noisy

def sin_sqrt_surface_from_meshgrid(X: np.array, Y: np.array, output_noise: float):
    """
    Compute a surface from the given meshgrid.
    """

    R = np.divide(np.sqrt(X**2 + Y**2),0.9)
    F_clean = 0.5*np.sin(1+R)  

    # Add noise to output
    F_noisy = F_clean + output_noise * np.random.randn(*F_clean.shape)

    return F_noisy

def compute_R_squared(feasible_sets):
    """ Compute the R-squared value for a collection of feasible sets. Which is the optimal RMSE, only on D_prime and for the l2 norm."""
    return sum(
        np.sum((Y - Y.mean(axis=0))**2)
        for Y in feasible_sets if len(Y)
    ) / sum(len(Y) for Y in feasible_sets)
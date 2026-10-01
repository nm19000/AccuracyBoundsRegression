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

def generate_dataset(
        bounds,
        n_sqrt_mesh,
        input_noise,
        output_noise):

    x = np.linspace(-bounds, bounds, n_sqrt_mesh)
    y = np.linspace(-bounds, bounds, n_sqrt_mesh)

    X1, X2 = np.meshgrid(x, y)

    X = np.column_stack([
        X1.ravel(),
        X2.ravel()
    ])

    # U([0,noise])
    X += np.random.uniform(
        0,
        input_noise,
        size=X.shape
    )

    R = np.sqrt(
        X[:,0]**2 +
        X[:,1]**2
    )

    y = np.sin(1 + 0.9*R)

    y += np.random.uniform(
        0,
        output_noise,
        size=y.shape
    )

    return X, y
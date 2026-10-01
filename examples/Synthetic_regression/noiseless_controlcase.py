import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nn_xy,
)
from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)

# ==========================================================
# Deterministic target function
# ==========================================================

def f_true(X):
    R = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    return np.sin(1 + 0.9 * R)

# ==========================================================
# Dataset: noiseless control
# ==========================================================

def generate_dataset_noiseless(bounds=2, n_sqrt_mesh=30):
    x = np.linspace(-bounds, bounds, n_sqrt_mesh)
    y = np.linspace(-bounds, bounds, n_sqrt_mesh)
    X1, X2 = np.meshgrid(x, y)

    X = np.column_stack([X1.ravel(), X2.ravel()])
    y = f_true(X)

    return X, y

# ==========================================================
# Kernel size computation
# ==========================================================

def compute_statistics(X, y, tau_x, tau_y, N_max=300):
    result = feasible_sets_features_nn_xy(
        X,
        y.reshape(-1, 1),
        N_max=N_max,
        tau_x=tau_x,
        tau_y=tau_y,
        ext_D=False,
    )

    ksize, M_prime, Nk_stats = compute_kernel_size(result["feasible_sets"])
    return ksize, result, M_prime, Nk_stats

# ==========================================================
# Experiment
# ==========================================================

def run_experiment(X_train, X_test, y_train, y_test, tau_x, tau_y, N_max):
    K_train, _, M_prime_train, Nk_stats_train = compute_statistics(
        X_train, y_train, tau_x=tau_x, tau_y=tau_y, N_max=N_max
    )
    K_test, _, M_prime_test, Nk_stats_test = compute_statistics(
        X_test, y_test, tau_x=tau_x, tau_y=tau_y, N_max=N_max
    )

    return {
        "tau_x": tau_x,
        "tau_y": tau_y,
        "N_max": N_max,
        "Bayes_RMSE": 0.0,
        "M_prime_train": M_prime_train,
        "Nk_stats_train": Nk_stats_train,
        "M_prime_test": M_prime_test,
        "Nk_stats_test": Nk_stats_test,
        "Kernel_train": K_train,
        "Kernel_test": K_test,
    }

# ==========================================================
# Main
# ==========================================================

def main():
    np.random.seed(42)

    X, y = generate_dataset_noiseless(bounds=2, n_sqrt_mesh=30)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=42
    )

    tau_x_grid = [0.0, 0.005, 0.01, 0.014, 0.02, 0.03]
    tau_y_grid = [0.0]   # pure input-neighborhood effect
    N_max_grid = [10, 20, 50, 100, 200, 300]

    results = []
    for tau_x in tau_x_grid:
        for tau_y in tau_y_grid:
            for N_max in N_max_grid:
                results.append(
                    run_experiment(
                        X_train, X_test, y_train, y_test,
                        tau_x=tau_x, tau_y=tau_y, N_max=N_max
                    )
                )

    df = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(df.round(6))

    df.to_csv(
        "synthetic_regression_noiseless_control.csv",
        index=False,
        float_format="%.6f"
    )

if __name__ == "__main__":
    main()

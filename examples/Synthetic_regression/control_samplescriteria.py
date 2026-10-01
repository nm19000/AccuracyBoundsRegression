import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

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
# Dataset: input noise + output noise
# ==========================================================

def generate_dataset(bounds=2, n_sqrt_mesh=30, input_noise=0.01, output_noise=0.03):
    x = np.linspace(-bounds, bounds, n_sqrt_mesh)
    y = np.linspace(-bounds, bounds, n_sqrt_mesh)
    X1, X2 = np.meshgrid(x, y)

    X_clean = np.column_stack([X1.ravel(), X2.ravel()])
    X_obs = X_clean + np.random.uniform(0, input_noise, size=X_clean.shape)

    y_clean = f_true(X_obs)
    y_obs = y_clean + np.random.uniform(0, output_noise, size=y_clean.shape)

    return X_obs, y_obs

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
# Compatibility check
# ==========================================================

def admissible_under_model(x_anchor, y_candidate, output_noise):
    fx = f_true(x_anchor.reshape(1, -1))[0]
    return (fx <= y_candidate) and (y_candidate <= fx + output_noise)

def compatibility_check_bruteforce(X, y, tau_x, tau_y, output_noise):
    """
    Checks all candidate reassignments accepted by the threshold rule.
    Reports how many are truly admissible under the known model.
    """
    Dx = pairwise_distances(X, metric="euclidean")
    Dy = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

    accepted = 0
    admissible = 0
    accepted_by_x = 0
    accepted_by_y = 0
    admissible_by_x = 0
    admissible_by_y = 0

    n = len(X)

    for k in range(n):
        for j in range(n):
            x_close = Dx[j, k] <= tau_x
            y_close = Dy[j, k] <= tau_y

            if x_close or y_close:
                accepted += 1
                ok = admissible_under_model(X[k], y[j], output_noise=output_noise)

                if ok:
                    admissible += 1

                if x_close:
                    accepted_by_x += 1
                    if ok:
                        admissible_by_x += 1

                if y_close:
                    accepted_by_y += 1
                    if ok:
                        admissible_by_y += 1

    return {
        "accepted_total": accepted,
        "admissible_total": admissible,
        "precision_total": admissible / accepted if accepted > 0 else np.nan,
        "accepted_by_x": accepted_by_x,
        "admissible_by_x": admissible_by_x,
        "precision_by_x": admissible_by_x / accepted_by_x if accepted_by_x > 0 else np.nan,
        "accepted_by_y": accepted_by_y,
        "admissible_by_y": admissible_by_y,
        "precision_by_y": admissible_by_y / accepted_by_y if accepted_by_y > 0 else np.nan,
    }

# ==========================================================
# Experiment
# ==========================================================

def run_experiment(X_train, X_test, y_train, y_test, tau_x, tau_y, N_max, output_noise):
    K_train, _, M_prime_train, Nk_stats_train = compute_statistics(
        X_train, y_train, tau_x=tau_x, tau_y=tau_y, N_max=N_max
    )
    K_test, _, M_prime_test, Nk_stats_test = compute_statistics(
        X_test, y_test, tau_x=tau_x, tau_y=tau_y, N_max=N_max
    )

    compat_train = compatibility_check_bruteforce(
        X_train, y_train, tau_x=tau_x, tau_y=tau_y, output_noise=output_noise
    )
    compat_test = compatibility_check_bruteforce(
        X_test, y_test, tau_x=tau_x, tau_y=tau_y, output_noise=output_noise
    )

    row = {
        "tau_x": tau_x,
        "tau_y": tau_y,
        "N_max": N_max,
        "M_prime_train": M_prime_train,
        "Nk_stats_train": Nk_stats_train,
        "M_prime_test": M_prime_test,
        "Nk_stats_test": Nk_stats_test,
        "Kernel_train": K_train,
        "Kernel_test": K_test,
    }

    row.update({f"train_{k}": v for k, v in compat_train.items()})
    row.update({f"test_{k}": v for k, v in compat_test.items()})
    return row

# ==========================================================
# Main
# ==========================================================

def main():
    np.random.seed(42)

    input_noise = 0.0
    output_noise = 0.03

    X, y = generate_dataset(
        bounds=2,
        n_sqrt_mesh=30,
        input_noise=input_noise,
        output_noise=output_noise
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=42
    )

    tau_x_grid = [0.00001, 0.01, 0.04, 0.1, 0.2]
    tau_y_grid = [0.01, 0.015, 0.02, 0.03]
    N_max_grid = [200]

    results = []
    for tau_x in tau_x_grid:
        for tau_y in tau_y_grid:
            for N_max in N_max_grid:
                results.append(
                    run_experiment(
                        X_train, X_test, y_train, y_test,
                        tau_x=tau_x, tau_y=tau_y, N_max=N_max,
                        output_noise=output_noise
                    )
                )

    df = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(df.round(6))

    df.to_csv(
        "synthetic_regression_compatibility_check.csv",
        index=False,
        float_format="%.6f"
    )

if __name__ == "__main__":
    main()

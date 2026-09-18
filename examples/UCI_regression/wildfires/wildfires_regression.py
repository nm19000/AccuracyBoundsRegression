import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nnx,
    feasible_sets_features_nn_xy,
)

from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)

# ==========================================================
# Dataset loading
# ==========================================================

def load_forest_fires_data(
    csv_path,
    test_size=0.2,
    random_state=42,
    log_target=False,
):
    """
    Load the UCI Forest Fires dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    test_size : float
        Fraction of the data used for testing.
    random_state : int
        Random seed for reproducibility.
    log_target : bool
        If True, use y = log(1 + area).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    df = pd.read_csv(csv_path)

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=["month", "day"],
        dtype=float,
    )

    # Target
    area = df["area"].to_numpy(dtype=float)
    y = np.log1p(area) if log_target else area

    # Features
    X = df.drop(columns=["area"]).to_numpy(dtype=float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Standardize input features using train statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset shape: {df.shape}")
    print(f"X shape:       {X.shape}")
    print(f"y shape:       {y.shape}")
    print(f"X_train:       {X_train.shape}")
    print(f"X_test:        {X_test.shape}")
    print(f"y_train:       {y_train.shape}")
    print(f"y_test:        {y_test.shape}")

    return X_train, X_test, y_train, y_test

# ==========================================================
# Kernel size computation
# ==========================================================

def compute_statistics_input_only(X, y, q, px=1, N_max=300):
    """
    Feasible sets based only on input-space proximity.
    """
    result = feasible_sets_features_nnx(
        X,
        y.reshape(-1, 1),
        N_max=N_max,
        qx=q,
        ext_D=False,
        px=px
    )

    ksize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"],
        metric="l2",
        p=2.0,
    )

    return ksize, result, M_prime, Nk_stats

def compute_statistics_union(X, y, qx, qy=0.2, px=1, N_max=300):
    """
    Feasible sets based on input-space OR target-space proximity.
    """
    result = feasible_sets_features_nn_xy(
        X,
        y.reshape(-1, 1),
        N_max=N_max,
        qx=qx,
        qy=qy,
        ext_D=False,
        px=px
    )

    ksize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"],
        metric="l2",
        p=2.0,
    )

    return ksize, result, M_prime, Nk_stats

# ==========================================================
# Experiment for one q
# ==========================================================

def run_experiment(
    q,
    X_train,
    X_test,
    y_train,
    y_test,
    mode="input_only",
    qy=0.1,
    N_max=300,
):
    if mode == "input_only":
        K_train, train_result, M_prime_train, Nk_stats_train = (
            compute_statistics_input_only(X_train, y_train, q, N_max=N_max)
        )
        K_test, test_result, M_prime_test, Nk_stats_test = (
            compute_statistics_input_only(X_test, y_test, q, N_max=N_max)
        )

        tau_y_train = np.nan
        tau_y_test = np.nan

    elif mode == "union":
        K_train, train_result, M_prime_train, Nk_stats_train = (
            compute_statistics_union(X_train, y_train, qx=q, qy=qy, N_max=N_max)
        )
        K_test, test_result, M_prime_test, Nk_stats_test = (
            compute_statistics_union(X_test, y_test, qx=q, qy=qy, N_max=N_max)
        )

        tau_y_train = train_result["tau_y"]
        tau_y_test = test_result["tau_y"]

    else:
        raise ValueError("mode must be 'input_only' or 'union'")

    row = {
        "mode": mode,
        "q": q,
        "tau_x_train": train_result["tau_x"],
        "tau_x_test": test_result["tau_x"],
        "tau_y_train": tau_y_train,
        "tau_y_test": tau_y_test,
        "M_prime_train": M_prime_train,
        "Nk_stats_train": Nk_stats_train,
        "M_prime_test": M_prime_test,
        "Nk_stats_test": Nk_stats_test,
        "Kernel_train": K_train,
        "Kernel_test": K_test,
        "LB_train": 0.5 * K_train,
        "UB_train": K_train,
        "LB_test": 0.5 * K_test,
        "UB_test": K_test,
    }

    return [row]

# ==========================================================
# Main
# ==========================================================

def main():

    np.random.seed(42)

    csv_path = "forestfires.csv"

    X_train, X_test, y_train, y_test = load_forest_fires_data(
        csv_path,
        test_size=0.5,
        random_state=42,
        log_target=False,
    )

    qs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    results = []

    # First pass: input-only feasible sets
    for q in qs:
        results.extend(
            run_experiment(
                q,
                X_train,
                X_test,
                y_train,
                y_test,
                mode="input_only",
                N_max=300,
            )
        )

    # Optional second pass: union-rule feasible sets
    for q in qs:
        results.extend(
            run_experiment(
                q,
                X_train,
                X_test,
                y_train,
                y_test,
                mode="union",
                qy=0.2,
                N_max=400,
            )
        )

    df = pd.DataFrame(results)

    round_cols = [
        "tau_x_train",
        "tau_x_test",
        "tau_y_train",
        "tau_y_test",
        "Kernel_train",
        "Kernel_test",
        "LB_train",
        "UB_train",
        "LB_test",
        "UB_test",
    ]

    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].round(6)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(df)

    df.to_csv(
        "forestfires_kernel_results.csv",
        index=False,
        float_format="%.6f",
    )

if __name__ == "__main__":
    main()

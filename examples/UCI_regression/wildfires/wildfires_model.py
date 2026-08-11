import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nnx,
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
    log_target=True,
):
    """
    Load the UCI Forest Fires dataset using weather-only inputs:
        temp, RH, wind, rain

    Returns:
        X_full_raw, y_full,
        X_train, X_test, y_train, y_test

    where:
      - X_full_raw is the raw full feature matrix (for full-dataset kernel computation)
      - X_train/X_test are standardized using train-only statistics (for modeling)
    """
    df = pd.read_csv(csv_path)

    area = df["area"].to_numpy(dtype=float)
    y_full = np.log1p(area) if log_target else area

    feature_cols = ["temp", "RH", "wind", "rain"]
    X_full_raw = df[feature_cols].to_numpy(dtype=float)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full_raw,
        y_full,
        test_size=test_size,
        random_state=random_state,
    )

    # Standardize for model training/evaluation using train-only stats
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(X_train_raw)
    X_test = scaler_train.transform(X_test_raw)

    print(f"Dataset shape: {df.shape}")
    print(f"Using features: {feature_cols}")
    print(f"X_full shape:   {X_full_raw.shape}")
    print(f"y_full shape:   {y_full.shape}")
    print(f"X_train:        {X_train.shape}")
    print(f"X_test:         {X_test.shape}")
    print(f"y_train:        {y_train.shape}")
    print(f"y_test:         {y_test.shape}")
    print(f"log_target:     {log_target}")

    return X_full_raw, y_full, X_train, X_test, y_train, y_test

# ==========================================================
# Kernel size computation
# ==========================================================

def compute_statistics_input_only(X, y, q, N_max=300):
    """
    Feasible sets based only on input-space proximity.
    """
    result = feasible_sets_features_nnx(
        X,
        y.reshape(-1, 1),
        N_max=N_max,
        qx=q,
        ext_D=False,
    )

    ksize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"],
        metric="l2",
        p=2.0,
    )

    return ksize, result, M_prime, Nk_stats

# ==========================================================
# Metrics / smoothness estimate
# ==========================================================

def rmse(y, yp):
    return np.sqrt(mean_squared_error(y, yp))

def empirical_lipschitz_constant(
    model,
    X,
    n_pairs=20000,
    eps=1e-12,
    q=0.99,
):
    """
    Empirical Lipschitz estimate from random pairs in X,
    using a quantile instead of the maximum for robustness.
    """
    y_pred = model.predict(X).reshape(-1, 1)

    n = len(X)
    i = np.random.randint(0, n, n_pairs)
    j = np.random.randint(0, n, n_pairs)

    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dy = np.abs(y_pred[i] - y_pred[j]).flatten()

    mask = dx > eps
    if not np.any(mask):
        return np.nan

    ratios = dy[mask] / dx[mask]
    return np.quantile(ratios, q)

# ==========================================================
# Candidate models
# ==========================================================

def build_candidates(name):
    """
    Return candidate models.
    Inputs are already standardized, so no Pipeline scaler here.
    """
    models = []

    if name == "MLP":
        for layers in [
            (32,),
            (64,),
            (128,),
            (64, 64),
            (128, 128),
        ]:
            for alpha in [1e-6, 1e-5, 1e-4]:
                models.append(
                    MLPRegressor(
                        hidden_layer_sizes=layers,
                        alpha=alpha,
                        max_iter=20000,
                        learning_rate_init=1e-3,
                        random_state=42,
                    )
                )

    elif name == "SVR":
        for C in [1, 10, 100, 1000]:
            for eps in [1e-3, 1e-2, 1e-1]:
                for gamma in ["scale", 0.1, 1, 10]:
                    models.append(
                        SVR(
                            kernel="rbf",
                            C=C,
                            epsilon=eps,
                            gamma=gamma,
                        )
                    )

    elif name == "GP":
        for length in [0.2, 0.5, 1.0, 2.0]:
            for noise in [1e-3, 1e-2, 5e-2]:
                kernel = (
                    ConstantKernel(1.0)
                    * RBF(length_scale=length)
                    + WhiteKernel(noise)
                )
                models.append(
                    GaussianProcessRegressor(
                        kernel=kernel,
                        normalize_y=True,
                        optimizer=None,
                        random_state=42,
                    )
                )

    else:
        raise ValueError(name)

    return models

# ==========================================================
# Model selection in the band [0.5K, K]
# ==========================================================

def train_to_target(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    K,
    mode,
    X_lip,
):
    """
    Select a candidate constrained to the band

        0.5 * K <= RMSE_train <= K.

    mode == "accurate_in_band":
        among candidates in the band, choose the one with smallest train RMSE.

    mode == "smooth_in_band":
        among candidates in the band, choose the one with smallest Lipschitz estimate.

    Fallback:
        if no candidate lies in the band, choose the one closest to the midpoint 0.75*K.
        For smooth_in_band, ties are broken by smaller Lipschitz.
    """
    candidates = build_candidates(model_name)

    lower = 0.5 * K
    upper = K
    midpoint = 0.75 * K

    fitted_models = []

    for model in candidates:
        model.fit(X_train, y_train)

        train_error = rmse(y_train, model.predict(X_train))
        test_error = rmse(y_test, model.predict(X_test))
        L = empirical_lipschitz_constant(model, X_lip)

        fitted_models.append({
            "model": model,
            "train_error": train_error,
            "test_error": test_error,
            "L": L,
        })

    in_band_models = [
        d for d in fitted_models
        if lower <= d["train_error"] <= upper
    ]

    used_fallback = False

    if len(in_band_models) == 0:
        used_fallback = True
        print(
            f"Warning: {model_name} has no candidate with "
            f"{lower:.6f} <= train RMSE <= {upper:.6f}. "
            f"Using closest-to-midpoint fallback."
        )

        if mode == "accurate_in_band":
            best = min(
                fitted_models,
                key=lambda d: abs(d["train_error"] - midpoint),
            )
        elif mode == "smooth_in_band":
            best = min(
                fitted_models,
                key=lambda d: (abs(d["train_error"] - midpoint), d["L"]),
            )
        else:
            raise ValueError(
                "mode must be 'accurate_in_band' or 'smooth_in_band'"
            )
    else:
        if mode == "accurate_in_band":
            best = min(in_band_models, key=lambda d: d["train_error"])
        elif mode == "smooth_in_band":
            best = min(in_band_models, key=lambda d: d["L"])
        else:
            raise ValueError(
                "mode must be 'accurate_in_band' or 'smooth_in_band'"
            )

    in_band = lower <= best["train_error"] <= upper

    return (
        best["train_error"],
        best["test_error"],
        best["L"],
        in_band,
        used_fallback,
    )

# ==========================================================
# Experiment for one q
# ==========================================================

def run_experiment(
    q,
    X_full_raw,
    y_full,
    X_train,
    X_test,
    y_train,
    y_test,
    N_max=300,
    model_names=("GP", "MLP", "SVR"),
):
    # Standardize full dataset separately for kernel computation
    scaler_full = StandardScaler()
    X_full_kernel = scaler_full.fit_transform(X_full_raw)

    K_full, full_result, M_prime_full, Nk_stats_full = (
        compute_statistics_input_only(X_full_kernel, y_full, q, N_max=N_max)
    )

    X_lip = X_train
    rows = []

    for name in model_names:
        print(f"q={q}, training {name}")

        train_o, test_o, L_o, in_band_o, fallback_o = train_to_target(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            K=K_full,
            mode="accurate_in_band",
            X_lip=X_lip,
        )

        rows.append({
            "mode": "input_only_full_kernel",
            "q": q,
            "Model": name + "^o",
            "selection_mode": "accurate_in_band",
            "tau_x_full": full_result["tau_x"],
            "M_prime_full": M_prime_full,
            "Nk_stats_full": Nk_stats_full,
            "Kernel_full": K_full,
            "LB_full": 0.5 * K_full,
            "UB_full": K_full,
            "RMSE_train": train_o,
            "RMSE_test": test_o,
            "Lhat": L_o,
            "in_band": in_band_o,
            "used_fallback": fallback_o,
        })

        train_s, test_s, L_s, in_band_s, fallback_s = train_to_target(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            K=K_full,
            mode="smooth_in_band",
            X_lip=X_lip,
        )

        rows.append({
            "mode": "input_only_full_kernel",
            "q": q,
            "Model": name,
            "selection_mode": "smooth_in_band",
            "tau_x_full": full_result["tau_x"],
            "M_prime_full": M_prime_full,
            "Nk_stats_full": Nk_stats_full,
            "Kernel_full": K_full,
            "LB_full": 0.5 * K_full,
            "UB_full": K_full,
            "RMSE_train": train_s,
            "RMSE_test": test_s,
            "Lhat": L_s,
            "in_band": in_band_s,
            "used_fallback": fallback_s,
        })

    return rows

# ==========================================================
# Main
# ==========================================================

def main():
    np.random.seed(42)

    csv_path = "forestfires.csv"

    X_full_raw, y_full, X_train, X_test, y_train, y_test = load_forest_fires_data(
        csv_path,
        test_size=0.3,
        random_state=42,
        log_target=True,
    )

    qs = [0.20]
    results = []

    for q in qs:
        results.extend(
            run_experiment(
                q,
                X_full_raw,
                y_full,
                X_train,
                X_test,
                y_train,
                y_test,
                N_max=300,
                model_names=("GP", "MLP", "SVR"),
            )
        )

    df = pd.DataFrame(results)

    round_cols = [
        "tau_x_full",
        "Kernel_full",
        "LB_full",
        "UB_full",
        "RMSE_train",
        "RMSE_test",
        "Lhat",
    ]

    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].round(6)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(df)

    df.to_csv(
        "forestfires_weather_only_full_kernel_model_results_band.csv",
        index=False,
        float_format="%.6f",
    )

if __name__ == "__main__":
    main()

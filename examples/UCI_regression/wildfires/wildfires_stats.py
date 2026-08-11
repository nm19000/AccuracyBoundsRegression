import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nnx,
)

from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)

# ==========================================================
# CONFIG
# ==========================================================

CSV_PATH = "forestfires.csv"
LOG_TARGET = True

# Weather-only features, as in your earlier script
FEATURE_COLS = ["temp", "RH", "wind", "rain"]

# Ablation grids
Q_GRID = np.round(np.arange(0.01, 0.11, 0.01), 2)
NMAX_GRID = [25, 50, 100, 150, 200, 300, 400]

# Set this if you want to use only one q or one N_max
# Q_GRID = [0.05]
# NMAX_GRID = [300]

OUTPUT_CSV = "forestfires_feasible_set_ablation_q_nmax.csv"

# ==========================================================
# Dataset loading
# ==========================================================

def load_forest_fires_full_dataset(
    csv_path,
    feature_cols,
    log_target=True,
):
    df = pd.read_csv(csv_path)

    area = df["area"].to_numpy(dtype=float)
    y = np.log1p(area) if log_target else area
    X_raw = df[feature_cols].to_numpy(dtype=float)

    # Standardize on the full dataset for full-dataset kernel analysis
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"Dataset shape: {df.shape}")
    print(f"Using features: {feature_cols}")
    print(f"X shape:        {X.shape}")
    print(f"y shape:        {y.shape}")
    print(f"log_target:     {log_target}")

    return X, y, df

# ==========================================================
# Core feasible-set statistics
# ==========================================================

def compute_statistics_input_only(X, y, q, N_max):
    """
    Input-only feasible sets:
        d_X(x_{k,n}, x_k) <= tau_X(q)

    using feasible_sets_features_nnx.
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

    row = {
        "mode": "input_only",
        "q": q,
        "N_max": N_max,
        "tau_x": result.get("tau_x", np.nan),
        "M_prime": M_prime,
        "Nk_stats": str(Nk_stats),
        "Kernel_size": ksize,
    }

    return row, result

# ==========================================================
# Optional diagnostics for q-stability
# ==========================================================

def add_stability_columns(df):
    """
    Add relative change and a simple discrete curvature proxy
    separately within each fixed N_max.
    """
    out = []

    for nmax, g in df.groupby("N_max", sort=True):
        g = g.sort_values("q").copy()

        g["rel_change_K"] = g["Kernel_size"].pct_change().abs()

        K = g["Kernel_size"].to_numpy()
        curvature = [np.nan]
        for i in range(1, len(K) - 1):
            curvature.append(abs(K[i + 1] - 2 * K[i] + K[i - 1]))
        curvature.append(np.nan)

        g["local_curvature"] = curvature
        out.append(g)

    return pd.concat(out, axis=0).sort_values(["N_max", "q"]).reset_index(drop=True)

# ==========================================================
# Main ablation loop
# ==========================================================

def run_ablation(
    X,
    y,
    q_grid,
    nmax_grid,
):
    rows = []

    for N_max in nmax_grid:
        for q in q_grid:
            print(f"Running input_only ablation with q={q:.2f}, N_max={N_max}")
            row, _ = compute_statistics_input_only(
                X=X,
                y=y,
                q=q,
                N_max=N_max,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df = add_stability_columns(df)
    return df

# ==========================================================
# Main
# ==========================================================

def main():
    np.random.seed(42)

    X, y, df_raw = load_forest_fires_full_dataset(
        CSV_PATH,
        feature_cols=FEATURE_COLS,
        log_target=LOG_TARGET,
    )

    df_results = run_ablation(
        X=X,
        y=y,
        q_grid=Q_GRID,
        nmax_grid=NMAX_GRID,
    )

    round_cols = [
        "tau_x",
        "Kernel_size",
        "rel_change_K",
        "local_curvature",
    ]

    for col in round_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(6)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\n===== FEASIBLE-SET ABLATION RESULTS =====")
    print(df_results)

    df_results.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.6f",
    )

if __name__ == "__main__":
    main()

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nnx,
)

from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)

def compute_R_squared(feasible_sets):
    """ Compute the R-squared value for a collection of feasible sets. Which is the optimal RMSE, only on D_prime and for the l2 norm."""
    return sum(
        np.sum((Y - Y.mean(axis=0))**2)
        for Y in feasible_sets if len(Y)
    ) / sum(len(Y) for Y in feasible_sets)
# ==========================================================
# CONFIG
# ==========================================================

LOCAL_XLSX = "ENB2012_data.xlsx"

FEATURE_COLS = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8",
]

TARGET_COLS = {
    "Y1": "Heating Load",
    "Y2": "Cooling Load",
}

LOG_TARGET = False

Q_GRID = np.round(np.arange(0.01, 0.11, 0.01), 2)

NMAX_GRID = [
    10,
    20,
    30,
    50,
    70,
    100,
    200,
    300,
]

REFERENCE_NMAX = 300

TAU_Y = 0.0

REL_CHANGE_THRESHOLD = 0.05
CURVATURE_THRESHOLD = 0.05
MIN_PLATEAU_LENGTH = 3

NMAX_SATURATION_TOL = 0.01

OUTPUT_Q_CSV = "energy_efficiency_q_ablation_xy.csv"
OUTPUT_NMAX_CSV = "energy_efficiency_kernel_size_nmax_ablation_xy.csv"
SUMMARY_CSV = "energy_efficiency_average_kernel_size_xy.csv"
Q_SUMMARY_CSV = "energy_efficiency_q_stability_summary_xy.csv"
SELECTED_Q_CSV = "energy_efficiency_selected_q_xy.csv"
NMAX_SUMMARY_CSV = "energy_efficiency_nmax_saturation_xy.csv"
PLOT_DIR = "energy_efficiency_kernel_size_plots_xy"

# ==========================================================
# Dataset loading
# ==========================================================

def load_energy_efficiency_dataset(
    local_path,
    feature_cols,
    target_cols,
):
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"\nCould not find the dataset:\n"
            f"    {local_path}\n\n"
            f"Make sure ENB2012_data.xlsx is in the same folder as this script."
        )

    print(f"Loading dataset: {local_path}")
    df = pd.read_excel(local_path)
    df.columns = [str(col).strip() for col in df.columns]

    print(f"Dataset shape: {df.shape}")
    print(f"Columns found: {df.columns.tolist()}")

    required_cols = feature_cols + list(target_cols.keys())
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X_raw = df[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    targets = {}
    for target_col in target_cols:
        y = df[target_col].to_numpy(dtype=float).reshape(-1, 1)
        if LOG_TARGET:
            y = np.log1p(y)
        targets[target_col] = y

    print("\n===== DATASET =====")
    print(f"Dataset shape: {df.shape}")
    print(f"Using features: {feature_cols}")
    print(f"X shape: {X.shape}")

    for target_col, target_name in target_cols.items():
        y = targets[target_col]
        print(f"{target_col} ({target_name}) shape: {y.shape}")
        print(f"    min  = {y.min():.6f}")
        print(f"    max  = {y.max():.6f}")
        print(f"    mean = {y.mean():.6f}")

    print(f"log_target: {LOG_TARGET}")

    return X, targets, df

# ==========================================================
# Kernel-size extraction
# ==========================================================

def extract_kernel_statistics(feasible_sets):
    """
    Uses the signature previously used in your scripts:
        compute_kernel_size(feasible_sets, metric="l2", p=2.0)
    returning:
        kernel_size, M_prime, Nk_stats
    """
    kernel_size, M_prime, Nk_stats = compute_kernel_size(
        feasible_sets,
        metric="l2",
        p=2.0,
    )

    feasible_sizes = np.asarray(
        [len(fs) for fs in feasible_sets],
        dtype=int,
    )

    return {
        "Kernel_size": float(kernel_size),
        "M_prime": M_prime,
        "Nk_stats": Nk_stats,
        "mean_feasible_set_size": float(np.mean(feasible_sizes)),
        "median_feasible_set_size": float(np.median(feasible_sizes)),
        "min_feasible_set_size": int(np.min(feasible_sizes)),
        "max_feasible_set_size": int(np.max(feasible_sizes)),
    }

# ==========================================================
# Core feasible-set statistics
# ==========================================================

def compute_statistics_xy(
    X,
    y,
    q,
    N_max,
    target_name,
):
    """
    Feasible-set definition:
        d_X(x_k, x_j) <= tau_x OR d_Y(y_k, y_j) <= tau_y

    Here:
        - q controls qx, hence tau_x
        - qy is held fixed at QY_FIXED unless TAU_Y > 0
    """
    try:
        result = feasible_sets_features_nnx(
            X,                  # features_x
            y,                  # features_y
            N_max=N_max,
            qx=q,
            ext_D=False,
        )
    except TypeError as e:
        raise TypeError(
            "Call to feasible_sets_features_nnx(...) failed.\n"
            f"Original error: {e}"
        )

    feasible_sets = result["feasible_sets"]

    statistics = extract_kernel_statistics(feasible_sets)

    feasible_sizes = np.asarray(
        [len(fs) for fs in feasible_sets],
        dtype=int,
    )

    R = np.sqrt(compute_R_squared(feasible_sets))

    truncation_limit = min(N_max + 1, len(y))

    fraction_at_Nmax = float(
        np.mean(feasible_sizes >= truncation_limit)
    )

    return {
        "target": target_name,
        "q": float(q),
        "N_max": int(N_max),
        "tau_x": float(result.get("tau_x", np.nan)),
        "M_prime": statistics["M_prime"],
        "Nk_stats": str(statistics["Nk_stats"]),
        "Kernel_size": statistics["Kernel_size"],
        "R": R,
        "mean_feasible_set_size": statistics["mean_feasible_set_size"],
        "median_feasible_set_size": statistics["median_feasible_set_size"],
        "min_feasible_set_size": statistics["min_feasible_set_size"],
        "max_feasible_set_size": statistics["max_feasible_set_size"],
        "fraction_at_Nmax": fraction_at_Nmax,
    }

# ==========================================================
# Stability diagnostics
# ==========================================================

def add_stability_columns(df):
    out = []

    for target, g in df.groupby("target", sort=True):
        g = g.sort_values("q").copy()
        K = g["Kernel_size"].to_numpy(dtype=float)

        rel_change = np.full(len(K), np.nan, dtype=float)
        for i in range(1, len(K)):
            rel_change[i] = (
                abs(K[i] - K[i - 1]) / max(abs(K[i - 1]), 1e-12)
            )

        curvature = np.full(len(K), np.nan, dtype=float)
        for i in range(1, len(K) - 1):
            curvature[i] = (
                abs(K[i + 1] - 2.0 * K[i] + K[i - 1]) / max(abs(K[i]), 1e-12)
            )

        g["rel_change_K"] = rel_change
        g["normalized_local_curvature"] = curvature

        out.append(g)

    return (
        pd.concat(out, axis=0)
        .sort_values(["target", "q"])
        .reset_index(drop=True)
    )

# ==========================================================
# Stage 1: q ablation at N_max = 300
# ==========================================================

def run_q_ablation(
    X,
    targets,
    q_grid,
    reference_nmax,
    target_cols,
):
    rows = []

    print("\n" + "=" * 70)
    print("STAGE 1: q ABLATION")
    print("=" * 70)
    print(f"q grid: {q_grid}")
    print(f"Fixed N_max: {reference_nmax}")

    for target_col, target_name in target_cols.items():
        y = targets[target_col]

        print("\n" + "-" * 70)
        print(f"TARGET: {target_col} - {target_name}")
        print("-" * 70)

        for q in q_grid:
            print(f"    q={q:.2f}, N_max={reference_nmax}")
            row = compute_statistics_xy(
                X=X,
                y=y,
                q=q,
                N_max=reference_nmax,
                target_name=target_col,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df = add_stability_columns(df)
    return df

# ==========================================================
# Stable-q selection
# ==========================================================

def select_stable_q(
    q_summary,
    rel_change_threshold,
    curvature_threshold,
    min_plateau_length,
):
    selected_rows = []

    for target in sorted(q_summary["target"].unique()):
        g = (
            q_summary[q_summary["target"] == target]
            .sort_values("q")
            .reset_index(drop=True)
            .copy()
        )

        q_values = g["q"].to_numpy(dtype=float)
        rel_change = g["rel_change_K"].to_numpy(dtype=float)
        curvature = g["normalized_local_curvature"].to_numpy(dtype=float)

        stable = (
            np.isfinite(rel_change)
            & np.isfinite(curvature)
            & (rel_change <= rel_change_threshold)
            & (curvature <= curvature_threshold)
        )

        g["stable"] = stable

        selected_q = None
        plateau_start = np.nan
        plateau_end = np.nan
        plateau_length = 0

        i = 0
        while i < len(stable):
            if not stable[i]:
                i += 1
                continue

            start = i
            while i + 1 < len(stable) and stable[i + 1]:
                i += 1
            end = i

            current_length = end - start + 1
            if current_length >= min_plateau_length:
                selected_q = q_values[start]
                plateau_start = q_values[start]
                plateau_end = q_values[end]
                plateau_length = current_length
                break

            i += 1

        fallback_used = False

        if selected_q is None:
            fallback_used = True
            score = (
                np.nan_to_num(rel_change, nan=np.inf)
                + np.nan_to_num(curvature, nan=np.inf)
            )

            if not np.any(np.isfinite(score)):
                best_idx = 0
            else:
                best_idx = int(np.nanargmin(score))

            selected_q = q_values[best_idx]

        selected = g[np.isclose(g["q"], selected_q)].iloc[0]

        selected_rows.append(
            {
                "target": target,
                "selected_q": selected_q,
                "tau_x": selected["tau_x"],
                "Kernel_size": selected["Kernel_size"],
                "M_prime": selected["M_prime"],
                "mean_feasible_set_size": selected["mean_feasible_set_size"],
                "median_feasible_set_size": selected["median_feasible_set_size"],
                "min_feasible_set_size": selected["min_feasible_set_size"],
                "max_feasible_set_size": selected["max_feasible_set_size"],
                "rel_change_K": selected["rel_change_K"],
                "normalized_local_curvature": selected["normalized_local_curvature"],
                "plateau_start_q": plateau_start,
                "plateau_end_q": plateau_end,
                "plateau_length": plateau_length,
                "fallback_used": fallback_used,
            }
        )

    return pd.DataFrame(selected_rows)

# ==========================================================
# Stage 2: N_max ablation at selected q
# ==========================================================

def run_nmax_ablation(
    X,
    targets,
    selected_q,
    nmax_grid,
    target_cols,
):
    rows = []

    print("\n" + "=" * 70)
    print("STAGE 2: N_max / KERNEL-SIZE ABLATION")
    print("=" * 70)
    print(f"N_max grid: {nmax_grid}")

    for target_col, target_name in target_cols.items():
        q_star = float(
            selected_q.loc[
                selected_q["target"] == target_col,
                "selected_q",
            ].iloc[0]
        )

        y = targets[target_col]

        print("\n" + "-" * 70)
        print(f"TARGET: {target_col} - {target_name}")
        print(f"Selected q = {q_star:.2f}")
        print("-" * 70)

        for N_max in nmax_grid:
            print(f"    q={q_star:.2f}, N_max={N_max}")
            row = compute_statistics_xy(
                X=X,
                y=y,
                q=q_star,
                N_max=N_max,
                target_name=target_col,
            )
            row["selected_q"] = q_star
            rows.append(row)

    return pd.DataFrame(rows)

# ==========================================================
# N_max saturation analysis
# ==========================================================

def compute_nmax_saturation(
    df,
    reference_nmax,
):
    rows = []

    for target in sorted(df["target"].unique()):
        g = (
            df[df["target"] == target]
            .sort_values("N_max")
            .copy()
        )

        reference_rows = g[g["N_max"] == reference_nmax]
        if reference_rows.empty:
            raise ValueError(
                f"Reference N_max={reference_nmax} not found for target={target}."
            )

        reference_kernel_size = float(reference_rows["Kernel_size"].iloc[0])
        q_star = float(reference_rows["q"].iloc[0])

        for _, row in g.iterrows():
            relative_to_reference = (
                abs(row["Kernel_size"] - reference_kernel_size)
                / max(abs(reference_kernel_size), 1e-12)
            )

            saturated = relative_to_reference <= NMAX_SATURATION_TOL

            rows.append(
                {
                    "target": target,
                    "selected_q": q_star,
                    "N_max": row["N_max"],
                    "Kernel_size": row["Kernel_size"],
                    "tau_x": row["tau_x"],
                    "mean_feasible_set_size": row["mean_feasible_set_size"],
                    "max_feasible_set_size": row["max_feasible_set_size"],
                    "fraction_at_Nmax": row["fraction_at_Nmax"],
                    "relative_to_reference": relative_to_reference,
                    "saturated": saturated,
                }
            )

    return pd.DataFrame(rows)

# ==========================================================
# Plot helpers
# ==========================================================

def plot_kernel_size_vs_q(
    q_summary,
    selected_q,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    for target in q_summary["target"].unique():
        g = q_summary[q_summary["target"] == target].sort_values("q")

        plt.figure(figsize=(9, 6))
        plt.plot(
            g["q"],
            g["Kernel_size"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color="tab:blue",
            label=f"N_max={REFERENCE_NMAX}",
        )

        selected_row = selected_q[selected_q["target"] == target]
        if not selected_row.empty:
            q_star = float(selected_row["selected_q"].iloc[0])
            k_star = float(selected_row["Kernel_size"].iloc[0])

            plt.scatter(
                [q_star],
                [k_star],
                s=100,
                color="red",
                zorder=5,
                label=f"selected q={q_star:.2f}",
            )
            plt.axvline(q_star, color="red", linestyle="--", alpha=0.5)

        target_name = TARGET_COLS[target]

        plt.xlabel("q", fontsize=12)
        plt.ylabel("Kernel size", fontsize=12)
        plt.title(f"{target}: Kernel Size vs q ({target_name})", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{target}_kernel_size_vs_q.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {filename}")

def plot_q_stability(
    q_summary,
    selected_q,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    for target in q_summary["target"].unique():
        g = q_summary[q_summary["target"] == target].sort_values("q")

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        axes[0].plot(
            g["q"],
            g["rel_change_K"],
            marker="o",
            linewidth=1.8,
            color="tab:blue",
        )
        axes[0].axhline(
            REL_CHANGE_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"threshold={REL_CHANGE_THRESHOLD:.2f}",
        )
        axes[0].set_ylabel("Relative change in K", fontsize=11)
        axes[0].set_title(
            f"{target}: q-stability diagnostics (N_max={REFERENCE_NMAX})",
            fontsize=13,
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            g["q"],
            g["normalized_local_curvature"],
            marker="o",
            linewidth=1.8,
            color="tab:orange",
        )
        axes[1].axhline(
            CURVATURE_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"threshold={CURVATURE_THRESHOLD:.2f}",
        )

        selected_row = selected_q[selected_q["target"] == target]
        if not selected_row.empty:
            q_star = float(selected_row["selected_q"].iloc[0])
            axes[0].axvline(q_star, color="green", linestyle=":", alpha=0.8)
            axes[1].axvline(
                q_star,
                color="green",
                linestyle=":",
                alpha=0.8,
                label=f"selected q={q_star:.2f}",
            )
            axes[1].legend()

        axes[1].set_xlabel("q", fontsize=11)
        axes[1].set_ylabel("Normalized curvature", fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = os.path.join(output_dir, f"{target}_q_stability.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {filename}")

def plot_kernel_size_vs_nmax(
    nmax_results,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    for target in nmax_results["target"].unique():
        g = nmax_results[nmax_results["target"] == target].sort_values("N_max")
        q_star = float(g["q"].iloc[0])

        plt.figure(figsize=(9, 6))
        plt.plot(
            g["N_max"],
            g["Kernel_size"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color="tab:green",
        )
        plt.axvline(
            REFERENCE_NMAX,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"reference N_max={REFERENCE_NMAX}",
        )

        target_name = TARGET_COLS[target]

        plt.xlabel("N_max", fontsize=12)
        plt.ylabel("Kernel size", fontsize=12)
        plt.title(
            f"{target}: Kernel Size vs N_max at selected q={q_star:.2f} ({target_name})",
            fontsize=13,
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{target}_kernel_size_vs_nmax.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {filename}")

def plot_feasible_set_size_vs_q(
    q_summary,
    selected_q,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    for target in q_summary["target"].unique():
        g = q_summary[q_summary["target"] == target].sort_values("q")

        plt.figure(figsize=(9, 6))
        plt.plot(
            g["q"],
            g["mean_feasible_set_size"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color="tab:purple",
        )

        selected_row = selected_q[selected_q["target"] == target]
        if not selected_row.empty:
            q_star = float(selected_row["selected_q"].iloc[0])
            plt.axvline(
                q_star,
                color="red",
                linestyle="--",
                alpha=0.6,
                label=f"selected q={q_star:.2f}",
            )

        target_name = TARGET_COLS[target]

        plt.xlabel("q", fontsize=12)
        plt.ylabel("Mean feasible-set size", fontsize=12)
        plt.title(f"{target}: Feasible-Set Size vs q ({target_name})", fontsize=13)
        plt.grid(True, alpha=0.3)

        if not selected_row.empty:
            plt.legend()

        plt.tight_layout()

        filename = os.path.join(output_dir, f"{target}_feasible_set_size_vs_q.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {filename}")

def plot_average_kernel_size(
    summary,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    labels = [
        f"{target}\n{TARGET_COLS[target]}"
        for target in summary["target"]
    ]

    plt.figure(figsize=(8, 6))
    colors = ["tab:blue", "tab:orange"]

    plt.bar(
        labels,
        summary["average_kernel_size"],
        color=colors[:len(summary)],
    )

    plt.ylabel("Average kernel size", fontsize=12)
    plt.title("Average Kernel Size Across N_max at Selected q", fontsize=13)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(output_dir, "average_kernel_size_by_target.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {filename}")

# ==========================================================
# Main
# ==========================================================

def main():
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\n" + "=" * 70)
    print("FEASIBLE-SET CONFIGURATION")
    print("=" * 70)
    print("Feasible-set definition:")
    print("    d_X(x_k,x_j) <= tau_x OR d_Y(y_k,y_j) <= tau_y")
    print(f"Q_GRID: {Q_GRID}")
    print(f"REFERENCE_NMAX: {REFERENCE_NMAX}")
    print(f"FINAL NMAX_GRID: {NMAX_GRID}")
    print(f"TAU_Y: {TAU_Y}")
    print("\nSelection rule:")
    print(f"    relative change <= {REL_CHANGE_THRESHOLD:.3f}")
    print(f"    curvature <= {CURVATURE_THRESHOLD:.3f}")
    print(f"    minimum plateau length = {MIN_PLATEAU_LENGTH}")

    X, targets, df_raw = load_energy_efficiency_dataset(
        local_path=LOCAL_XLSX,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
    )

    q_results = run_q_ablation(
        X=X,
        targets=targets,
        q_grid=Q_GRID,
        reference_nmax=REFERENCE_NMAX,
        target_cols=TARGET_COLS,
    )

    q_results = q_results.round(6)
    q_results.to_csv(OUTPUT_Q_CSV, index=False, float_format="%.6f")

    print("\n" + "=" * 70)
    print("q ABLATION RESULTS SAVED")
    print("=" * 70)
    print(f"    {OUTPUT_Q_CSV}")

    q_summary = q_results.sort_values(["target", "q"]).copy()
    q_summary.to_csv(Q_SUMMARY_CSV, index=False, float_format="%.6f")

    print("\nq stability summary saved to:")
    print(f"    {Q_SUMMARY_CSV}")

    selected_q = select_stable_q(
        q_summary=q_summary,
        rel_change_threshold=REL_CHANGE_THRESHOLD,
        curvature_threshold=CURVATURE_THRESHOLD,
        min_plateau_length=MIN_PLATEAU_LENGTH,
    )

    selected_q = selected_q.round(6)
    selected_q.to_csv(SELECTED_Q_CSV, index=False, float_format="%.6f")

    print("\n" + "=" * 70)
    print("SELECTED STABLE q")
    print("=" * 70)
    print(f"q selection N_max: {REFERENCE_NMAX}")
    print(f"Relative-change threshold: {REL_CHANGE_THRESHOLD:.3f}")
    print(f"Curvature threshold: {CURVATURE_THRESHOLD:.3f}")
    print(f"Minimum plateau length: {MIN_PLATEAU_LENGTH}")
    print("\n")
    print(selected_q.to_string(index=False))

    nmax_results = run_nmax_ablation(
        X=X,
        targets=targets,
        selected_q=selected_q,
        nmax_grid=NMAX_GRID,
        target_cols=TARGET_COLS,
    )

    nmax_results = nmax_results.round(6)
    nmax_results.to_csv(OUTPUT_NMAX_CSV, index=False, float_format="%.6f")

    print("\n" + "=" * 70)
    print("N_max / KERNEL-SIZE ABLATION SAVED")
    print("=" * 70)
    print(f"    {OUTPUT_NMAX_CSV}")

    nmax_summary = compute_nmax_saturation(
        df=nmax_results,
        reference_nmax=REFERENCE_NMAX,
    )

    nmax_summary = nmax_summary.round(6)
    nmax_summary.to_csv(NMAX_SUMMARY_CSV, index=False, float_format="%.6f")

    print("\nN_max saturation summary saved to:")
    print(f"    {NMAX_SUMMARY_CSV}")


    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\n" + "=" * 70)
    print(f"q ABLATION (N_max={REFERENCE_NMAX})")
    print("=" * 70)
    print(q_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("SELECTED CONFIGURATIONS")
    print("=" * 70)
    print(selected_q.to_string(index=False))

    print("\n" + "=" * 70)
    print("KERNEL-SIZE ABLATION OVER N_max")
    print("=" * 70)
    print(nmax_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("N_max SATURATION ANALYSIS")
    print("=" * 70)
    print(nmax_summary.to_string(index=False))


    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_kernel_size_vs_q(
        q_summary=q_summary,
        selected_q=selected_q,
        output_dir=PLOT_DIR,
    )

    plot_q_stability(
        q_summary=q_summary,
        selected_q=selected_q,
        output_dir=PLOT_DIR,
    )

    plot_kernel_size_vs_nmax(
        nmax_results=nmax_results,
        output_dir=PLOT_DIR,
    )

    plot_feasible_set_size_vs_q(
        q_summary=q_summary,
        selected_q=selected_q,
        output_dir=PLOT_DIR,
    )


    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"q ablation CSV: {OUTPUT_Q_CSV}")
    print(f"Final N_max ablation CSV: {OUTPUT_NMAX_CSV}")
    print(f"q stability CSV: {Q_SUMMARY_CSV}")
    print(f"Selected q CSV: {SELECTED_Q_CSV}")
    print(f"N_max saturation CSV: {NMAX_SUMMARY_CSV}")
    print(f"Plots: {PLOT_DIR}/")

    print("\nSelected q values:")
    for _, row in selected_q.iterrows():
        plateau_start = (
            f"{row['plateau_start_q']:.2f}"
            if pd.notna(row["plateau_start_q"]) else "nan"
        )
        plateau_end = (
            f"{row['plateau_end_q']:.2f}"
            if pd.notna(row["plateau_end_q"]) else "nan"
        )

        print(
            f"    {row['target']}: "
            f"q={row['selected_q']:.2f}, "
            f"tau_x={row['tau_x']:.6f}, "
            f"Kernel_size={row['Kernel_size']:.6f}, "
            f"plateau=[{plateau_start}, {plateau_end}], "
            f"length={int(row['plateau_length'])}, "
            f"fallback_used={bool(row['fallback_used'])}"
        )

if __name__ == "__main__":
    main()
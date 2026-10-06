
import os
import warnings
import inspect

import numpy as np
import pandas as pd

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
# CONFIGURATION
# ==========================================================

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

LOCAL_XLS = "Concrete_Data.xls"

# ----------------------------------------------------------
# Internal feature names
# ----------------------------------------------------------

FEATURE_COLS = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]

TARGET_COL = "Concrete compressive strength"

# ----------------------------------------------------------
# Target transformation
# ----------------------------------------------------------

LOG_TARGET = False

# ----------------------------------------------------------
# q ablation grid
# ----------------------------------------------------------

Q_GRID = np.array(
    [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
    ],
    dtype=float,
)

# ----------------------------------------------------------
# N_max settings
# ----------------------------------------------------------

REFERENCE_NMAX = 300

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


# ----------------------------------------------------------
# q-selection stability criteria
# ----------------------------------------------------------

REL_CHANGE_THRESHOLD = 0.050

CURVATURE_THRESHOLD = 0.050

MIN_PLATEAU_LENGTH = 3

# ----------------------------------------------------------
# N_max saturation tolerance
# ----------------------------------------------------------

NMAX_SATURATION_TOL = 0.050

# ----------------------------------------------------------
# Output directories
# ----------------------------------------------------------

OUTPUT_DIR = "concrete_kernel_results"


# ----------------------------------------------------------
# Output CSV files
# ----------------------------------------------------------

OUTPUT_Q_CSV = os.path.join(
    OUTPUT_DIR,
    "concrete_q_ablation.csv",
)

Q_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "concrete_q_stability_summary.csv",
)

SELECTED_Q_CSV = os.path.join(
    OUTPUT_DIR,
    "concrete_selected_q.csv",
)

OUTPUT_NMAX_CSV = os.path.join(
    OUTPUT_DIR,
    "concrete_nmax_ablation.csv",
)

NMAX_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "concrete_nmax_saturation_summary.csv",
)


# ==========================================================
# UCI COLUMN NAME MAPPING
# ==========================================================

# The official UCI Concrete Compressive Strength Excel file
# uses long column names. We rename them immediately after
# loading.

UCI_COLUMN_RENAME = {
    "Cement (component 1)(kg in a m^3 mixture)": "Cement",

    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":
        "Blast Furnace Slag",

    "Fly Ash (component 3)(kg in a m^3 mixture)":
        "Fly Ash",

    "Water  (component 4)(kg in a m^3 mixture)":
        "Water",

    "Superplasticizer (component 5)(kg in a m^3 mixture)":
        "Superplasticizer",

    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":
        "Coarse Aggregate",

    "Fine Aggregate (component 7)(kg in a m^3 mixture)":
        "Fine Aggregate",

    "Age (day)":
        "Age",

    "Concrete compressive strength(MPa, megapascals)":
        "Concrete compressive strength",
}


# ==========================================================
# DATASET LOADING
# ==========================================================

def load_concrete_dataset(
    local_path,
    feature_cols,
    target_col,
):
    """
    Load the UCI Concrete Compressive Strength dataset.

    The raw UCI Excel file contains long column names.
    They are renamed to the short names used throughout
    this script.
    """

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"\nCould not find the dataset:\n"
            f"    {local_path}\n\n"
            f"Download 'Concrete_Data.xls' from the UCI "
            f"Concrete Compressive Strength dataset and "
            f"place it in the same folder as this script."
        )

    print(
        f"Loading dataset: {local_path}"
    )

    # ------------------------------------------------------
    # Read raw Excel file
    # ------------------------------------------------------

    try:
        df = pd.read_excel(local_path)
    except ImportError as exc:
        raise ImportError(
            "\nCould not read the Excel file.\n\n"
            "You may need the xlrd package for the "
            "legacy .xls UCI file.\n\n"
            "Install it with:\n"
            "    pip install xlrd"
        ) from exc

    # ------------------------------------------------------
    # Normalize column-name whitespace
    # ------------------------------------------------------

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "Raw columns found:"
    )

    for col in df.columns:
        print(
            f"    {col}"
        )

    # ------------------------------------------------------
    # Check expected UCI columns
    # ------------------------------------------------------

    missing_raw_columns = [
        col
        for col in UCI_COLUMN_RENAME
        if col not in df.columns
    ]

    if missing_raw_columns:
        raise ValueError(
            "\nThe Excel file does not contain the "
            "expected UCI Concrete columns.\n\n"
            "Missing columns:\n"
            + "\n".join(
                f"    {col}"
                for col in missing_raw_columns
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                f"    {col}"
                for col in df.columns
            )
        )

    # ------------------------------------------------------
    # Rename UCI columns
    # ------------------------------------------------------

    df = df.rename(
        columns=UCI_COLUMN_RENAME
    )

    print(
        "\nColumns after UCI -> internal-name mapping:"
    )

    for col in df.columns:
        print(
            f"    {col}"
        )

    # ------------------------------------------------------
    # Check internal columns
    # ------------------------------------------------------

    required_cols = (
        feature_cols
        + [target_col]
    )

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing required internal columns: "
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # ------------------------------------------------------
    # Convert required columns to numeric
    # ------------------------------------------------------

    for col in required_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # ------------------------------------------------------
    # Remove rows containing missing required values
    # ------------------------------------------------------

    n_before = len(df)

    df = df.dropna(
        subset=required_cols
    ).reset_index(drop=True)

    n_after = len(df)

    if n_after < n_before:
        print(
            f"\nRemoved "
            f"{n_before - n_after} rows "
            f"containing missing/non-numeric values."
        )

    if n_after == 0:
        raise ValueError(
            "No valid observations remain after "
            "removing missing values."
        )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    X_raw = (
        df[feature_cols]
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------
    # Standardize features
    # ------------------------------------------------------
    #
    # This is important because the input variables have
    # substantially different numerical scales.
    #
    # The feasible-set calculation uses distances in X,
    # so standardization prevents variables such as
    # aggregate mass from dominating variables such as age.
    # ------------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    y = (
        df[target_col]
        .to_numpy(dtype=float)
        .reshape(-1, 1)
    )

    if LOG_TARGET:
        y = np.log1p(y)

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Number of observations: "
        f"{len(df)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_cols)}"
    )

    print("\nFeatures:")

    for col in feature_cols:
        print(
            f"    {col}"
        )

    print(
        f"\nX shape: {X.shape}"
    )

    print(
        f"\nTarget: {target_col}"
    )

    print(
        f"    shape = {y.shape}"
    )

    print(
        f"    min   = {y.min():.6f}"
    )

    print(
        f"    max   = {y.max():.6f}"
    )

    print(
        f"    mean  = {y.mean():.6f}"
    )

    print(
        f"    std   = {y.std():.6f}"
    )

    print(
        f"\nMissing values remaining: "
        f"{df[required_cols].isna().sum().sum()}"
    )

    print(
        f"log_target: {LOG_TARGET}"
    )

    return (
        X,
        y,
        df,
        scaler,
    )


# ==========================================================
# FEASIBLE-SET FUNCTION CALL
# ==========================================================

def call_feasible_sets_features_nnx(
    X,
    y,
    q,
    N_max,
):
    """
    Call feasible_sets_features_nnx().
    """

    try:
        signature = inspect.signature(
            feasible_sets_features_nnx
        )

        parameters = signature.parameters

        kwargs = {}

        # N_max
        if "N_max" in parameters:
            kwargs["N_max"] = N_max
        elif "n_max" in parameters:
            kwargs["n_max"] = N_max
        else:
            raise TypeError(
                "feasible_sets_features_nnx() does not "
                "appear to accept an N_max/n_max argument."
            )

        # Feature-side q
        if "qx" in parameters:
            kwargs["qx"] = q
        elif "q_x" in parameters:
            kwargs["q_x"] = q
        elif "q" in parameters:
            kwargs["q"] = q
        else:
            raise TypeError(
                "feasible_sets_features_nnx() does not "
                "appear to accept a q/qx/q_x argument."
            )


        # ext_D is used by the version shown in your original
        # script. Only provide it when supported.
        if "ext_D" in parameters:
            kwargs["ext_D"] = False

        result = feasible_sets_features_nnx(
            X,
            y,
            **kwargs,
        )

        return result

    except Exception as exc:
        raise RuntimeError(
            "\nCall to "
            "feasible_sets_features_nnx(...) failed.\n\n"
            f"Original error:\n{exc}\n\n"
            "Installed function signature:\n"
            f"{inspect.signature(feasible_sets_features_nnx)}"
        ) from exc


# ==========================================================
# KERNEL-SIZE EXTRACTION
# ==========================================================

def extract_kernel_statistics(
    feasible_sets,
):
    """
    Compute kernel-size statistics from feasible sets.
    """

    if feasible_sets is None:
        raise ValueError(
            "feasible_sets is None."
        )

    if len(feasible_sets) == 0:
        raise ValueError(
            "feasible_sets is empty."
        )

    kernel_size, M_prime, Nk_stats = (
        compute_kernel_size(
            feasible_sets,
            metric="l2",
            p=2.0,
        )
    )

    R = np.sqrt(compute_R_squared(feasible_sets))

    feasible_sizes = np.asarray(
        [
            len(fs)
            for fs in feasible_sets
        ],
        dtype=int,
    )

    if len(feasible_sizes) == 0:
        raise ValueError(
            "No feasible-set sizes were returned."
        )

    return {
        "Kernel_size": float(
            kernel_size
        ),

        "M_prime": M_prime,

        "Nk_stats": Nk_stats,

        "mean_feasible_set_size": float(
            np.mean(feasible_sizes)
        ),

        "median_feasible_set_size": float(
            np.median(feasible_sizes)
        ),

        "min_feasible_set_size": int(
            np.min(feasible_sizes)
        ),

        "max_feasible_set_size": int(
            np.max(feasible_sizes)
        ),

        "R": float(R),
    }


# ==========================================================
# CORE FEASIBLE-SET STATISTICS
# ==========================================================

def compute_statistics_concrete(
    X,
    y,
    q,
    N_max,
):
    """
    Compute feasible-set and kernel-size statistics.

    Feasible-set computation:

        d_X(x_k, x_j) <= tau_x
    
    The target criterion (d_Y(y_k, y_j) <= tau_y) is omitted here, and ideal noiseless targets are assumed.

    q controls the feature-side neighborhood.

    Parameters
    ----------
    X : ndarray
        Standardized feature matrix.

    y : ndarray
        Concrete compressive strength target.

    q : float
        Feature-side q parameter.

    N_max : int
        Maximum feasible-set size/search parameter.
    """

    result = call_feasible_sets_features_nnx(
        X=X,
        y=y,
        q=q,
        N_max=N_max,
    )

    # ------------------------------------------------------
    # Validate returned result
    # ------------------------------------------------------

    if not isinstance(result, dict):
        raise TypeError(
            "\nExpected "
            "feasible_sets_features_nnx() "
            "to return a dictionary.\n\n"
            f"Received type: {type(result)}\n"
            f"Received value: {result}"
        )

    if "feasible_sets" not in result:
        raise KeyError(
            "\nThe result returned by "
            "feasible_sets_features_nnx() "
            "does not contain 'feasible_sets'.\n\n"
            f"Available keys: {list(result.keys())}"
        )

    feasible_sets = result[
        "feasible_sets"
    ]

    statistics = (
        extract_kernel_statistics(
            feasible_sets
        )
    )

    feasible_sizes = np.asarray(
        [
            len(fs)
            for fs in feasible_sets
        ],
        dtype=int,
    )

    # ------------------------------------------------------
    # N_max truncation diagnostic
    # ------------------------------------------------------
    #
    # Depending on the implementation of
    # feasible_sets_features_nnx, the returned feasible
    # set may contain the query point itself.
    #
    # We therefore retain the same N_max + 1 convention
    # used in your original script.
    # ------------------------------------------------------

    truncation_limit = min(
        N_max + 1,
        len(y),
    )

    fraction_at_Nmax = float(
        np.mean(
            feasible_sizes
            >= truncation_limit
        )
    )

    # ------------------------------------------------------
    # Extract tau_x if supplied by the package
    # ------------------------------------------------------

    tau_x = result.get(
        "tau_x",
        np.nan,
    )

    try:
        tau_x = float(tau_x)
    except (TypeError, ValueError):
        tau_x = np.nan

    return {
        "q": float(q),

        "N_max": int(N_max),

        "tau_x": tau_x,

        "M_prime": statistics[
            "M_prime"
        ],

        "Nk_stats": str(
            statistics["Nk_stats"]
        ),

        "Kernel_size": statistics[
            "Kernel_size"
        ],

        "mean_feasible_set_size": (
            statistics[
                "mean_feasible_set_size"
            ]
        ),

        "median_feasible_set_size": (
            statistics[
                "median_feasible_set_size"
            ]
        ),

        "min_feasible_set_size": (
            statistics[
                "min_feasible_set_size"
            ]
        ),

        "max_feasible_set_size": (
            statistics[
                "max_feasible_set_size"
            ]
        ),

        "fraction_at_Nmax": (
            fraction_at_Nmax
        ),

        "R": (statistics["R"]),
    }


# ==========================================================
# STABILITY DIAGNOSTICS
# ==========================================================

def add_stability_columns(
    df,
):
    """
    Add q-stability diagnostics.

    Relative change:

        |K_i - K_{i-1}| / |K_{i-1}|

    Normalized local curvature:

        |K_{i+1} - 2K_i + K_{i-1}| / |K_i|
    """

    g = (
        df.sort_values("q")
        .copy()
        .reset_index(drop=True)
    )

    K = g[
        "Kernel_size"
    ].to_numpy(dtype=float)

    rel_change = np.full(
        len(K),
        np.nan,
        dtype=float,
    )

    for i in range(1, len(K)):
        rel_change[i] = (
            abs(
                K[i] - K[i - 1]
            )
            / max(
                abs(K[i - 1]),
                1e-12,
            )
        )

    curvature = np.full(
        len(K),
        np.nan,
        dtype=float,
    )

    for i in range(1, len(K) - 1):
        curvature[i] = (
            abs(
                K[i + 1]
                - 2.0 * K[i]
                + K[i - 1]
            )
            / max(
                abs(K[i]),
                1e-12,
            )
        )

    g[
        "rel_change_K"
    ] = rel_change

    g[
        "normalized_local_curvature"
    ] = curvature

    return g


# ==========================================================
# STAGE 1: q ABLATION
# ==========================================================

def run_q_ablation(
    X,
    y,
    q_grid,
    reference_nmax,
):
    """
    Run q ablation at fixed N_max.
    """

    rows = []

    for q in q_grid:


        row = (
            compute_statistics_concrete(
                X=X,
                y=y,
                q=q,
                N_max=reference_nmax,
            )
        )

        rows.append(row)

    df = pd.DataFrame(
        rows
    )

    df = add_stability_columns(
        df
    )

    return df


# ==========================================================
# STABLE-q SELECTION
# ==========================================================

def select_stable_q(
    q_summary,
    rel_change_threshold,
    curvature_threshold,
    min_plateau_length,
):
    """
    Select the first q belonging to a stable plateau.

    A point is stable when:

        relative change <= threshold

    AND

        normalized curvature <= threshold

    A plateau must contain at least
    min_plateau_length consecutive stable points.

    If no plateau is found, fall back to the q with the
    smallest combined diagnostic score.
    """

    g = (
        q_summary
        .sort_values("q")
        .reset_index(drop=True)
        .copy()
    )

    q_values = g[
        "q"
    ].to_numpy(dtype=float)

    rel_change = g[
        "rel_change_K"
    ].to_numpy(dtype=float)

    curvature = g[
        "normalized_local_curvature"
    ].to_numpy(dtype=float)

    stable = (
        np.isfinite(rel_change)
        & np.isfinite(curvature)
        & (
            rel_change
            <= rel_change_threshold
        )
        & (
            curvature
            <= curvature_threshold
        )
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

        while (
            i + 1 < len(stable)
            and stable[i + 1]
        ):
            i += 1

        end = i

        current_length = (
            end - start + 1
        )

        if (
            current_length
            >= min_plateau_length
        ):
            selected_q = (
                q_values[start]
            )

            plateau_start = (
                q_values[start]
            )

            plateau_end = (
                q_values[end]
            )

            plateau_length = (
                current_length
            )

            break

        i += 1

    fallback_used = False

    # ------------------------------------------------------
    # Fallback
    # ------------------------------------------------------

    if selected_q is None:

        fallback_used = True

        score = (
            np.nan_to_num(
                rel_change,
                nan=np.inf,
            )
            +
            np.nan_to_num(
                curvature,
                nan=np.inf,
            )
        )

        if not np.any(
            np.isfinite(score)
        ):
            best_idx = 0

        else:
            best_idx = int(
                np.nanargmin(score)
            )

        selected_q = (
            q_values[best_idx]
        )

    selected_rows = g[
        np.isclose(
            g["q"],
            selected_q,
        )
    ]

    if selected_rows.empty:
        raise RuntimeError(
            f"Could not locate selected q="
            f"{selected_q} in q summary."
        )

    selected = (
        selected_rows.iloc[0]
    )

    return pd.DataFrame(
        [
            {
                "selected_q": selected_q,

                "tau_x": selected[
                    "tau_x"
                ],

                "Kernel_size": selected[
                    "Kernel_size"
                ],

                "M_prime": selected[
                    "M_prime"
                ],

                "mean_feasible_set_size": (
                    selected[
                        "mean_feasible_set_size"
                    ]
                ),

                "median_feasible_set_size": (
                    selected[
                        "median_feasible_set_size"
                    ]
                ),

                "min_feasible_set_size": (
                    selected[
                        "min_feasible_set_size"
                    ]
                ),

                "max_feasible_set_size": (
                    selected[
                        "max_feasible_set_size"
                    ]
                ),

                "rel_change_K": selected[
                    "rel_change_K"
                ],

                "normalized_local_curvature": (
                    selected[
                        "normalized_local_curvature"
                    ]
                ),

                "plateau_start_q": (
                    plateau_start
                ),

                "plateau_end_q": (
                    plateau_end
                ),

                "plateau_length": (
                    plateau_length
                ),

                "fallback_used": (
                    fallback_used
                ),
            }
        ]
    )


# ==========================================================
# STAGE 2: N_max ABLATION
# ==========================================================

def run_nmax_ablation(
    X,
    y,
    selected_q,
    nmax_grid,
):
    """
    Run N_max ablation at the selected q.
    """

    rows = []

    q_star = float(
        selected_q[
            "selected_q"
        ].iloc[0]
    )

    for N_max in nmax_grid:

        row = (
            compute_statistics_concrete(
                X=X,
                y=y,
                q=q_star,
                N_max=N_max,
            )
        )

        row[
            "selected_q"
        ] = q_star

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ==========================================================
# N_max SATURATION ANALYSIS
# ==========================================================

def compute_nmax_saturation(
    df,
    reference_nmax,
):
    """
    Compare every N_max against the reference N_max.

    saturated =
        relative difference from reference
        <= NMAX_SATURATION_TOL
    """

    g = (
        df.sort_values("N_max")
        .copy()
        .reset_index(drop=True)
    )

    reference_rows = g[
        g["N_max"]
        == reference_nmax
    ]

    if reference_rows.empty:
        raise ValueError(
            f"Reference N_max="
            f"{reference_nmax} "
            f"not found in N_max results."
        )

    reference_kernel_size = float(
        reference_rows[
            "Kernel_size"
        ].iloc[0]
    )

    q_star = float(
        reference_rows[
            "q"
        ].iloc[0]
    )

    rows = []

    for _, row in g.iterrows():

        relative_to_reference = (
            abs(
                row["Kernel_size"]
                - reference_kernel_size
            )
            / max(
                abs(
                    reference_kernel_size
                ),
                1e-12,
            )
        )

        saturated = (
            relative_to_reference
            <= NMAX_SATURATION_TOL
        )

        rows.append(
            {
                "selected_q": q_star,

                "N_max": row[
                    "N_max"
                ],

                "Kernel_size": row[
                    "Kernel_size"
                ],

                "R": row["R"],

                "tau_x": row[
                    "tau_x"
                ],

                "mean_feasible_set_size": (
                    row[
                        "mean_feasible_set_size"
                    ]
                ),

                "max_feasible_set_size": (
                    row[
                        "max_feasible_set_size"
                    ]
                ),

                "fraction_at_Nmax": row[
                    "fraction_at_Nmax"
                ],

                "relative_to_reference": (
                    relative_to_reference
                ),

                "saturated": saturated,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    np.random.seed(42)

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    # ------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "CONCRETE COMPRESSIVE STRENGTH "
        "FEASIBLE-SET ANALYSIS"
    )

    print("=" * 70)

    print(
        "\nDataset:"
        "\n    UCI Concrete Compressive Strength"
    )

    print(
        f"\nQ_GRID: {Q_GRID}"
    )

    print(
        f"REFERENCE_NMAX: "
        f"{REFERENCE_NMAX}"
    )

    print(
        f"FINAL NMAX_GRID: "
        f"{NMAX_GRID}"
    )

    print(
        "\nq selection rule:"
        f"\n    relative change <= "
        f"{REL_CHANGE_THRESHOLD:.3f}"
        f"\n    curvature <= "
        f"{CURVATURE_THRESHOLD:.3f}"
        f"\n    minimum plateau length = "
        f"{MIN_PLATEAU_LENGTH}"
    )

    print(
        "\nN_max saturation rule:"
        f"\n    relative difference <= "
        f"{NMAX_SATURATION_TOL:.3f}"
    )

    # ======================================================
    # LOAD DATA
    # ======================================================

    X, y, df_raw, scaler = (
        load_concrete_dataset(
            local_path=LOCAL_XLS,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
        )
    )

    # ======================================================
    # STAGE 1
    # q ABLATION
    # ======================================================

    q_results = run_q_ablation(
        X=X,
        y=y,
        q_grid=Q_GRID,
        reference_nmax=REFERENCE_NMAX,
    )

    q_results = q_results.round(
        6
    )

    q_results.to_csv(
        OUTPUT_Q_CSV,
        index=False,
        float_format="%.6f",
    )


    # ------------------------------------------------------
    # q stability summary
    # ------------------------------------------------------

    q_summary = (
        q_results
        .sort_values("q")
        .copy()
    )

    q_summary.to_csv(
        Q_SUMMARY_CSV,
        index=False,
        float_format="%.6f",
    )

    print(
        "\nq stability summary saved to:"
    )

    print(
        f"    {Q_SUMMARY_CSV}"
    )

    # ======================================================
    # SELECT q
    # ======================================================

    selected_q = select_stable_q(
        q_summary=q_summary,

        rel_change_threshold=(
            REL_CHANGE_THRESHOLD
        ),

        curvature_threshold=(
            CURVATURE_THRESHOLD
        ),

        min_plateau_length=(
            MIN_PLATEAU_LENGTH
        ),
    )

    selected_q = selected_q.round(
        6
    )

    selected_q.to_csv(
        SELECTED_Q_CSV,
        index=False,
        float_format="%.6f",
    )

    print("\n" + "=" * 70)

    print(
        "SELECTED STABLE q"
    )

    print("=" * 70)

    print(
        f"q selection N_max: "
        f"{REFERENCE_NMAX}"
    )

    print(
        f"Relative-change threshold: "
        f"{REL_CHANGE_THRESHOLD:.3f}"
    )

    print(
        f"Curvature threshold: "
        f"{CURVATURE_THRESHOLD:.3f}"
    )

    print(
        f"Minimum plateau length: "
        f"{MIN_PLATEAU_LENGTH}"
    )

    print()

    print(
        selected_q.to_string(
            index=False
        )
    )

    # ======================================================
    # STAGE 2
    # N_max ABLATION
    # ======================================================

    nmax_results = (
        run_nmax_ablation(
            X=X,
            y=y,
            selected_q=selected_q,
            nmax_grid=NMAX_GRID,
        )
    )

    nmax_results = (
        nmax_results.round(6)
    )

    nmax_results.to_csv(
        OUTPUT_NMAX_CSV,
        index=False,
        float_format="%.6f",
    )

    print("\n" + "=" * 70)

    print(
        "N_max / KERNEL-SIZE "
        "ABLATION SAVED"
    )

    print("=" * 70)

    print(
        f"    {OUTPUT_NMAX_CSV}"
    )

    # ======================================================
    # N_max SATURATION
    # ======================================================

    nmax_summary = (
        compute_nmax_saturation(
            df=nmax_results,
            reference_nmax=REFERENCE_NMAX,
        )
    )

    nmax_summary = (
        nmax_summary.round(6)
    )

    nmax_summary.to_csv(
        NMAX_SUMMARY_CSV,
        index=False,
        float_format="%.6f",
    )

    print(
        "\nN_max saturation summary saved to:"
    )

    print(
        f"    {NMAX_SUMMARY_CSV}"
    )

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}",
    )

    print("\n" + "=" * 70)

    print(
        f"q ABLATION "
        f"(N_max={REFERENCE_NMAX})"
    )

    print("=" * 70)

    print(
        q_results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print(
        "SELECTED CONFIGURATION"
    )

    print("=" * 70)

    print(
        selected_q.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print(
        "KERNEL-SIZE ABLATION "
        "OVER N_max"
    )

    print("=" * 70)

    print(
        nmax_results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print(
        "N_max SATURATION ANALYSIS"
    )

    print("=" * 70)

    print(
        nmax_summary.to_string(
            index=False
        )
    )



    # ======================================================
    # COMPLETE
    # ======================================================

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"q ablation CSV: "
        f"{OUTPUT_Q_CSV}"
    )

    print(
        f"N_max ablation CSV: "
        f"{OUTPUT_NMAX_CSV}"
    )

    print(
        f"q stability CSV: "
        f"{Q_SUMMARY_CSV}"
    )

    print(
        f"Selected q CSV: "
        f"{SELECTED_Q_CSV}"
    )

    print(
        f"N_max saturation CSV: "
        f"{NMAX_SUMMARY_CSV}"
    )

    # ------------------------------------------------------
    # Final selected configuration
    # ------------------------------------------------------

    print(
        "\nSelected q:"
    )

    q_star = float(
        selected_q[
            "selected_q"
        ].iloc[0]
    )

    plateau_start = (
        selected_q[
            "plateau_start_q"
        ].iloc[0]
    )

    plateau_end = (
        selected_q[
            "plateau_end_q"
        ].iloc[0]
    )

    if pd.notna(
        plateau_start
    ):
        plateau_start_str = (
            f"{plateau_start:.2f}"
        )
    else:
        plateau_start_str = "nan"

    if pd.notna(
        plateau_end
    ):
        plateau_end_str = (
            f"{plateau_end:.2f}"
        )
    else:
        plateau_end_str = "nan"

    tau_x_star = selected_q[
        "tau_x"
    ].iloc[0]

    kernel_star = selected_q[
        "Kernel_size"
    ].iloc[0]

    plateau_length = int(
        selected_q[
            "plateau_length"
        ].iloc[0]
    )

    fallback_used = bool(
        selected_q[
            "fallback_used"
        ].iloc[0]
    )

    print(
        f"    Concrete compressive "
        f"strength: "
        f"q={q_star:.2f}, "
        f"tau_x={tau_x_star:.6f}, "
        f"Kernel_size={kernel_star:.6f}, "
        f"plateau="
        f"[{plateau_start_str}, "
        f"{plateau_end_str}], "
        f"length={plateau_length}, "
        f"fallback_used={fallback_used}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()


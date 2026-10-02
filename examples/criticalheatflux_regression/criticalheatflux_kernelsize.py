import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nn_xy,
)
from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)
from utils import chf_uncertainty


# ==========================================================
# CONFIGURATION
# ==========================================================

LOCAL_XML = "uniform.xml"

FEATURE_COLS = [
    "Diameter",
    "Length",
    "Pressure",
    "MassFlux",
    "EquilibriumQuality",
]

TARGET_COL = "HeatFlux"

LOG_TARGET = False

TAU_X = 0.171840


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

NMAX_SATURATION_TOL = 0.050

OUTPUT_DIR = "chf_uniform_kernel_results"

OUTPUT_NMAX_CSV = os.path.join(
    OUTPUT_DIR,
    "chf_uniform_kernelsize_nmax_ablation.csv",
)



# ==========================================================
# XML HELPERS
# ==========================================================

def get_xml_value(element, tag, required=True):
    child = element.find(tag)

    if child is None or child.text is None or not child.text.strip():
        if required:
            raise ValueError(
                f"Missing or empty XML element <{tag}> "
                f"in <{element.tag}>."
            )
        return None

    return child.text.strip()


def get_xml_float(element, tag, required=True):
    value = get_xml_value(element, tag, required)

    if value is None:
        return np.nan

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Could not convert <{tag}>{value}</{tag}> to float."
        ) from exc


# ==========================================================
# DATASET LOADING
# ==========================================================

def load_chf_uniform_dataset(local_path, feature_cols, target_col):

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Could not find the CHF dataset: {local_path}"
        )

    try:
        root = ET.parse(local_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(
            f"Could not parse {local_path}: {exc}"
        ) from exc

    dataset_elements = root.findall(".//dataset")

    if not dataset_elements:
        dataset_elements = root.findall(".//Dataset")

    if not dataset_elements:
        raise ValueError(
            "Could not find any <dataset> elements in the XML file."
        )

    rows = []

    for index, dataset in enumerate(dataset_elements, start=1):
        try:
            rows.append({
                "TestID": get_xml_value(
                    dataset,
                    "TestID",
                    required=False,
                ) or str(index),

                "Diameter": get_xml_float(
                    dataset,
                    "Diameter",
                ),

                "Length": get_xml_float(
                    dataset,
                    "Length",
                ),

                "Pressure": get_xml_float(
                    dataset,
                    "Pressure",
                ),

                "MassFlux": get_xml_float(
                    dataset,
                    "MassFlux",
                ),

                "EquilibriumQuality": get_xml_float(
                    dataset,
                    "EquilibriumQuality",
                ),

                "HeatFlux": get_xml_float(
                    dataset,
                    "HeatFlux",
                ),

                "Perimeter": get_xml_float(
                    dataset,
                    "Perimeter",
                    required=False,
                ),

                "Area": get_xml_float(
                    dataset,
                    "Area",
                    required=False,
                ),

                "Power": get_xml_float(
                    dataset,
                    "Power",
                    required=False,
                ),

                "MassFlow": get_xml_float(
                    dataset,
                    "MassFlow",
                    required=False,
                ),

                "InletTemperature": get_xml_float(
                    dataset,
                    "InletTemperature",
                    required=False,
                ),

                "InletEnthalpy": get_xml_float(
                    dataset,
                    "InletEnthalpy",
                    required=False,
                ),

                "Source": get_xml_value(
                    dataset,
                    "Source",
                    required=False,
                ),

                "Fluid": get_xml_value(
                    dataset,
                    "Fluid",
                    required=False,
                ),
            })

        except Exception:
            # Skip malformed XML records.
            continue

    if not rows:
        raise ValueError(
            "No valid CHF observations could be extracted."
        )

    df = pd.DataFrame(rows)

    required_cols = feature_cols + [target_col]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in required_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_cols
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid observations remain after cleaning."
        )

    # ------------------------------------------------------
    # Physical validity
    # ------------------------------------------------------

    valid = (
        (df["Diameter"] > 0)
        & (df["Length"] > 0)
        & (df["Pressure"] > 0)
        & (df["MassFlux"] > 0)
        & (df["HeatFlux"] > 0)
    )

    df = df.loc[valid].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No physically valid CHF observations remain."
        )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    X_raw = df[
        feature_cols
    ].to_numpy(dtype=float)

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    y = df[
        target_col
    ].to_numpy(
        dtype=float
    ).reshape(-1, 1)

    if LOG_TARGET:
        y = np.log1p(y)

    return X, y, df, scaler


# ==========================================================
# KERNEL STATISTICS
# ==========================================================

def extract_kernel_statistics(feasible_sets):

    if feasible_sets is None or len(feasible_sets) == 0:
        raise ValueError(
            "feasible_sets is empty."
        )

    kernel_size, M_prime, Nk_stats = compute_kernel_size(
        feasible_sets,
        metric="l2",
        p=2.0,
    )

    sizes = np.asarray(
        [
            len(fs)
            for fs in feasible_sets
        ],
        dtype=int,
    )

    if sizes.size == 0:
        raise ValueError(
            "No feasible-set sizes were returned."
        )

    return {
        "Kernel_size": float(kernel_size),
        "M_prime": M_prime,
        "Nk_stats": Nk_stats,
        "mean_feasible_set_size": float(
            np.mean(sizes)
        ),
        "median_feasible_set_size": float(
            np.median(sizes)
        ),
        "min_feasible_set_size": int(
            np.min(sizes)
        ),
        "max_feasible_set_size": int(
            np.max(sizes)
        ),
    }


# ==========================================================
# CHF STATISTICS
# ==========================================================

def compute_statistics_chf(
    X,
    y,
    N_max,
    tau_x,
):
    """
    Compute feasible-set and kernel-size statistics.
    """

    # Direct call to feasible_sets_features_nn_xy().
    result = feasible_sets_features_nn_xy(
        X,
        y,
        tau_x=tau_x,
        N_max=N_max,
        target_uncertainty=chf_uncertainty,
        ext_D=False,
    )

    if not isinstance(result, dict):
        raise TypeError(
            "feasible_sets_features_nn_xy() "
            "must return a dictionary."
        )

    if "feasible_sets" not in result:
        raise KeyError(
            "Result does not contain 'feasible_sets'."
        )

    feasible_sets = result[
        "feasible_sets"
    ]

    statistics = extract_kernel_statistics(
        feasible_sets
    )

    feasible_sizes = np.asarray(
        [
            len(fs)
            for fs in feasible_sets
        ],
        dtype=int,
    )

    truncation_limit = min(
        N_max + 1,
        len(y),
    )

    fraction_at_Nmax = float(
        np.mean(
            feasible_sizes >= truncation_limit
        )
    )

    return {
        "N_max": int(N_max),

        "tau_x": float(
            result.get(
                "tau_x",
                np.nan,
            )
        ),

        "tau_x_configured": float(
            tau_x
        ),

        "tau_y": float(
            result.get(
                "tau_y",
                np.nan,
            )
        ),

        "M_prime": statistics[
            "M_prime"
        ],

        "Nk_stats": str(
            statistics[
                "Nk_stats"
            ]
        ),

        "Kernel_size": statistics[
            "Kernel_size"
        ],

        "mean_feasible_set_size": statistics[
            "mean_feasible_set_size"
        ],

        "median_feasible_set_size": statistics[
            "median_feasible_set_size"
        ],

        "min_feasible_set_size": statistics[
            "min_feasible_set_size"
        ],

        "max_feasible_set_size": statistics[
            "max_feasible_set_size"
        ],

        "fraction_at_Nmax": fraction_at_Nmax,
    }


# ==========================================================
# N_MAX ABLATION
# ==========================================================

def run_nmax_ablation(
    X,
    y,
    tau_x,
    nmax_grid,
):

    rows = []

    for N_max in nmax_grid:

        rows.append(
            compute_statistics_chf(
                X,
                y,
                N_max,
                tau_x,
            )
        )

    return pd.DataFrame(rows)



# ==========================================================
# MAIN
# ==========================================================

def main():

    np.random.seed(42)

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Load and clean data
    # ------------------------------------------------------

    X, y, df, _ = load_chf_uniform_dataset(
        LOCAL_XML,
        FEATURE_COLS,
        TARGET_COL,
    )

    cleaned_csv = os.path.join(
        OUTPUT_DIR,
        "chf_uniform_cleaned_data.csv",
    )

    df.to_csv(
        cleaned_csv,
        index=False,
    )

    # ------------------------------------------------------
    # N_max ablation
    # ------------------------------------------------------

    nmax_results = run_nmax_ablation(
        X,
        y,
        TAU_X,
        NMAX_GRID,
    ).round(6)

    nmax_results.to_csv(
        OUTPUT_NMAX_CSV,
        index=False,
        float_format="%.6f",
    )


    # ------------------------------------------------------
    # Minimal output
    # ------------------------------------------------------

    print(f"Cleaned data: {cleaned_csv}")
    print(f"N_max results: {OUTPUT_NMAX_CSV}")



if __name__ == "__main__":
    main()

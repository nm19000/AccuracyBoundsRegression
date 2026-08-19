import os
import warnings
import inspect
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

LOCAL_XML = "uniform.xml"

# ----------------------------------------------------------
# Dataset identification
# ----------------------------------------------------------

DATASET_NAME = (
    "Critical Heat Flux - Uniform Axial Heating "
    "[KAERI/TR-1665/2000]"
)

# ----------------------------------------------------------
# Internal feature names
# ----------------------------------------------------------

FEATURE_COLS = [
    "Diameter",
    "Length",
    "Pressure",
    "MassFlux",
    "EquilibriumQuality",
]

TARGET_COL = "HeatFlux"

# ----------------------------------------------------------
# Target transformation
# ----------------------------------------------------------

LOG_TARGET = False


# ----------------------------------------------------------
# Feature-side uncertainty threshold
# ----------------------------------------------------------
#
# This is the tau_x value supplied to
# feasible_sets_features_nn_xy().
# ----------------------------------------------------------

TAU_X = 0.171840

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
# N_max saturation tolerance
# ----------------------------------------------------------

NMAX_SATURATION_TOL = 0.050

# ----------------------------------------------------------
# Output directories
# ----------------------------------------------------------

OUTPUT_DIR = "chf_uniform_kernel_results"

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots",
)

# ----------------------------------------------------------
# Output CSV files
# ----------------------------------------------------------

OUTPUT_NMAX_CSV = os.path.join(
    OUTPUT_DIR,
    "chf_uniform_nmax_ablation.csv",
)

NMAX_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "chf_uniform_nmax_saturation_summary.csv",
)


# ==========================================================
# XML HELPERS
# ==========================================================

def get_xml_value(
    element,
    tag,
    required=True,
):
    """
    Extract the text value of a child XML element.
    """

    child = element.find(tag)

    if child is None:

        if required:
            raise ValueError(
                f"Could not find XML element "
                f"<{tag}> inside <{element.tag}>."
            )

        return None

    if child.text is None:

        if required:
            raise ValueError(
                f"XML element <{tag}> is empty."
            )

        return None

    value = child.text.strip()

    if value == "":

        if required:
            raise ValueError(
                f"XML element <{tag}> is empty."
            )

        return None

    return value


def get_xml_float(
    element,
    tag,
    required=True,
):
    """
    Extract an XML value and convert it to float.
    """

    value = get_xml_value(
        element,
        tag,
        required=required,
    )

    if value is None:
        return np.nan

    try:

        return float(value)

    except ValueError as exc:

        raise ValueError(
            f"Could not convert "
            f"<{tag}>{value}</{tag}> "
            f"to float."
        ) from exc


# ==========================================================
# XML DATASET LOADING
# ==========================================================

def load_chf_uniform_dataset(
    local_path,
    feature_cols,
    target_col,
):
    """
    Load the uniform KAERI CHF dataset from XML.

    Principal ML inputs:

        Diameter
        Length
        Pressure
        MassFlux
        EquilibriumQuality

    Target:

        HeatFlux
    """

    if not os.path.exists(local_path):

        raise FileNotFoundError(
            f"\nCould not find the CHF dataset:\n"
            f"    {local_path}\n\n"
            f"Place 'uniform.xml' in the same folder "
            f"as this script."
        )

    print(
        f"Loading CHF XML dataset: "
        f"{local_path}"
    )

    # ------------------------------------------------------
    # Parse XML
    # ------------------------------------------------------

    try:

        tree = ET.parse(
            local_path
        )

    except ET.ParseError as exc:

        raise ValueError(
            "\nCould not parse uniform.xml.\n\n"
            f"XML parsing error:\n{exc}"
        ) from exc

    root = tree.getroot()

    print(
        f"XML root tag: "
        f"<{root.tag}>"
    )

    # ------------------------------------------------------
    # Locate dataset elements
    # ------------------------------------------------------

    dataset_elements = root.findall(
        ".//dataset"
    )

    if not dataset_elements:

        dataset_elements = root.findall(
            ".//Dataset"
        )

    if not dataset_elements:

        raise ValueError(
            "\nCould not find any "
            "<dataset> elements in the XML file.\n\n"
            f"Root tag: {root.tag}\n\n"
            "The XML structure may differ from the "
            "expected KAERI CHF benchmark format."
        )

    print(
        f"Number of XML datasets found: "
        f"{len(dataset_elements)}"
    )

    # ------------------------------------------------------
    # Extract records
    # ------------------------------------------------------

    rows = []
    skipped = []

    for index, dataset in enumerate(
        dataset_elements,
        start=1,
    ):

        try:

            test_id = get_xml_value(
                dataset,
                "TestID",
                required=False,
            )

            if test_id is None:
                test_id = str(index)

            diameter = get_xml_float(
                dataset,
                "Diameter",
            )

            length = get_xml_float(
                dataset,
                "Length",
            )

            pressure = get_xml_float(
                dataset,
                "Pressure",
            )

            mass_flux = get_xml_float(
                dataset,
                "MassFlux",
            )

            quality = get_xml_float(
                dataset,
                "EquilibriumQuality",
            )

            heat_flux = get_xml_float(
                dataset,
                "HeatFlux",
            )

            perimeter = get_xml_float(
                dataset,
                "Perimeter",
                required=False,
            )

            area = get_xml_float(
                dataset,
                "Area",
                required=False,
            )

            power = get_xml_float(
                dataset,
                "Power",
                required=False,
            )

            mass_flow = get_xml_float(
                dataset,
                "MassFlow",
                required=False,
            )

            inlet_temperature = get_xml_float(
                dataset,
                "InletTemperature",
                required=False,
            )

            inlet_enthalpy = get_xml_float(
                dataset,
                "InletEnthalpy",
                required=False,
            )

            source = get_xml_value(
                dataset,
                "Source",
                required=False,
            )

            fluid = get_xml_value(
                dataset,
                "Fluid",
                required=False,
            )

            rows.append(
                {
                    "TestID": test_id,
                    "Diameter": diameter,
                    "Length": length,
                    "Pressure": pressure,
                    "MassFlux": mass_flux,
                    "EquilibriumQuality": quality,
                    "HeatFlux": heat_flux,
                    "Perimeter": perimeter,
                    "Area": area,
                    "Power": power,
                    "MassFlow": mass_flow,
                    "InletTemperature": inlet_temperature,
                    "InletEnthalpy": inlet_enthalpy,
                    "Source": source,
                    "Fluid": fluid,
                }
            )

        except Exception as exc:

            skipped.append(
                {
                    "xml_index": index,
                    "error": str(exc),
                }
            )

    # ------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------

    if not rows:

        raise ValueError(
            "No valid CHF observations could "
            "be extracted from uniform.xml."
        )

    df = pd.DataFrame(rows)

    print(
        f"Successfully extracted "
        f"{len(df)} test cases."
    )

    if skipped:

        print(
            f"Skipped {len(skipped)} XML records."
        )

        for item in skipped[:10]:

            print(
                f"    XML index "
                f"{item['xml_index']}: "
                f"{item['error']}"
            )

    # ------------------------------------------------------
    # Check required columns
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
            f"\nMissing required columns "
            f"after XML parsing:\n"
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
    # Remove missing required values
    # ------------------------------------------------------

    n_before = len(df)

    df = df.dropna(
        subset=required_cols
    ).reset_index(
        drop=True
    )

    n_after = len(df)

    if n_after < n_before:

        print(
            f"\nRemoved "
            f"{n_before - n_after} rows "
            f"with missing/non-numeric "
            f"required values."
        )

    if n_after == 0:

        raise ValueError(
            "No valid observations remain "
            "after cleaning."
        )

    # ------------------------------------------------------
    # Validate physical values
    # ------------------------------------------------------

    physical_checks = {
        "Diameter": df["Diameter"] > 0,
        "Length": df["Length"] > 0,
        "Pressure": df["Pressure"] > 0,
        "MassFlux": df["MassFlux"] > 0,
        "HeatFlux": df["HeatFlux"] > 0,
    }

    invalid_mask = np.zeros(
        len(df),
        dtype=bool,
    )

    for col, mask in physical_checks.items():

        invalid_count = int(
            (~mask).sum()
        )

        if invalid_count > 0:

            print(
                f"WARNING: "
                f"{invalid_count} rows have "
                f"non-positive {col}."
            )

            invalid_mask |= ~mask

    if invalid_mask.any():

        print(
            f"Removing "
            f"{invalid_mask.sum()} "
            f"physically invalid rows."
        )

        df = df.loc[
            ~invalid_mask
        ].reset_index(
            drop=True
        )

    if len(df) == 0:

        raise ValueError(
            "No physically valid CHF "
            "observations remain."
        )

    # ======================================================
    # FEATURES
    # ======================================================

    X_raw = (
        df[feature_cols]
        .to_numpy(
            dtype=float
        )
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    # ======================================================
    # TARGET
    # ======================================================

    y = (
        df[target_col]
        .to_numpy(
            dtype=float
        )
        .reshape(
            -1,
            1,
        )
    )

    if LOG_TARGET:

        y = np.log1p(
            y
        )

    # ======================================================
    # DIAGNOSTICS
    # ======================================================

    print("\n" + "=" * 70)
    print("CHF DATASET")
    print("=" * 70)

    print(
        f"Dataset: "
        f"{DATASET_NAME}"
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
        f"\nX shape: "
        f"{X.shape}"
    )

    print(
        f"\nTarget: "
        f"{target_col}"
    )

    print(
        f"    shape = "
        f"{y.shape}"
    )

    print(
        f"    min   = "
        f"{y.min():.6f}"
    )

    print(
        f"    max   = "
        f"{y.max():.6f}"
    )

    print(
        f"    mean  = "
        f"{y.mean():.6f}"
    )

    print(
        f"    std   = "
        f"{y.std():.6f}"
    )

    print(
        f"\nMissing values remaining: "
        f"{df[required_cols].isna().sum().sum()}"
    )

    print(
        f"log_target: "
        f"{LOG_TARGET}"
    )

    print(
        "\nFeature ranges before "
        "standardization:"
    )

    for col in feature_cols:

        print(
            f"    {col:<22} "
            f"min={df[col].min():.6g}, "
            f"max={df[col].max():.6g}"
        )

    print(
        "\nTarget range:"
    )

    print(
        f"    HeatFlux: "
        f"{df[target_col].min():.6g} "
        f"to "
        f"{df[target_col].max():.6g} "
        f"W/m^2"
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

def call_feasible_sets_features_nn_xy(
    X,
    y,
    N_max,
    tau_x_value,
    target_uncertainty=chf_uncertainty,
):
    """
    Call feasible_sets_features_nn_xy() while supporting
    different versions of the accuracy_bounds package.

    tau_x_value is explicitly supplied by the caller.
    """

    try:

        signature = inspect.signature(
            feasible_sets_features_nn_xy
        )

        parameters = signature.parameters

        kwargs = {}

        # --------------------------------------------------
        # N_max
        # --------------------------------------------------

        if "N_max" in parameters:

            kwargs["N_max"] = N_max

        elif "n_max" in parameters:

            kwargs["n_max"] = N_max

        else:

            raise TypeError(
                "feasible_sets_features_nn_xy() "
                "does not appear to accept "
                "N_max/n_max."
            )

        # --------------------------------------------------
        # Target uncertainty
        # --------------------------------------------------

        if "target_uncertainty" in parameters:

            kwargs[
                "target_uncertainty"
            ] = target_uncertainty

        elif "uncertainty" in parameters:

            kwargs[
                "uncertainty"
            ] = target_uncertainty

        elif "uncertainty_function" in parameters:

            kwargs[
                "uncertainty_function"
            ] = target_uncertainty

        # --------------------------------------------------
        # ext_D
        # --------------------------------------------------

        if "ext_D" in parameters:

            kwargs["ext_D"] = False

        print(
            "\nCalling feasible_sets_features_nn_xy:"
        )

        print(
            f"    N_max = "
            f"{N_max}"
        )


        print(
            f"    tau_x = "
            f"{tau_x_value:.6f}"
        )

        print(
            f"    target_uncertainty = "
            f"{target_uncertainty}"
        )

        print(
            f"    kwargs = "
            f"{kwargs}"
        )

        result = feasible_sets_features_nn_xy(
            X,
            y,
            tau_x=tau_x_value,
            **kwargs,
        )

        return result

    except Exception as exc:

        raise RuntimeError(
            "\nCall to "
            "feasible_sets_features_nn_xy(...) "
            "failed.\n\n"
            f"Original error:\n{exc}\n\n"
            "Installed function signature:\n"
            f"{inspect.signature(feasible_sets_features_nn_xy)}"
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

    feasible_sizes = np.asarray(
        [
            len(fs)
            for fs in feasible_sets
        ],
        dtype=int,
    )

    if len(feasible_sizes) == 0:

        raise ValueError(
            "No feasible-set sizes "
            "were returned."
        )

    return {
        "Kernel_size": float(
            kernel_size
        ),

        "M_prime": M_prime,

        "Nk_stats": Nk_stats,

        "mean_feasible_set_size": float(
            np.mean(
                feasible_sizes
            )
        ),

        "median_feasible_set_size": float(
            np.median(
                feasible_sizes
            )
        ),

        "min_feasible_set_size": int(
            np.min(
                feasible_sizes
            )
        ),

        "max_feasible_set_size": int(
            np.max(
                feasible_sizes
            )
        ),
    }


# ==========================================================
# CORE CHF STATISTICS
# ==========================================================

def compute_statistics_chf(
    X,
    y,
    N_max,
    tau_x_value,
):
    """
    Compute feasible-set and kernel-size statistics for
    the uniform CHF dataset.
    """

    result = (
        call_feasible_sets_features_nn_xy(
            X=X,
            y=y,
            tau_x_value=tau_x_value,
            N_max=N_max,
            target_uncertainty=chf_uncertainty,
        )
    )

    # ------------------------------------------------------
    # Validate result
    # ------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):

        raise TypeError(
            "\nExpected "
            "feasible_sets_features_nn_xy() "
            "to return a dictionary.\n\n"
            f"Received type: "
            f"{type(result)}\n"
            f"Received value: "
            f"{result}"
        )

    if "feasible_sets" not in result:

        raise KeyError(
            "\nThe result returned by "
            "feasible_sets_features_nn_xy() "
            "does not contain "
            "'feasible_sets'.\n\n"
            f"Available keys: "
            f"{list(result.keys())}"
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
    # Extract tau_x returned by package
    # ------------------------------------------------------

    returned_tau_x = result.get(
        "tau_x",
        np.nan,
    )

    try:

        returned_tau_x = float(
            returned_tau_x
        )

    except (
        TypeError,
        ValueError,
    ):

        returned_tau_x = np.nan

    # ------------------------------------------------------
    # Extract tau_y
    # ------------------------------------------------------

    tau_y = result.get(
        "tau_y",
        np.nan,
    )

    try:

        tau_y = float(
            tau_y
        )

    except (
        TypeError,
        ValueError,
    ):

        tau_y = np.nan

    return {

        "N_max": int(
            N_max
        ),

        "tau_x": returned_tau_x,

        "tau_x_configured": float(
            tau_x_value
        ),

        "tau_y": tau_y,

        "M_prime": (
            statistics[
                "M_prime"
            ]
        ),

        "Nk_stats": str(
            statistics[
                "Nk_stats"
            ]
        ),

        "Kernel_size": (
            statistics[
                "Kernel_size"
            ]
        ),

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
    }


# ==========================================================
# N_max ABLATION
# ==========================================================

def run_nmax_ablation(
    X,
    y,
    tau_x_value,
    nmax_grid,
):
    """
    Run N_max ablation at a specified q and tau_x.
    """

    rows = []

    tau_x_value = float(tau_x_value)

    print("\n" + "=" * 70)
    print(
        "N_max / KERNEL-SIZE ABLATION"
    )
    print("=" * 70)


    print(
        f"tau_x = "
        f"{tau_x_value:.6f}"
    )

    print(
        f"N_max grid: "
        f"{nmax_grid}"
    )

    for N_max in nmax_grid:

        print(
            f"tau_x={tau_x_value:.6f}, "
            f"N_max={N_max}"
        )

        row = (
            compute_statistics_chf(
                X=X,
                y=y,
                tau_x_value=tau_x_value,
                N_max=N_max,
            )
        )

        rows.append(
            row
        )

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
    """

    g = (
        df.sort_values(
            "N_max"
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    reference_rows = g[
        g["N_max"]
        == reference_nmax
    ]

    if reference_rows.empty:

        raise ValueError(
            f"Reference N_max="
            f"{reference_nmax} "
            f"not found."
        )

    reference_kernel_size = float(
        reference_rows[
            "Kernel_size"
        ].iloc[0]
    )

    tau_x_value = float(
        reference_rows[
            "tau_x_configured"
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

                "tau_x":
                    tau_x_value,

                "N_max":
                    row["N_max"],

                "Kernel_size":
                    row["Kernel_size"],

                "mean_feasible_set_size":
                    row[
                        "mean_feasible_set_size"
                    ],

                "max_feasible_set_size":
                    row[
                        "max_feasible_set_size"
                    ],

                "fraction_at_Nmax":
                    row[
                        "fraction_at_Nmax"
                    ],

                "relative_to_reference":
                    relative_to_reference,

                "saturated":
                    saturated,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# PLOT: KERNEL SIZE VS N_max
# ==========================================================

def plot_kernel_size_vs_nmax(
    nmax_results,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    g = nmax_results.sort_values(
        "N_max"
    )


    tau_x_value = float(
        g["tau_x_configured"].iloc[0]
    )

    plt.figure(
        figsize=(9, 6)
    )

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
        label=(
            f"reference N_max="
            f"{REFERENCE_NMAX}"
        ),
    )

    plt.xlabel(
        "N_max",
        fontsize=12,
    )

    plt.ylabel(
        "Kernel size",
        fontsize=12,
    )

    plt.title(
        "Uniform CHF Dataset: "
        "Kernel Size vs N_max "
        f"tau_x={tau_x_value:.6f}",
        fontsize=13,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        output_dir,
        "chf_uniform_kernel_size_vs_nmax.png",
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: "
        f"{filename}"
    )


# ==========================================================
# PLOT: N_max SATURATION
# ==========================================================

def plot_nmax_saturation(
    nmax_summary,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    g = nmax_summary.sort_values(
        "N_max"
    )


    tau_x_value = float(
        g["tau_x"].iloc[0]
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        g["N_max"],
        g[
            "relative_to_reference"
        ],
        marker="o",
        linewidth=2.0,
        markersize=5,
        color="tab:red",
        label="Relative to reference",
    )

    plt.axhline(
        NMAX_SATURATION_TOL,
        color="black",
        linestyle="--",
        label=(
            f"saturation tolerance="
            f"{NMAX_SATURATION_TOL:.2f}"
        ),
    )

    plt.xlabel(
        "N_max",
        fontsize=12,
    )

    plt.ylabel(
        "Relative difference from reference K",
        fontsize=12,
    )

    plt.title(
        "Uniform CHF Dataset: "
        "N_max Saturation "
        f"tau_x={tau_x_value:.6f}",
        fontsize=13,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        output_dir,
        "chf_uniform_nmax_saturation.png",
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: "
        f"{filename}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    np.random.seed(
        42
    )

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

    os.makedirs(
        PLOT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "UNIFORM CRITICAL HEAT FLUX "
        "FEASIBLE-SET ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:"
        "\n    KAERI/TR-1665/2000"
        "\n    Uniform axial heating"
    )

    print(
        "\nFeature representation:"
        "\n    Diameter"
        "\n    Length"
        "\n    Pressure"
        "\n    MassFlux"
        "\n    EquilibriumQuality"
    )

    print(
        "\nTarget:"
        "\n    Average critical HeatFlux"
    )

    print(
        "\nFeasible-set definition:"
        "\n    d_X(x_k,x_j) <= tau_x"
        "\n        OR"
        "\n    d_Y(y_k,y_j) <= tau_y"
    )


    print(
        f"tau_x = "
        f"{TAU_X:.6f}"
    )

    print(
        f"REFERENCE_NMAX: "
        f"{REFERENCE_NMAX}"
    )

    print(
        f"NMAX_GRID: "
        f"{NMAX_GRID}"
    )

    print(
        "\nN_max saturation rule:"
        f"\n    relative difference <= "
        f"{NMAX_SATURATION_TOL:.3f}"
    )

    # ======================================================
    # LOAD DATA
    # ======================================================

    (
        X,
        y,
        df_raw,
        scaler,
    ) = load_chf_uniform_dataset(
        local_path=LOCAL_XML,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    # ======================================================
    # SAVE CLEANED DATA
    # ======================================================

    cleaned_csv = os.path.join(
        OUTPUT_DIR,
        "chf_uniform_cleaned_data.csv",
    )

    df_raw.to_csv(
        cleaned_csv,
        index=False,
    )

    print(
        "\nCleaned CHF dataset saved to:"
    )

    print(
        f"    {cleaned_csv}"
    )

    # ======================================================
    # q / tau_x CONFIGURATION
    # ======================================================


    tau_x_value = float(
        TAU_X
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "q / tau_x CONFIGURATION"
    )

    print(
        "=" * 70
    )


    print(
        f"tau_x = "
        f"{tau_x_value:.6f}"
    )

    print(
        f"Reference N_max = "
        f"{REFERENCE_NMAX}"
    )

    # ======================================================
    # N_max ABLATION
    # ======================================================

    nmax_results = run_nmax_ablation(
        X=X,
        y=y,
        tau_x_value=tau_x_value,
        nmax_grid=NMAX_GRID,
    )

    nmax_results = (
        nmax_results.round(
            6
        )
    )

    nmax_results.to_csv(
        OUTPUT_NMAX_CSV,
        index=False,
        float_format="%.6f",
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "N_max / KERNEL-SIZE "
        "ABLATION SAVED"
    )

    print(
        "=" * 70
    )

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
        nmax_summary.round(
            6
        )
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
        lambda x:
        f"{x:.6f}",
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"tau_x = "
        f"{tau_x_value:.6f}"
    )

    print(
        f"Reference N_max = "
        f"{REFERENCE_NMAX}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "KERNEL-SIZE ABLATION "
        "OVER N_max"
    )

    print(
        "=" * 70
    )

    print(
        nmax_results.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "N_max SATURATION ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        nmax_summary.to_string(
            index=False
        )
    )

    # ======================================================
    # PLOTS
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING PLOTS"
    )

    print(
        "=" * 70
    )

    plot_kernel_size_vs_nmax(
        nmax_results=nmax_results,
        output_dir=PLOT_DIR,
    )

    plot_nmax_saturation(
        nmax_summary=nmax_summary,
        output_dir=PLOT_DIR,
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"tau_x: "
        f"{tau_x_value:.6f}"
    )

    print(
        f"Cleaned data: "
        f"{cleaned_csv}"
    )

    print(
        f"N_max ablation CSV: "
        f"{OUTPUT_NMAX_CSV}"
    )

    print(
        f"N_max saturation CSV: "
        f"{NMAX_SUMMARY_CSV}"
    )

    print(
        f"Plots: "
        f"{PLOT_DIR}/"
    )

    # ======================================================
    # FINAL CONFIGURATION
    # ======================================================

    print(
        "\nFinal configuration:"
    )

    print(
        f"    tau_x={tau_x_value:.6f}"
    )

    print(
        f"    Reference N_max="
        f"{REFERENCE_NMAX}"
    )

    print(
        f"    N_max grid="
        f"{NMAX_GRID}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()
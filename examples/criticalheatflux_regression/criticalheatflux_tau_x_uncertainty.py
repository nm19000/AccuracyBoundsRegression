#!/usr/bin/env python3
"""
criticalheatflux_tau_x_uncertainty.py

Calculate the physics/measurement-derived input uncertainty
radius tau_x for the KAERI/TR-1665/2000 uniform axial heating
CHF dataset.

Features
--------
    Diameter
    Length
    Pressure
    MassFlux
    EquilibriumQuality

Target
------
    HeatFlux

Definition
----------
For each observation k:

    tau_x_unc,k =
        sqrt(
            (sigma_D / s_D)^2
            + (sigma_L / s_L)^2
            + (sigma_P / s_P)^2
            + (sigma_G / s_G)^2
            + (sigma_xe / s_xe)^2
        )

where

    sigma_i = physical measurement uncertainty
    s_i     = training-set standard deviation of feature i

For relative uncertainties:

    sigma_i = u_i * |X_i|

For absolute uncertainties:

    sigma_i = u_i

The resulting tau_x_unc is expressed in standardized feature
space.

IMPORTANT
---------

It calculates the physics/measurement-derived
uncertainty radius. The final conservative tau_x reported by this script is:

    tau_x = max(tau_x_unc,k)

For the current assumptions and dataset, this should give
approximately:

    tau_x = 0.171840

Uncertainty assumptions
-----------------------
    Diameter              ±1.0 %
    Length                ±1.0 %
    Pressure              ±0.5 %
    MassFlux              ±2.0 %
    EquilibriumQuality    ±0.01 absolute

These are representative assumptions and should not be treated
as official KAERI instrument specifications unless independently
verified from the experimental documentation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "chf_uniform_kernel_results/"
    "chf_uniform_cleaned_data.csv"
)

OUTPUT_DIR = Path(
    "chf_uniform_kernel_results"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


FEATURES = [
    "Diameter",
    "Length",
    "Pressure",
    "MassFlux",
    "EquilibriumQuality",
]

TARGET = "HeatFlux"


# ============================================================
# REPRESENTATIVE INPUT UNCERTAINTIES
# ============================================================

RELATIVE_INPUT_UNCERTAINTIES = {

    "Diameter": 0.01,       # ±1.0 %

    "Length": 0.01,         # ±1.0 %

    "Pressure": 0.005,       # ±0.5 %

    "MassFlux": 0.020,       # ±2.0 %
}


# ============================================================
# ABSOLUTE INPUT UNCERTAINTIES
# ============================================================

ABSOLUTE_INPUT_UNCERTAINTIES = {

    "EquilibriumQuality": 0.010,
}


# ============================================================
# UTILITY
# ============================================================

def print_header(title):
    """Print formatted section header."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_uncertainty_configuration():
    """Ensure every feature has exactly one uncertainty."""

    relative_features = set(
        RELATIVE_INPUT_UNCERTAINTIES.keys()
    )

    absolute_features = set(
        ABSOLUTE_INPUT_UNCERTAINTIES.keys()
    )

    all_features = (
        relative_features |
        absolute_features
    )

    missing = set(FEATURES) - all_features

    if missing:
        raise ValueError(
            "No uncertainty specified for:\n"
            + "\n".join(sorted(missing))
        )

    duplicated = (
        relative_features &
        absolute_features
    )

    if duplicated:
        raise ValueError(
            "Features have both relative and absolute "
            "uncertainties:\n"
            + "\n".join(sorted(duplicated))
        )

    extra = all_features - set(FEATURES)

    if extra:
        raise ValueError(
            "Uncertainty specified for unknown features:\n"
            + "\n".join(sorted(extra))
        )

    for feature, value in (
        RELATIVE_INPUT_UNCERTAINTIES.items()
    ):

        if value < 0:
            raise ValueError(
                f"Relative uncertainty for {feature} "
                "cannot be negative."
            )

    for feature, value in (
        ABSOLUTE_INPUT_UNCERTAINTIES.items()
    ):

        if value < 0:
            raise ValueError(
                f"Absolute uncertainty for {feature} "
                "cannot be negative."
            )


# ============================================================
# CALCULATE TAU_X
# ============================================================

def calculate_tau_x_uncertainty(
    X,
    feature_scales,
    relative_uncertainties,
    absolute_uncertainties,
):
    """
    Calculate row-wise measurement-derived tau_x.

    Returns
    -------
    tau_x_unc : pandas.Series
        RSS uncertainty radius for every observation.

    physical_uncertainty : pandas.DataFrame
        Physical uncertainty sigma_i.

    standardized_uncertainty : pandas.DataFrame
        sigma_i / s_i for each feature.
    """

    physical_uncertainty = {}
    standardized_uncertainty = {}

    for feature in FEATURES:

        values = X[feature].to_numpy(
            dtype=float
        )

        scale = feature_scales[feature]

        if scale <= 0:
            raise ValueError(
                f"Invalid training standard deviation "
                f"for {feature}: {scale}"
            )

        # ----------------------------------------------------
        # Relative uncertainty
        # ----------------------------------------------------

        if feature in relative_uncertainties:

            relative_u = (
                relative_uncertainties[feature]
            )

            sigma_physical = (
                np.abs(values)
                * relative_u
            )

        # ----------------------------------------------------
        # Absolute uncertainty
        # ----------------------------------------------------

        elif feature in absolute_uncertainties:

            absolute_u = (
                absolute_uncertainties[feature]
            )

            sigma_physical = np.full(
                len(values),
                absolute_u,
                dtype=float,
            )

        else:

            raise ValueError(
                f"No uncertainty specified for "
                f"{feature}"
            )

        # ----------------------------------------------------
        # Convert to standardized feature space
        # ----------------------------------------------------

        sigma_standardized = (
            sigma_physical / scale
        )

        physical_uncertainty[feature] = (
            sigma_physical
        )

        standardized_uncertainty[feature] = (
            sigma_standardized
        )

    physical_uncertainty = pd.DataFrame(
        physical_uncertainty,
        index=X.index,
    )

    standardized_uncertainty = pd.DataFrame(
        standardized_uncertainty,
        index=X.index,
    )

    # --------------------------------------------------------
    # Root-sum-square
    # --------------------------------------------------------

    tau_x_unc = np.sqrt(
        np.sum(
            standardized_uncertainty.to_numpy()
            ** 2,
            axis=1,
        )
    )

    tau_x_unc = pd.Series(
        tau_x_unc,
        index=X.index,
        name="tau_x_unc",
    )

    return (
        tau_x_unc,
        physical_uncertainty,
        standardized_uncertainty,
    )


# ============================================================
# PRINT ASSUMPTIONS
# ============================================================

def print_uncertainty_assumptions():

    print_header(
        "INPUT MEASUREMENT UNCERTAINTY ASSUMPTIONS"
    )

    print("\nRelative uncertainties:")

    for feature in FEATURES:

        if feature in RELATIVE_INPUT_UNCERTAINTIES:

            value = (
                RELATIVE_INPUT_UNCERTAINTIES[feature]
                * 100.0
            )

            print(
                f"  {feature:25s}"
                f" ±{value:.3f}%"
            )

    print("\nAbsolute uncertainties:")

    for feature in FEATURES:

        if feature in ABSOLUTE_INPUT_UNCERTAINTIES:

            value = (
                ABSOLUTE_INPUT_UNCERTAINTIES[feature]
            )

            print(
                f"  {feature:25s}"
                f" ±{value:.6f}"
            )

    print("\nNOTE:")

    print(
        "  These values are representative assumptions."
    )

    print(
        "  They are not automatically treated as official"
    )

    print(
        "  KAERI instrument specifications."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "CHF INPUT UNCERTAINTY / TAU_X ANALYSIS"
    )

    print("\nDataset:")

    print(f"  {DATA_FILE}")

    print("\nFeatures:")

    for feature in FEATURES:
        print(f"  {feature}")

    print("\nTarget:")
    print(f"  {TARGET}")

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_uncertainty_configuration()

    print_uncertainty_assumptions()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print_header(
        "LOADING DATASET"
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_FILE}\n\n"
            "Run the CHF cleaning/kernel script first."
        )

    df = pd.read_csv(DATA_FILE)

    print(
        f"\nNumber of observations: "
        f"{len(df)}"
    )

    print(
        f"Number of columns: "
        f"{len(df.columns)}"
    )

    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ========================================================
    # REMOVE MISSING VALUES
    # ========================================================

    df_model = (
        df[required_columns]
        .dropna()
        .copy()
    )

    print_header(
        "MISSING-VALUE HANDLING"
    )

    print(
        f"\nBefore: {len(df)}"
    )

    print(
        f"After : {len(df_model)}"
    )

    print(
        f"Removed: {len(df) - len(df_model)}"
    )

    # ========================================================
    # X / y
    # ========================================================

    X = df_model[FEATURES].copy()
    y = df_model[TARGET].copy()

    # ========================================================
    # FEATURE STATISTICS
    # ========================================================

    print_header(
        "ORIGINAL FEATURE STATISTICS"
    )

    for feature in FEATURES:

        values = X[feature]

        print(f"\n{feature}:")

        print(
            f"  min  = {values.min():,.8g}"
        )

        print(
            f"  max  = {values.max():,.8g}"
        )

        print(
            f"  mean = {values.mean():,.8g}"
        )

        print(
            f"  std  = {values.std():,.8g}"
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print_header(
        "TRAIN / TEST SPLIT"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(
        f"\nTotal observations: "
        f"{len(X)}"
    )

    print(
        f"Training observations: "
        f"{len(X_train)}"
    )

    print(
        f"Testing observations : "
        f"{len(X_test)}"
    )

    print(
        f"\nTest fraction: "
        f"{TEST_SIZE:.2f}"
    )

    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print_header(
        "STANDARDIZATION"
    )

    scaler = StandardScaler()

    scaler.fit(X_train)

    feature_scales = {
        feature: scale
        for feature, scale in zip(
            FEATURES,
            scaler.scale_,
        )
    }

    print(
        "\nTraining-set standard deviations:"
    )

    for feature in FEATURES:

        print(
            f"  {feature:25s}"
            f" {feature_scales[feature]:,.10g}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  The scaler is fitted ONLY on X_train."
    )

    print(
        "  X_test does not influence the scaling."
    )

    # ========================================================
    # CALCULATE TAU_X FOR ALL OBSERVATIONS
    # ========================================================

    print_header(
        "CALCULATING MEASUREMENT-DERIVED TAU_X"
    )

    (
        tau_x_unc,
        physical_uncertainty,
        standardized_uncertainty,
    ) = calculate_tau_x_uncertainty(
        X,
        feature_scales,
        RELATIVE_INPUT_UNCERTAINTIES,
        ABSOLUTE_INPUT_UNCERTAINTIES,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(tau_x_unc) != len(X):

        raise RuntimeError(
            "tau_x_unc does not contain the same "
            "number of rows as X."
        )

    if tau_x_unc.isna().any():

        raise RuntimeError(
            "NaN values found in tau_x_unc."
        )

    # ========================================================
    # TAU_X STATISTICS
    # ========================================================

    print_header(
        "MEASUREMENT-DERIVED TAU_X RESULTS"
    )

    print(
        "\nRow-wise tau_x_unc statistics:"
    )

    print(
        f"  Minimum : "
        f"{tau_x_unc.min():.6f}"
    )

    print(
        f"  5th pct : "
        f"{tau_x_unc.quantile(0.05):.6f}"
    )

    print(
        f"  25th pct: "
        f"{tau_x_unc.quantile(0.25):.6f}"
    )

    print(
        f"  Median  : "
        f"{tau_x_unc.median():.6f}"
    )

    print(
        f"  Mean    : "
        f"{tau_x_unc.mean():.6f}"
    )

    print(
        f"  75th pct: "
        f"{tau_x_unc.quantile(0.75):.6f}"
    )

    print(
        f"  95th pct: "
        f"{tau_x_unc.quantile(0.95):.6f}"
    )

    print(
        f"  Maximum : "
        f"{tau_x_unc.max():.6f}"
    )

    # ========================================================
    # SELECT TAU_X
    # ========================================================

    #
    # Conservative dataset-wide choice:
    #
    #     tau_x = max(tau_x_unc,k)
    #
    # This guarantees that the selected radius is at least
    # as large as the calculated measurement uncertainty for
    # every observation in the dataset.
    #

    tau_x = tau_x_unc.max()

    print_header(
        "SELECTED MEASUREMENT-DERIVED TAU_X"
    )

    print(
        "\nThe dataset-wide tau_x is defined as:"
    )

    print(
        "  tau_x = max(tau_x_unc,k)"
    )

    print(
        "\nTherefore:"
    )

    print(
        f"  tau_x = {tau_x:.6f}"
    )

    # ========================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================

    print_header(
        "STANDARDIZED UNCERTAINTY CONTRIBUTIONS"
    )

    print(
        "\nMean standardized uncertainty:"
    )

    for feature in FEATURES:

        value = (
            standardized_uncertainty[
                feature
            ].mean()
        )

        print(
            f"  {feature:25s}"
            f" {value:.6f}"
        )

    print(
        "\nRMS standardized uncertainty:"
    )

    for feature in FEATURES:

        values = (
            standardized_uncertainty[
                feature
            ].to_numpy()
        )

        rms = np.sqrt(
            np.mean(values ** 2)
        )

        print(
            f"  {feature:25s}"
            f" {rms:.6f}"
        )

    # ========================================================
    # BUILD DETAILED OUTPUT
    # ========================================================

    output_df = X.copy()

    output_df[TARGET] = y

    output_df["split"] = np.where(
        output_df.index.isin(X_train.index),
        "train",
        "test",
    )

    # --------------------------------------------------------
    # Physical uncertainties
    # --------------------------------------------------------

    for feature in FEATURES:

        output_df[
            f"{feature}_sigma_physical"
        ] = physical_uncertainty[feature]

    # --------------------------------------------------------
    # Standardized uncertainty contributions
    # --------------------------------------------------------

    for feature in FEATURES:

        output_df[
            f"{feature}_delta_z"
        ] = standardized_uncertainty[feature]

    # --------------------------------------------------------
    # Row-wise tau_x
    # --------------------------------------------------------

    output_df["tau_x_unc"] = tau_x_unc

    # --------------------------------------------------------
    # Selected dataset-wide tau_x
    # --------------------------------------------------------

    output_df["tau_x_selected"] = tau_x

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_file = (
        OUTPUT_DIR
        / "chf_uniform_tau_x_uncertainty.csv"
    )

    output_df.to_csv(
        detailed_file,
        index=True,
    )

    print_header(
        "DETAILED TAU_X RESULTS SAVED"
    )

    print(
        f"\n{detailed_file}"
    )

    # ========================================================
    # SAVE STANDARDIZATION PARAMETERS
    # ========================================================

    standardization_df = pd.DataFrame({

        "Feature": FEATURES,

        "Training_Mean":
            scaler.mean_,

        "Training_Std":
            scaler.scale_,

    })

    standardization_file = (
        OUTPUT_DIR
        / "tau_x_standardization_parameters.csv"
    )

    standardization_df.to_csv(
        standardization_file,
        index=False,
    )

    print(
        "\nStandardization parameters:"
    )

    print(
        f"  {standardization_file}"
    )

    # ========================================================
    # SAVE UNCERTAINTY ASSUMPTIONS
    # ========================================================

    uncertainty_rows = []

    for feature in FEATURES:

        if feature in RELATIVE_INPUT_UNCERTAINTIES:

            uncertainty_rows.append({

                "Feature": feature,

                "Uncertainty_Type":
                    "relative",

                "Uncertainty":
                    RELATIVE_INPUT_UNCERTAINTIES[
                        feature
                    ],

                "Uncertainty_Percent":
                    100.0
                    * RELATIVE_INPUT_UNCERTAINTIES[
                        feature
                    ],

            })

        else:

            uncertainty_rows.append({

                "Feature": feature,

                "Uncertainty_Type":
                    "absolute",

                "Uncertainty":
                    ABSOLUTE_INPUT_UNCERTAINTIES[
                        feature
                    ],

                "Uncertainty_Percent":
                    np.nan,

            })

    uncertainty_assumptions_df = (
        pd.DataFrame(uncertainty_rows)
    )

    uncertainty_file = (
        OUTPUT_DIR
        / "tau_x_uncertainty_assumptions.csv"
    )

    uncertainty_assumptions_df.to_csv(
        uncertainty_file,
        index=False,
    )

    print(
        "\nUncertainty assumptions:"
    )

    print(
        f"  {uncertainty_file}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame({

        "Statistic": [

            "Minimum",

            "5th_percentile",

            "25th_percentile",

            "Median",

            "Mean",

            "75th_percentile",

            "95th_percentile",

            "Maximum",

            "Selected_tau_x",

        ],

        "Value": [

            tau_x_unc.min(),

            tau_x_unc.quantile(0.05),

            tau_x_unc.quantile(0.25),

            tau_x_unc.median(),

            tau_x_unc.mean(),

            tau_x_unc.quantile(0.75),

            tau_x_unc.quantile(0.95),

            tau_x_unc.max(),

            tau_x,

        ],
    })

    summary_file = (
        OUTPUT_DIR
        / "tau_x_uncertainty_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print(
        "\nSummary:"
    )

    print(
        f"  {summary_file}"
    )

    # ========================================================
    # SAVE FEATURE CONTRIBUTIONS
    # ========================================================

    contribution_rows = []

    for feature in FEATURES:

        values = (
            standardized_uncertainty[
                feature
            ]
        )

        contribution_rows.append({

            "Feature": feature,

            "Mean":
                values.mean(),

            "Median":
                values.median(),

            "RMS":
                np.sqrt(
                    np.mean(
                        values.to_numpy() ** 2
                    )
                ),

            "Minimum":
                values.min(),

            "Maximum":
                values.max(),

        })

    contribution_df = pd.DataFrame(
        contribution_rows
    )

    contribution_file = (
        OUTPUT_DIR
        / "tau_x_feature_contributions.csv"
    )

    contribution_df.to_csv(
        contribution_file,
        index=False,
    )

    print(
        "\nFeature contributions:"
    )

    print(
        f"  {contribution_file}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "COMPLETE"
    )

    print(
        "\nPhysics/measurement-derived tau_x_unc:"
    )

    print(
        f"  Mean   = "
        f"{tau_x_unc.mean():.6f}"
    )

    print(
        f"  Median = "
        f"{tau_x_unc.median():.6f}"
    )

    print(
        f"  P95    = "
        f"{tau_x_unc.quantile(0.95):.6f}"
    )

    print(
        f"  Max    = "
        f"{tau_x_unc.max():.6f}"
    )

    print(
        "\nSelected dataset-wide measurement "
        "uncertainty radius:"
    )

    print(
        f"  tau_x = {tau_x:.6f}"
    )

    print(
        "\nResults directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
criticalheatflux_uniform_models.py

Train and evaluate three regression models on the
KAERI/TR-1665/2000 uniform axial heating CHF dataset.

Models
------
1. Random Forest
2. Extra Trees
3. HistGradientBoosting

Preprocessing
-------------
The five input features are standardized using StandardScaler.

IMPORTANT:
    The scaler is fitted ONLY on the training set.

    X_train
        -> fit StandardScaler
        -> transform X_train

    X_test
        -> transform using the SAME training scaler

The target HeatFlux is NOT standardized, so all reported
RMSE and MAE values remain in physical units of W/m².

Evaluation
----------
- Train RMSE
- Test RMSE
- Train MAE
- Test MAE
- Train R²
- Test R²

Dataset
-------
chf_uniform_kernel_results/chf_uniform_cleaned_data.csv

Outputs
-------
chf_uniform_model_results/
    uniform_model_metrics.csv

    random_forest_test_predictions.csv
    extra_trees_test_predictions.csv
    histgradientboosting_test_predictions.csv

    train_test_split.csv

    feature_importance.csv

    standardization_parameters.csv

    standardized_train_features.csv
    standardized_test_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "chf_uniform_kernel_results/chf_uniform_cleaned_data.csv"
)

OUTPUT_DIR = Path(
    "chf_uniform_model_results"
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
# UTILITY FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    """
    Root mean squared error.
    """
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train a model pipeline and calculate train/test performance.

    The pipeline contains the StandardScaler and regression model,
    so the scaler is fitted only using X_train.
    """

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    train_rmse = rmse(
        y_train,
        y_train_pred,
    )

    test_rmse = rmse(
        y_test,
        y_test_pred,
    )

    train_mae = mean_absolute_error(
        y_train,
        y_train_pred,
    )

    test_mae = mean_absolute_error(
        y_test,
        y_test_pred,
    )

    train_r2 = r2_score(
        y_train,
        y_train_pred,
    )

    test_r2 = r2_score(
        y_test,
        y_test_pred,
    )

    # --------------------------------------------------------
    # PRINT PERFORMANCE
    # --------------------------------------------------------

    print("\nPerformance:")

    print(
        f"  Train RMSE : "
        f"{train_rmse:,.3f} W/m²"
    )

    print(
        f"  Test RMSE  : "
        f"{test_rmse:,.3f} W/m²"
    )

    print(
        f"  Train MAE  : "
        f"{train_mae:,.3f} W/m²"
    )

    print(
        f"  Test MAE   : "
        f"{test_mae:,.3f} W/m²"
    )

    print(
        f"  Train R²   : "
        f"{train_r2:.6f}"
    )

    print(
        f"  Test R²    : "
        f"{test_r2:.6f}"
    )

    # --------------------------------------------------------
    # RETURN METRICS
    # --------------------------------------------------------

    metrics = {
        "Model": name,

        "Train_RMSE_W_m2": train_rmse,
        "Test_RMSE_W_m2": test_rmse,

        "Train_MAE_W_m2": train_mae,
        "Test_MAE_W_m2": test_mae,

        "Train_R2": train_r2,
        "Test_R2": test_r2,
    }

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "index": y_test.index,
        "y_true": y_test.to_numpy(),
        "y_pred": y_test_pred,
        "absolute_error": np.abs(
            y_test.to_numpy() - y_test_pred
        ),
        "squared_error": (
            y_test.to_numpy() - y_test_pred
        ) ** 2,
    })

    return (
        model,
        metrics,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNIFORM CHF REGRESSION MODEL COMPARISON")
    print("=" * 70)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset:")
    print(f"  {DATA_FILE}")

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_FILE}\n\n"
            "Run the CHF cleaning script first, "
            "or update DATA_FILE."
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nNumber of observations: "
        f"{len(df)}"
    )

    print(
        f"Number of columns: "
        f"{len(df.columns)}"
    )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = (
        FEATURES + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # REMOVE MISSING VALUES
    # ========================================================

    df_model = df[
        required_columns
    ].copy()

    n_before = len(
        df_model
    )

    df_model = (
        df_model
        .dropna()
        .reset_index(drop=True)
    )

    n_after = len(
        df_model
    )

    print(
        "\nMissing-value handling:"
    )

    print(
        f"  Before: {n_before}"
    )

    print(
        f"  After : {n_after}"
    )

    print(
        f"  Removed: "
        f"{n_before - n_after}"
    )

    # ========================================================
    # X AND y
    # ========================================================

    X = df_model[
        FEATURES
    ].copy()

    y = df_model[
        TARGET
    ].copy()

    print("\nFeatures:")

    for feature in FEATURES:
        print(
            f"  {feature}"
        )

    print("\nTarget:")
    print(
        f"  {TARGET}"
    )

    print("\nX shape:")
    print(
        f"  {X.shape}"
    )

    # ========================================================
    # ORIGINAL FEATURE STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("ORIGINAL FEATURE STATISTICS")
    print("=" * 70)

    for feature in FEATURES:

        print(
            f"\n{feature}:"
        )

        print(
            f"  min  = "
            f"{X[feature].min():,.6g}"
        )

        print(
            f"  max  = "
            f"{X[feature].max():,.6g}"
        )

        print(
            f"  mean = "
            f"{X[feature].mean():,.6g}"
        )

        print(
            f"  std  = "
            f"{X[feature].std():,.6g}"
        )

    # ========================================================
    # TARGET STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("TARGET STATISTICS")
    print("=" * 70)

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        f"  min  = "
        f"{y.min():,.3f} W/m²"
    )

    print(
        f"  max  = "
        f"{y.max():,.3f} W/m²"
    )

    print(
        f"  mean = "
        f"{y.mean():,.3f} W/m²"
    )

    print(
        f"  std  = "
        f"{y.std():,.3f} W/m²"
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
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

    print("\n" + "=" * 70)
    print("FEATURE STANDARDIZATION")
    print("=" * 70)

    print(
        "\nStandardScaler is fitted ONLY on X_train."
    )

    print(
        "X_test is transformed using the "
        "training-set scaler."
    )

    print(
        "\nTarget HeatFlux is NOT standardized."
    )

    print(
        "Therefore RMSE and MAE remain in W/m²."
    )

    # --------------------------------------------------------
    # Fit an explicit scaler for diagnostics/saving.
    #
    # The actual models below each contain their own scaler
    # inside a Pipeline. This explicit scaler is only used to
    # inspect and save the same train-fitted transformation.
    # --------------------------------------------------------

    diagnostic_scaler = StandardScaler()

    X_train_scaled = (
        diagnostic_scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        diagnostic_scaler.transform(
            X_test
        )
    )

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES,
        index=X_train.index,
    )

    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES,
        index=X_test.index,
    )

    # --------------------------------------------------------
    # STANDARDIZED TRAINING STATISTICS
    # --------------------------------------------------------

    print(
        "\nTraining-set standardized statistics:"
    )

    for feature in FEATURES:

        train_mean = (
            X_train_scaled_df[
                feature
            ].mean()
        )

        train_std = (
            X_train_scaled_df[
                feature
            ].std(
                ddof=0
            )
        )

        test_mean = (
            X_test_scaled_df[
                feature
            ].mean()
        )

        test_std = (
            X_test_scaled_df[
                feature
            ].std(
                ddof=0
            )
        )

        print(
            f"  {feature:25s} "
            f"train mean={train_mean: .6f}, "
            f"train std={train_std: .6f}, "
            f"test mean={test_mean: .6f}, "
            f"test std={test_std: .6f}"
        )

    # ========================================================
    # SAVE STANDARDIZED DATA
    # ========================================================

    standardized_train_file = (
        OUTPUT_DIR /
        "standardized_train_features.csv"
    )

    standardized_test_file = (
        OUTPUT_DIR /
        "standardized_test_features.csv"
    )

    X_train_scaled_df.to_csv(
        standardized_train_file,
        index=True,
        index_label="index",
    )

    X_test_scaled_df.to_csv(
        standardized_test_file,
        index=True,
        index_label="index",
    )

    print(
        "\nStandardized training features saved:"
    )

    print(
        f"  {standardized_train_file}"
    )

    print(
        "Standardized test features saved:"
    )

    print(
        f"  {standardized_test_file}"
    )

    # ========================================================
    # SAVE STANDARDIZATION PARAMETERS
    # ========================================================

    scaler_file = (
        OUTPUT_DIR /
        "standardization_parameters.csv"
    )

    scaler_df = pd.DataFrame({
        "Feature": FEATURES,
        "Training_Mean": diagnostic_scaler.mean_,
        "Training_Scale": diagnostic_scaler.scale_,
    })

    scaler_df.to_csv(
        scaler_file,
        index=False,
    )

    print(
        "\nStandardization parameters saved:"
    )

    print(
        f"  {scaler_file}"
    )

    # ========================================================
    # DEFINE MODELS
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print(
        "\nEach model uses:"
    )

    print(
        "  StandardScaler -> regression model"
    )

    models = {

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "Random Forest": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=1.0,
                    bootstrap=True,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        # ----------------------------------------------------
        # EXTRA TREES
        # ----------------------------------------------------

        "Extra Trees": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=1.0,
                    bootstrap=False,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        # ----------------------------------------------------
        # HISTOGRAM GRADIENT BOOSTING
        # ----------------------------------------------------

        "HistGradientBoosting": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=500,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    all_metrics = []

    all_predictions = {}

    trained_models = {}

    for name, model in models.items():

        (
            trained_model,
            metrics,
            predictions,
        ) = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        trained_models[
            name
        ] = trained_model

        all_metrics.append(
            metrics
        )

        all_predictions[
            name
        ] = predictions

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        all_metrics
    )

    results_df = (
        results_df
        .sort_values(
            "Test_RMSE_W_m2"
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.4f}",
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_test_rmse = (
        results_df.iloc[0][
            "Test_RMSE_W_m2"
        ]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"\nModel: "
        f"{best_model_name}"
    )

    print(
        f"Test RMSE: "
        f"{best_test_rmse:,.3f} W/m²"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_file = (
        OUTPUT_DIR /
        "uniform_model_metrics.csv"
    )

    results_df.to_csv(
        metrics_file,
        index=False,
    )

    print(
        "\nMetrics saved:"
    )

    print(
        f"  {metrics_file}"
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    print(
        "\nTest predictions saved:"
    )

    for name, predictions in (
        all_predictions.items()
    ):

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        prediction_file = (
            OUTPUT_DIR /
            f"{safe_name}_test_predictions.csv"
        )

        predictions.to_csv(
            prediction_file,
            index=False,
        )

        print(
            f"  {prediction_file}"
        )

    # ========================================================
    # SAVE TRAIN / TEST SPLIT
    # ========================================================

    split_df = pd.DataFrame({
        "index": X.index,
        "split": np.where(
            X.index.isin(
                X_train.index
            ),
            "train",
            "test",
        ),
    })

    split_file = (
        OUTPUT_DIR /
        "train_test_split.csv"
    )

    split_df.to_csv(
        split_file,
        index=False,
    )

    print(
        f"\nTrain/test split saved:"
    )

    print(
        f"  {split_file}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance_rows = []

    for name, pipeline in (
        trained_models.items()
    ):

        # ----------------------------------------------------
        # Extract the underlying tree model.
        # ----------------------------------------------------

        model = pipeline.named_steps[
            "model"
        ]

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

            print(
                f"\n{name}:"
            )

            for feature, value in zip(
                FEATURES,
                importance,
            ):

                print(
                    f"  {feature:25s} "
                    f"{value:.6f}"
                )

                feature_importance_rows.append({
                    "Model": name,
                    "Feature": feature,
                    "Importance": value,
                })

        else:

            print(
                f"\n{name}:"
            )

            print(
                "  Feature importance "
                "not available."
            )

    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    if feature_importance_rows:

        importance_df = (
            pd.DataFrame(
                feature_importance_rows
            )
        )

        importance_file = (
            OUTPUT_DIR /
            "feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False,
        )

        print(
            "\nFeature importance saved:"
        )

        print(
            f"  {importance_file}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "\nResults directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nPreprocessing:"
    )

    print(
        "  Input features standardized: YES"
    )

    print(
        "  Scaler fitted on training data only: YES"
    )

    print(
        "  Test data used during scaler fitting: NO"
    )

    print(
        "  Target standardized: NO"
    )

    print(
        "\nBest model:"
    )

    print(
        f"  {best_model_name}"
    )

    print(
        "\nBest test RMSE:"
    )

    print(
        f"  {best_test_rmse:,.3f} W/m²"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
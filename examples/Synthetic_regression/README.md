
# Synthetic Regression Accuracy Bounds

This example shows how to use the `accuracy_bounds` package to estimate achievable regression-error bounds for a dataset and compare them with trained models.

The workflow is designed to be reusable: replace the example dataset or regression models with your own data and models while keeping the same analysis steps.

## Overview

Given input data X and targets y, the example:

1. Generates a clean synthetic regression surface.
2. Adds optional input and target noise.
3. Splits the data into training and test sets.
4. Constructs feasible sets from locally similar observations.
5. Computes the average kernel size.
6. Trains several regression models.
7. Compares model errors with the estimated bounds.
8. Measures model sensitivity using an empirical Lipschitz estimate.

The kernel-size estimate is based on the target variation among inputs that are considered indistinguishable at the selected resolution:

$$
\frac{1}{2}K
\leq
\mathcal L
\leq
K,
\qquad
K=Kersize(\mathcal D_{M'},2)_K.
$$

Here, $$K=Kersize(\mathcal D_{M'},2)_K$$ is an estimate of the irreducible error scale associated with ambiguity in the dataset.

## Installation

```bash
pip install numpy pandas scikit-learn
pip install -e .
```

The second command installs the local `accuracy_bounds` package from the repository root.

## Running the Example

```bash
python path/to/synthetic_regression_example.py
```

The script writes its results to:

```text
synthetic_regression_results.csv
```

## Minimal Workflow

The essential API consists of three steps:

```python
from accuracy_bounds.regressionproblems.feasible_sets import (
    feasible_sets_features_nn_xy,
)
from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)

# X: input features, shape (n_samples, n_features)
# y: targets, shape (n_samples,) or (n_samples, 1)

result = feasible_sets_features_nn_xy(
    X,
    y.reshape(-1, 1),
    N_max=300,
    qx=0.05,
    tau_y=0.03,
    ext_D=True,
)

K, M_prime, set_statistics = compute_kernel_size(
    result["feasible_sets"]
)
```

The resulting value `K` is the estimated average kernel size. `result["D_prime"]` contains the enlarged dataset generated from the feasible sets.

## What Are `qx` and `tau_y`?

The two main resolution parameters are:

- `qx`: controls the input-space threshold $$\tau_{X}$$;
- `tau_y`: specifies the target-space threshold $$\tau_{Y}$$.

They define when observations are treated as locally comparable:

$$
d_{X}(x_i,x_j)
\leq
\tau_{\mathcal X}
$$

or

$$
d_{Y}(y_i,y_j)
\leq
\tau_{Y}.
$$

In practical terms:

- `qx` controls how far apart inputs may be while still being considered similar;
- `tau_y` controls how close target values must be to satisfy the target-space criterion.

These are not model hyperparameters. They define the resolution at which ambiguity is measured. Their values should reflect the scale and noise level of the application.

For example:

```python
result = feasible_sets_features_nn_xy(
    X,
    y.reshape(-1, 1),
    N_max=300,
    qx=0.05,
    tau_y=0.03,
    ext_D=True,
)
```

Smaller thresholds produce more local feasible sets. Larger thresholds may include more observations and measure ambiguity at a broader scale.

## Data Preparation

Input features should generally be placed on comparable scales before distance-based analysis. Otherwise, features with larger numerical units can dominate the input distance.

The example standardizes features inside each model pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", regressor),
])
```

When using the feasible-set functions, ensure that the distance calculation is also appropriate for the scale of `X`. Depending on the application, this may mean standardizing `X` before feasible-set construction or supplying a suitable metric implementation.

Target scaling should be considered separately. If `y` is rescaled, `tau_y`, kernel sizes, and RMSE values are rescaled as well.

## Models

The example compares three model families:

- Gaussian process regression (`"GP"`);
- multilayer perceptrons (`"MLP"`);
- support vector regression (`"SVR"`).

Candidate models are created with:

```python
models = build_candidates("GP")
```

To use a different model, either add it to `build_candidates()` or pass a fitted estimator to your own evaluation code. The accuracy-bound calculation is independent of the model architecture.

## Model Evaluation

The selected models are evaluated using:

- training RMSE;
- test RMSE;
- RMSE on the enlarged dataset $$D'$$;
- empirical Lipschitz estimate $$L$$.

The empirical Lipschitz estimate is calculated as

$$
 L
=
\max_{i,j}
\frac{
|y_i- y_j|
}{
\|x_i-x_j\|_2
}.
$$

It is an approximate sensitivity measure based on sampled input pairs, not an exact global Lipschitz constant.

## Accuracy Regimes

The example demonstrates two model-selection regimes relative to the kernel size $$K$$.

### High-Accuracy Regime

The `"unstable"` regime selects a model with very low training error:

$$
RMSE_{\mathrm{train}} < 0.4K.
$$

This model may fit the observed training data closely but can be more sensitive to small input changes.

### Stable Regime

The `"stable"` regime selects a model whose training error lies within a relaxed kernel-size interval:

$$
0.8K
\leq
RMSE_{\mathrm{train}}
\leq
K.
$$

Among models in this interval, the model with the smallest empirical Lipschitz estimate is selected.

These regimes are illustrative selection rules, not requirements of the package. Users can define their own model-selection criteria.

## Using Your Own Dataset

Replace the synthetic data-generation step with your own data:

```python
X = your_features
y = your_targets

K, result, M_prime, statistics = compute_statistics(
    X,
    y,
    q=0.05,
)
```

Then train any regression model using the same training data and compare its predictions with:

```python
train_rmse = rmse(y_train, model.predict(X_train))
test_rmse = rmse(y_test, model.predict(X_test))
```

A typical application workflow is:

```python
# 1. Prepare and scale X if necessary
# 2. Compute feasible sets and K
# 3. Train a regression model
# 4. Evaluate train and test RMSE
# 5. Compare the observed errors with K/2 and K
# 6. Optionally evaluate on result["D_prime"]
```

The package does not require a particular regression algorithm.

## Output

The generated CSV contains, among other fields:

| Column | Description |
|---|---|
| `q` | Input-distance quantile |
| `tau_x train` | Training input threshold |
| `tau_x test` | Test input threshold |
| `Kernel train` | Training average kernel size |
| `Kernel test` | Test average kernel size |
| `RMSE train` | Training error |
| `RMSE test` | Test error |
| `RMSE D'` | Error on the enlarged dataset |
| `Lhat` | Empirical Lipschitz estimate |
| `Model` | Model family and selection regime |

Models with a `^o` suffix, such as `GP^o`, use the high-accuracy regime. Unsuffixed models, such as `GP`, use the stable regime.

## Interpreting the Results

The kernel size is a dataset-level estimate, not a guarantee that every model's error will fall exactly inside the interval

$$
[K/2,K]= [Kersize(\mathcal D_{M'},2)_K/2,Kersize(\mathcal D_{M'},2)_K]
$$

Observed errors can differ because of:

- finite-sample effects;
- model misspecification;
- optimization;
- train/test splitting;
- differences between the original dataset and the enlarged dataset \(D'\);
- the selected input and target metrics;
- the values of `qx` and `tau_y`.

The purpose of the comparison is to assess whether model performance is consistent with the ambiguity measured in the data, and to illustrate the tradeoff between low training error and prediction stability.

## Reproducibility and Limitations

The example fixes NumPy's random seed to `42`. Results may still vary with different versions of NumPy, pandas, or scikit-learn.

The kernel-size estimate depends on the chosen metrics, thresholds, and sampling limits. Therefore, `qx`, `tau_y`, and `N_max` should be documented and justified for each application rather than copied unchanged from this example.

                        
                        
                                

                                

                                
                                
                                    
                                

                                
                                

                                
                            
                        
                    

                    
                        
                            
                            
                            
                                
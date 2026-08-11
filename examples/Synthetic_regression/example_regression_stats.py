import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
    feasible_sets_features_nn_xy,
)

from accuracy_bounds.regressionproblems.kersize_compute import (
    compute_kernel_size,
)


# ==========================================================
# Dataset
# ==========================================================

def generate_dataset(
        bounds,
        n_sqrt_mesh,
        input_noise,
        output_noise):

    x = np.linspace(-bounds, bounds, n_sqrt_mesh)
    y = np.linspace(-bounds, bounds, n_sqrt_mesh)

    X1, X2 = np.meshgrid(x, y)

    X = np.column_stack([
        X1.ravel(),
        X2.ravel()
    ])

    # U([0,noise])
    X += np.random.uniform(
        0,
        input_noise,
        size=X.shape
    )

    R = np.sqrt(
        X[:,0]**2 +
        X[:,1]**2
    )

    y = np.sin(1 + 0.9*R)

    y += np.random.uniform(
        0,
        output_noise,
        size=y.shape
    )

    return X, y



# ==========================================================
# Kernel size computation
# ==========================================================

def compute_statistics(X, y, q, N_max=300):

    result = feasible_sets_features_nn_xy(
        X,
        y.reshape(-1,1),
        N_max=N_max,
        qx=q,
        tau_y=0.03,
        ext_D=False,
    )

    ksize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"]
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
        N_max=300):


    K_train, train_result, M_prime_train, Nk_stats_train = compute_statistics(
        X_train,
        y_train,
        q,
        N_max=N_max
    )

    K_test, test_result, M_prime_test, Nk_stats_test = compute_statistics(
            X_test,
            y_test,
            q,
            N_max=N_max
        )


    rows=[]


    rows.append({

            "q":q,

            "N_max":N_max,

            "tau_x train":
            train_result["tau_x"],

            "tau_x test":
            test_result["tau_x"],

            "M_prime_train":
            M_prime_train, 

            "Nk_stats_train":
            Nk_stats_train,

            "M_prime_test":
            M_prime_test, 
            
            "Nk_stats_test":
            Nk_stats_test,

            "Kernel train":
            K_train,

            "Kernel test":
            K_test,
        })



    return rows



# ==========================================================
# Main
# ==========================================================

def main():


    np.random.seed(42)



    # noisy dataset

    X,y = generate_dataset(
        bounds=2,
        n_sqrt_mesh=30,
        input_noise=0.01,
        output_noise=0.03
    )



    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.9,
        random_state=42
    )


    qs=[
        0.01,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20
    ]

    N_max=[10,20,30,40,50,100,200,300,450]

    results=[]

    q=0.1

    #for q in qs:
    for N_max in N_max:

        results.extend(
            run_experiment(
                q,
                X_train,
                X_test,
                y_train,
                y_test,
                N_max=N_max
            )
        )



    df = pd.DataFrame(results)

    # Round numerical columns for reporting
    round_cols = [
        "Kernel train",
        "Kernel test",
        "q",
        "tau_x train",
        "tau_x test"
    ]

    df[round_cols] = df[round_cols].round(6)

    # Optional: nicer display in terminal
    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}"
    )

    print(df)

    df.to_csv(
        "synthetic_regression_results.csv",
        index=False,
        float_format="%.6f"
    )


if __name__=="__main__":

    main()
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

def compute_statistics(X, y, q):

    result = feasible_sets_features_nn_xy(
        X,
        y.reshape(-1,1),
        N_max=300,
        qx=q,
        tau_y=0.01,
        ext_D=True,
    )

    ksize, _, _ = compute_kernel_size(
        result["feasible_sets"]
    )

    return ksize, result



# ==========================================================
# Lipschitz estimate
# ==========================================================

def empirical_lipschitz_constant(
        model,
        X,
        n_pairs=100000,
        eps=1e-12):

    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1,1)

    n = len(X)

    i = np.random.randint(
        0,n,n_pairs
    )

    j = np.random.randint(
        0,n,n_pairs
    )

    dx = np.linalg.norm(
        X[i]-X[j],
        axis=1
    )

    dy = np.abs(
        y_pred[i]-y_pred[j]
    ).flatten()

    mask = dx > eps

    return np.max(
        dy[mask]/dx[mask]
    )



def rmse(y,yp):

    return np.sqrt(
        mean_squared_error(y,yp)
    )



# ==========================================================
# Candidate models
# ==========================================================

def build_candidates(name):

    models=[]


    if name=="MLP":

        for layers in [
        (32,),
        (64,),
        (128,),
        (64,64),
        (128,128),
        (128,64,32),
        ]:

            for alpha in [
                0,
                1e-6,
                1e-5,
                1e-4,
            ]:

                models.append(
                    Pipeline([
                    (
                    "scaler",
                    StandardScaler()
                    ),
                    (
                    "model",
                    MLPRegressor(
                    hidden_layer_sizes=layers,
                    alpha=alpha,
                    max_iter=50000,
                    learning_rate_init=1e-3,
                    random_state=42
                        )
                    )]))
                


    elif name=="SVR":

        for C in [
            10,
            100,
            1000,
            10000,
            ]:

            for eps in [
                1e-5,
                1e-4,
                1e-3,
            ]:

                for gamma in [
                    "scale",
                    1,
                    10,
                    100,
                ]:

                    models.append(
                    Pipeline([
                        (
                        "scaler",
                        StandardScaler()
                        ),
                        (
                        "model",
                        SVR(
                            kernel="rbf",
                            C=C,
                            epsilon=eps,
                            gamma=gamma
                        )
                        )
                        ])
                        )



    elif name=="GP":

        for length in [
            0.2,
            0.5,
            1.0,
            2.0,
        ]:

            for noise in [
                1e-3,
                1e-2,
                5e-2,
            ]:

                kernel = (
                    ConstantKernel(1.0)
                    *
                    RBF(length_scale=length)
                    +
                    WhiteKernel(noise)
                )


                models.append(
                    Pipeline([
                        (
                        "scaler",
                        StandardScaler()
                        ),
                        (
                        "model",
                        GaussianProcessRegressor(
                            kernel=kernel,
                            normalize_y=True,
                            optimizer=None,
                            random_state=42
                        )
                        )
                    ])
                )

    else:
        raise ValueError(name)


    return models


# ==========================================================
# Train models according to accuracy regime
# ==========================================================

def train_to_target(
        model_name,
        X_train,
        y_train,
        X_test,
        y_test,
        X_prime,
        y_prime,
        K,
        mode,
        X_lip):


    candidates = build_candidates(model_name)


    best_model = None
    best_train_error = None
    best_L = np.inf


    fitted_models = []


    for model in candidates:

        model.fit(
            X_train,
            y_train
        )


        train_error = rmse(
            y_train,
            model.predict(X_train)
        )


        L = empirical_lipschitz_constant(
            model,
            X_lip
        )


        fitted_models.append(
            (
                model,
                train_error,
                L
            )
        )


        # --------------------------------------------------
        # phi^o : unstable high accuracy model
        # --------------------------------------------------

        if mode=="unstable":

            if train_error < 0.4*K:

                if (
                    best_model is None
                    or train_error < best_train_error
                ):

                    best_model = model
                    best_train_error = train_error
                    best_L = L



        # --------------------------------------------------
        # phi : stable model
        # --------------------------------------------------

        elif mode=="stable":

            if (
                0.8*K <= train_error <= K
            ):

                # choose smoothest model

                if L < best_L:

                    best_model = model
                    best_train_error = train_error
                    best_L = L



        else:

            raise ValueError(
                "mode must be stable or unstable"
            )



    # ------------------------------------------------------
    # fallback if no model satisfies condition
    # ------------------------------------------------------

    if best_model is None:

        print(
            f"Warning: {model_name} "
            f"has no candidate in {mode} regime"
        )


        if mode=="unstable":

            # closest to interpolation
            best_model, best_train_error, best_L = min(
                fitted_models,
                key=lambda x:x[1]
            )


        else:

            # closest to target interval
            best_model, best_train_error, best_L = min(
                fitted_models,
                key=lambda x:
                abs(x[1]-0.7*K)
            )



    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    test_error = rmse(
        y_test,
        best_model.predict(X_test)
    )


    prime_error = rmse(
        y_prime,
        best_model.predict(X_prime)
    )


    return (
        best_train_error,
        test_error,
        prime_error,
        best_L,
    )



# ==========================================================
# Experiment for one q
# ==========================================================

def run_experiment(
        q,
        X_train,
        X_test,
        y_train,
        y_test,
        X_lip):


    K_test, test_sets = compute_statistics(
        X_train,
        y_train,
        q
    )


    D_prime = test_sets["D_prime"]


    X_prime = np.vstack(
        [
            x for x,_ in D_prime
        ]
    )


    y_prime = np.array(
        [
            yy for _,yy in D_prime
        ]
    )


    rows=[]


    for name in [
        "GP",
        "MLP",
        "SVR"
    ]:


        print(
            f"q={q}, training {name}"
        )



        # ==================================================
        # phi_n^o
        # high accuracy unstable
        # ==================================================

        (
            train_o,
            test_o,
            prime_o,
            L_o

        ) = train_to_target(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_prime,
            y_prime,
            K_test,
            mode="unstable",
            X_lip=X_lip
        )


        rows.append({

            "q":q,

            "Model":
            name+"^o",

            "Kernel test":
            K_test,

            "RMSE train":
            train_o,

            "RMSE test":
            test_o,

            "RMSE D'":
            prime_o,

            "Lhat":
            L_o
        })



        # ==================================================
        # phi_n
        # stable relaxed accuracy
        # ==================================================

        (
            train_s,
            test_s,
            prime_s,
            L_s

        ) = train_to_target(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_prime,
            y_prime,
            K_test,
            mode="stable",
            X_lip=X_lip
        )



        rows.append({

            "q":q,

            "Model":
            name,

            "Kernel test":
            K_test,

            "RMSE train":
            train_s,

            "RMSE test":
            test_s,

            "RMSE D'":
            prime_s,

            "Lhat":
            L_s
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



    # ------------------------------------------------------
    # clean grid for Lipschitz estimation
    # ------------------------------------------------------

    X_lip,_ = generate_dataset(
        bounds=2,
        n_sqrt_mesh=100,
        input_noise=0,
        output_noise=0
    )



    qs=[
        0.01,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20
    ]



    results=[]


    for q in qs:

        results.extend(
            run_experiment(
                q,
                X_train,
                X_test,
                y_train,
                y_test,
                X_lip
            )
        )



    df = pd.DataFrame(results)

    # Round numerical columns for reporting
    round_cols = [
        "Kernel test",
        "RMSE train",
        "RMSE test",
        "RMSE D'",
        "Lhat"
    ]

    df[round_cols] = df[round_cols].round(6)

    # Optional: nicer display in terminal
    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}"
    )

    print(df)

    df.to_csv(
        "synthetic_tradeoff_results.csv",
        index=False,
        float_format="%.6f"
    )


if __name__=="__main__":

    main()
import numpy as np
from sklearn.neighbors import NearestNeighbors


def feasible_sets_features_nnx(
    features_x,
    features_y,
    N_max=50,
    qx=0.05,
    ext_D=False,
):
    """
    Construct feasible sets for a collection of images using wavelet features.

    Wavelet features are extracted from each image and a k-nearest-neighbor
    search is performed in the resulting feature space. For each sample k, the
    feasible set consists of neighboring traget samples y_j where x_j, from (x_j,y_j), satisfies

        d_X(x_k, x_j) <= tau_x,

    where tau_x is estimated from the distribution of nearest-neighbor
    distances in feature space.

    Args:
        features_x:
            Array of shape ``(N, D)`` containing the input features.
        features_y:
            Array of shape ``(N, D')`` containing the target features.
        N_max:
            Maximum number of nearest neighbors considered for each sample.
        qx:
            Quantile used to estimate the feature-space threshold ``tau_x``.
        wavelet:
            Wavelet used to compute the image features.

    Returns:
        dict containing:
            - ``tau_x``: Estimated feature-space threshold.
            - ``feasible_sets``: List whose k-th entry contains the indices of
              the feasible set associated with sample k.
            - ``distances``: k-nearest-neighbor distances in feature space.
            - ``indices``: Indices corresponding to ``distances``.
    """

    N = len(features_x)



    # --------------------------------------------------
    # kNN search in input features
  
    n_neighbors = min(N_max + 1, N)

    nbrs_x = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )
    nbrs_x.fit(features_x)

    distances_x, indices_x = nbrs_x.kneighbors(features_x)

    # --------------------------------------------------
    # Global thresholds

    nearest_dist_x = distances_x[:, 1]
    tau_x = float(np.quantile(nearest_dist_x, qx))

    # --------------------------------------------------
    # Compute feasible sets

    feasible_sets = []
    D_prime = []

    for k in range(N):

        feasible = []

        for idx, dist in zip(indices_x[k], distances_x[k]):

            if dist <= tau_x:

                feasible.append(features_y[idx])
                if ext_D:
                    D_prime.append((features_x[k], features_y[idx]))

        feasible_sets.append(np.array(feasible))


    return {
    "tau_x": tau_x,
    "feasible_sets": feasible_sets,
    "distances": distances_x,
    "indices": indices_x ,
    "D_prime": D_prime
}

def feasible_sets_features_nn_xy(
    features_x,
    features_y,
    N_max=50,
    qx=0.05,
    qy=0.05,
    tau_x = 0.0,
    tau_y = 0.0,
    ext_D=False,
):
    """
    Construct feasible sets for a collection of images using wavelet features.

    Wavelet features are extracted from each image and a k-nearest-neighbor
    search is performed in the resulting feature space. For each sample k, the
    feasible set consists of neighboring traget samples y_j where (x_j,y_j) satisfy

        d_X(x_k, x_j) <= tau_x    or    d_y(y_k, y_j) <= tau_y,

    where tau_x is estimated from the distribution of nearest-neighbor
    distances in feature space, and tau_y is either supplied by the user or
    estimated from the corresponding target differences.

    Args:
        features_x:
            Array of shape ``(N, D)`` containing the input features.
        features_y:
            Array of shape ``(N, D')`` containing the target features.
        N_max:
            Maximum number of nearest neighbors considered for each sample.
        qx:
            Quantile used to estimate the feature-space threshold ``tau_x``.
        wavelet:
            Wavelet used to compute the image features.

    Returns:
        dict containing:
            - ``tau_x``: Estimated feature-space threshold.
            - ``tau_y``: Estimated or user-specified target-space noise threshold.
            - ``feasible_sets``: List whose k-th entry contains the indices of
              the feasible set associated with sample k.
            - ``distances``: k-nearest-neighbor distances in feature space.
            - ``indices``: Indices corresponding to ``distances``.
    """

    N = len(features_x)



    # --------------------------------------------------
    # kNN search in input features
  
    n_neighbors = min(N_max + 1, N)

    nbrs_x = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )
    nbrs_x.fit(features_x)

    distances_x, indices_x = nbrs_x.kneighbors(features_x)

    # --------------------------------------------------
    # Global thresholds input data

    if tau_x == 0.0:
        nearest_dist_x = distances_x[:, 1]
        tau_x = float(np.quantile(nearest_dist_x, qx))

    # kNN search in target space 
    
    nbrs_y = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )
    nbrs_y.fit(features_y)

    distances_y, indices_y = nbrs_y.kneighbors(features_y)

    # --------------------------------------------------
    # Global thresholds target data

    if tau_y == 0.0:
        nearest_dist_y = distances_y[:, 1]
        tau_y = float(np.quantile(nearest_dist_y, qy))

    # --------------------------------------------------
    # Compute feasible sets

    feasible_sets = []
    D_prime = []

    for k in range(N):

        feasible = []

        for idx, dist in zip(indices_x[k], distances_x[k]):

            if dist <= tau_x:

                feasible.append(features_y[idx])
                if ext_D:
                    D_prime.append((features_x[k], features_y[idx]))
        
        for idx, dist in zip(indices_y[k], distances_y[k]):

            if dist <= tau_y:

                feasible.append(features_y[idx])
                if ext_D:
                    D_prime.append((features_x[k], features_y[idx]))

        feasible_sets.append(np.array(feasible))


    return {
    "tau_x": tau_x,
    "tau_y": tau_y,
    "feasible_sets": feasible_sets,
    "distances_x": distances_x,
    "indices_x": indices_x ,
    "distances_y": distances_y,
    "D_prime": D_prime
}
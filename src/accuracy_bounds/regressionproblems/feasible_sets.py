import numpy as np
from sklearn.neighbors import NearestNeighbors


def feasible_sets_features_nnx(
    features_x,
    features_y,
    N_max=50,
    qx=0.05,
    ext_D=False,
    metricx='minkowski',
    px=2,
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
        ext_D:
            If True, also construct the extended dataset ``D_prime``.
        metricx:
            Distance metric to use for nearest-neighbor search on inputs (features). Default is 'minkowski', 
            results in l2 norm for p=2.
        px:
            Power parameter for the Minkowski metric. Default is 2 (Euclidean distance).

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
        algorithm="auto",
        metric=metricx, 
        p=px
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
    tau_x=0.0,
    tau_y=0.0,
    target_uncertainty=None,
    ext_D=False,
    metricx='minkowski',
    px=2,
    metricy='minkowski',
    py=2
        ):
    """ Construct feasible sets using nearest neighbors in both input and target spaces. 
    For each sample ``(x_k, y_k)``, the feasible set consists of target values
    ``y_j`` whose corresponding samples satisfy

    d_X(x_k, x_j) <= tau_x or d_Y(y_k, y_j) <= tau_Y,

    where ``tau_x`` is estimated from the distribution of nearest-neighbor
    distances in the input feature space unless supplied explicitly. The target-space threshold ``tau_Y`` may either be

    * a global constant ``tau_y`` (default behavior), or
    * a sample-dependent uncertainty model provided through
    ``target_uncertainty``.

    When ``target_uncertainty`` is supplied, two target samples are considered
    feasible whenever

    d_Y(y_k, y_j) <= max(sigma(y_k),sigma(y_j)),

    where ``sigma(y) = target_uncertainty(y)`` is the measurement uncertainty
    associated with target value ``y``.

    Args:
        features_x : ndarray of shape (N, D)
            Input feature vectors.

        features_y : ndarray of shape (N, D')
            Target vectors.

        N_max : int, default=50
            Maximum number of nearest neighbors considered.

        qx : float, default=0.05
            Quantile used to estimate ``tau_x`` when ``tau_x == 0``.

        qy : float, default=0.05
            Quantile used to estimate ``tau_y`` when ``tau_y == 0`` and no
            ``target_uncertainty`` model is supplied.

        tau_x : float, default=0
            Input-space threshold. If zero, it is estimated automatically.

        tau_y : float, default=0
            Constant target-space threshold. Ignored when
            ``target_uncertainty`` is provided.

        target_uncertainty : callable or None, default=None
            Function mapping a target value to its measurement uncertainty.
            If supplied, feasible target neighbors are determined using
            sample-dependent thresholds instead of the global ``tau_y``.

        ext_D : bool, default=False
            If True, also construct the extended dataset ``D_prime``.

        metricx:
            Distance metric to use for nearest-neighbor search on inputs (features). Default is 'minkowski', 
            results in l2 norm for p=2.
        px:
            Power parameter for the Minkowski metric. Default is 2 (Euclidean distance).

        metricy:
            Distance metric to use for nearest-neighbor search on targets (features). Default is 'minkowski', 
                    results in l2 norm for p=2.
        py:
            Power parameter for the Minkowski metric. Default is 2 (Euclidean distance).

    Returns
    -------
    dict
        Dictionary containing

        * ``tau_x`` : input-space threshold.
        * ``tau_y`` : constant target-space threshold.
        * ``feasible_sets`` : feasible target sets.
        * ``distances_x`` : input-space neighbor distances.
        * ``indices_x`` : input-space neighbor indices.
        * ``distances_y`` : target-space neighbor distances.
        * ``D_prime`` : extended dataset (if requested).
    """

    N = len(features_x)

    if target_uncertainty is not None:
        sigma = np.asarray(
            target_uncertainty(features_y.ravel())
        )

    # --------------------------------------------------
    # kNN search in input features
  
    n_neighbors = min(N_max + 1, N)

    nbrs_x = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric = metricx,
        p =px
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
        algorithm="auto",
        metric = metricy,
        p=py
    )
    nbrs_y.fit(features_y)

    distances_y, indices_y = nbrs_y.kneighbors(features_y)

    # --------------------------------------------------
    # Global thresholds target data

    if target_uncertainty is None and tau_y == 0.0:
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

            if target_uncertainty is None:

                threshold = tau_y

            if dist <= threshold:

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
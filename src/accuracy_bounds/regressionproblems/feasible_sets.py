import numpy as np
from sklearn.neighbors import NearestNeighbors
from pseudometricfeatures import wavelet_features_chw

def feasible_sets_wavelet(
    images,
    y,
    N_max=50,
    qx=0.10,
    tau_y=None,
    qy=0.10,
    wavelet="haar",
):
    """
    Construct feasible sets for a collection of images using wavelet features.

    Wavelet features are extracted from each image and a k-nearest-neighbor
    search is performed in the resulting feature space. For each sample k, the
    feasible set consists of neighboring samples j satisfying

        d_X(x_k, x_j) <= tau_x    or    d_y(y_k, y_j) <= tau_y,

    where tau_x is estimated from the distribution of nearest-neighbor
    distances in feature space, and tau_y is either supplied by the user or
    estimated from the corresponding target differences.

    Args:
        images:
            Array of shape ``(N, C, H, W)`` containing the input images.
        y:
            One-dimensional array of target values of length ``N``.
        N_max:
            Maximum number of nearest neighbors considered for each sample.
        qx:
            Quantile used to estimate the feature-space threshold ``tau_x``.
        tau_y:
            Target-space threshold. If ``None``, it is estimated from the
            ``qy``-quantile of the target differences between each sample and
            its nearest neighbor in feature space.
        qy:
            Quantile used when estimating ``tau_y``.
        wavelet:
            Wavelet used to compute the image features.

    Returns:
        dict containing:
            - ``tau_x``: Estimated feature-space threshold.
            - ``tau_y``: Estimated or user-specified target-space threshold.
            - ``feasible_sets``: List whose k-th entry contains the indices of
              the feasible set associated with sample k.
            - ``distances``: k-nearest-neighbor distances in feature space.
            - ``indices``: Indices corresponding to ``distances``.
    """
    N = len(images)
    y = np.asarray(y, dtype=np.float32)

    # --------------------------------------------------
    # Compute wavelet features

    feats = np.stack([
        wavelet_features_chw(images[i], wavelet=wavelet)
        for i in range(N)
    ])

    # --------------------------------------------------
    # kNN search
  
    n_neighbors = min(N_max + 1, N)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )
    nbrs.fit(feats)

    distances, indices = nbrs.kneighbors(feats)

    # --------------------------------------------------
    # Global thresholds

    nearest_dist = distances[:, 1]
    tau_x = float(np.quantile(nearest_dist, qx))

    if tau_y is None:
        y_nn = y[indices[:, 1]]
        tau_y = float(
            np.quantile(np.abs(y_nn - y), qy)
        )

    # --------------------------------------------------
    # Compute feasible sets
   
    feasible_sets = []

    for k in range(N):

        cand_idx = indices[k, 1:]
        cand_dx = distances[k, 1:]
        cand_dy = np.abs(y[cand_idx] - y[k])

        mask = (cand_dx <= tau_x) | (cand_dy <= tau_y)

        feasible_sets.append(cand_idx[mask])


    return {
    "tau_x": tau_x,
    "tau_y": tau_y,
    "feasible_sets": feasible_sets,
    "distances": distances,
    "indices": indices,
}
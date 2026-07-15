import numpy as np
from sklearn.neighbors import NearestNeighbors


def feasible_sets_classification(
    features,
    labels,
    N_max=50,
    qx=0.10,
):
    """
    Construct feasible label sets for a classification problem using
    nearest-neighbor distances in feature space.

    For each sample k, a feasible set is constructed by selecting neighboring
    samples whose feature distance satisfies

        d_X(x_k, x_j) <= tau_x.

    The feasible set contains the labels of these neighboring samples. The
    threshold tau_x is estimated from the qx-quantile of nearest-neighbor
    distances in feature space.

    Args:
        features:
            Array of shape ``(N, D)`` containing input features.
        labels:
            Array of length ``N`` containing class labels.
        N_max:
            Maximum number of nearest neighbors considered for each sample.
        qx:
            Quantile used to estimate the feature-space threshold ``tau_x``.

    Returns:
        dict containing:
            - ``tau_x``:
                Estimated feature-space threshold.
            - ``feasible_sets``:
                List where ``feasible_sets[k]`` contains the class labels
                appearing in the feasible neighborhood of sample k.
            - ``distances``:
                Matrix of nearest-neighbor distances.
            - ``indices``:
                Matrix of nearest-neighbor indices.
            - ``D_prime``:
                List of tuples ``(sample_index, class_label)`` representing
                feasible sample-label pairs.
    """

    N = len(features)
    labels = np.asarray(labels)

    # --------------------------------------------------
    # kNN search
    # --------------------------------------------------
    n_neighbors = min(N_max + 1, N)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )

    nbrs.fit(features)

    distances, indices = nbrs.kneighbors(features)

    # --------------------------------------------------
    # Estimate tau_x
    # --------------------------------------------------
    nearest_dist = distances[:, 1]
    tau_x = float(np.quantile(nearest_dist, qx))

    # --------------------------------------------------
    # Compute feasible sets
    # --------------------------------------------------
    feasible_sets = []
    D_prime = []

    for k in range(N):

        feasible = []

        for idx, dist in zip(indices[k, 1:], distances[k, 1:]):

            if dist <= tau_x:

                feasible.append(labels[idx])
                D_prime.append((k, labels[idx]))

        feasible_sets.append(np.array(feasible))


    return {
        "tau_x": tau_x,
        "feasible_sets": feasible_sets,
        "D_prime": D_prime,
        "distances": distances,
        "indices": indices,
    }
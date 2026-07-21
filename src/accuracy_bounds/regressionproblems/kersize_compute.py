import numpy as np
from scipy.spatial.distance import cdist

def compute_kernel_size(feasible_sets, metric="l2", p=2.0):
    """
    Compute the average kernel size from precomputed feasible sets.

    Parameters
    ----------
    feasible_sets : list[np.ndarray]
        feasible_sets[k] contains the feasible set for the target sample k.
    metric : str
        Metric used to compute the kernel size in the target space.
    p : float
        Power in the kernel size definition.

    Returns
    -------
    kersize : float
    M_prime : int
    Nk_stats : tuple
        (min, mean, median, max)
    """

    M_prime = 0
    sum_term = 0.0
    Nk_list = []

    for feas in feasible_sets:

        Nk = len(feas)
        Nk_list.append(Nk)
        M_prime += Nk

        if Nk < 2:
            continue


        if metric == "l1":
            diffs = cdist(feas, feas, metric="cityblock") ** p
        elif metric == "l2":
            diffs = cdist(feas, feas, metric="euclidean") ** p

        pair_sum = np.sum(np.triu(diffs, k=1))

        sum_term += (2.0 / (Nk - 1)) * pair_sum

    kersize_p = 0.0 if M_prime == 0 else sum_term / M_prime
    kersize = (
        float(kersize_p ** (1.0 / p))
        if p != 0
        else float(kersize_p)
    )

    Nk_arr = np.array(Nk_list) if Nk_list else np.array([0])

    Nk_stats = (
        int(Nk_arr.min()),
        float(Nk_arr.mean()),
        float(np.median(Nk_arr)),
        int(Nk_arr.max()),
    )

    return kersize, M_prime, Nk_stats
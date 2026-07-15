import numpy as np

def compute_kernel_size_one_hot(feasible_sets):
    """
    Compute the average kernel size from precomputed feasible sets for one-hot encoded targets.

    Parameters
    ----------
    feasible_sets : list[np.ndarray]
        feasible_sets[k] contains the feasible set for the target sample k.

    Returns
    -------
    kersize : float
    M_prime : int
    Nk_stats : tuple
        (min, mean, median, max)
    """

    M_prime = 0
    average_kersize = 0
    Nk_arr = np.array([len(labels) for labels in feasible_sets])

    for labels in feasible_sets:

        Nk = len(labels)
        M_prime += Nk

        if Nk < 2:
            continue

        counts = np.bincount(labels, minlength=10)

        same_pairs = (counts * (counts - 1) // 2).sum()

        total_pairs = Nk * (Nk - 1) // 2

        diff_pairs = total_pairs - same_pairs

        inner = 2 * diff_pairs / (Nk - 1)

        average_kersize += inner


    if M_prime == 0:
        average_kersize = 0
    else:   
        average_kersize = average_kersize / M_prime

    Nk_stats = (
        int(Nk_arr.min()),
        float(Nk_arr.mean()),
        float(np.median(Nk_arr)),
        int(Nk_arr.max()),
    )

    return average_kersize, M_prime, Nk_stats
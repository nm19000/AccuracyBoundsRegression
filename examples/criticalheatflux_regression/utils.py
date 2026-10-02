import numpy as np


def chf_uncertainty(q):
    """
    Estimate measurement uncertainty for critical heat flux.

    Parameters
    ----------
    q : array-like
        Critical heat flux / HeatFlux in W/m^2.

    Returns
    -------
    sigma : ndarray
        Estimated 1-sigma uncertainty in W/m^2.

    Model
    -----
        sigma(q) = sqrt(sigma_floor^2 + (relative_error * q)^2)

    This represents:
        - an approximately constant absolute component at low CHF
        - a fractional component that grows with heat flux

    NOTE:
        This is a generic heteroskedastic measurement model.
        The coefficients should be calibrated against the actual
        KAERI experimental uncertainty information if available.
    """

    q = np.asarray(q, dtype=float)

    relative_error = 0.1      # 10%
    sigma_floor = 1_000.0       # W/m^2

    return np.sqrt(
        sigma_floor**2
        + (relative_error * np.abs(q))**2
    )

def compute_R_squared(feasible_sets):
    """ Compute the R-squared value for a collection of feasible sets. Which is the optimal RMSE, only on D_prime and for the l2 norm."""
    return sum(
        np.sum((Y - Y.mean(axis=0))**2)
        for Y in feasible_sets if len(Y)
    ) / sum(len(Y) for Y in feasible_sets)
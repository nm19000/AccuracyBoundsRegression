import numpy as np

def tc_uncertainty(v):
    """
    Estimate wind-speed measurement uncertainty (knots)
    as a function of tropical cyclone intensity.

    Input:
        v : array-like wind speeds in knots

    Returns:
        array of uncertainty values in knots
    """

    
    v = np.asarray(v)

    return np.select(
        [
            v < 34,
            v < 64,
            v < 83,
            v < 96,
            v < 113,
        ],
        [
            7.0,    # tropical depression
            8.0,    # tropical storm
            10.0,   # category 1
            12.0,   # category 2
            15.0,   # category 3
        ],
        default=18.0  # category 4-5
    )

def tc_uncertainty2(v):
    """
    CYGNSS-inspired wind-speed uncertainty model.

    Based on Ruf et al. (2020):
    low wind speeds have approximately constant uncertainty,
    while high wind speeds have fractional uncertainty.
    """

    v = np.asarray(v)

    return np.where(
        v < 39.0,       # 20 m/s converted to knots
        2.7,            # 1.4 m/s in knots
        0.17 * v
    )
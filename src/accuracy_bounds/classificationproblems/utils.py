import numpy as np
from .pseudometricfeatures import wavelet_features_chw

def feasible_sets_features(
    images,
    wavelet="haar",
):
    """
Compute wavelet feature vectors for a collection of images.

Each image is transformed into a one-dimensional feature vector using
``wavelet_features_chw``. The resulting feature matrix can be used for
subsequent tasks such as nearest-neighbor searches or feasible set
construction.

Args:
    images:
        Array of shape ``(N, C, H, W)`` containing the input images.
    y:
        One-dimensional array of target values. This argument is included for
        interface consistency with related functions but is not used in the
        computation.
    wavelet:
        Name of the wavelet used to compute the image features.

Returns:
    np.ndarray:
        Array of shape ``(N, d)``, where each row is the wavelet feature vector
        corresponding to one input image.
"""

    N = len(images)
    y = np.asarray(y, dtype=np.float32)

    # --------------------------------------------------
    # Compute wavelet features

    feats = np.stack([
        wavelet_features_chw(images[i], wavelet=wavelet)
        for i in range(N)
    ])

    return feats

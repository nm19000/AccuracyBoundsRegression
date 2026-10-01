import pywt
import numpy as np

def wavelet_features_chw(img_chw, wavelet="haar"):
    """
    img_chw: np.ndarray (C,H,W)
    returns: 1D np.ndarray features
    """
    C, H, W = img_chw.shape
    feats = []
    for c in range(C):
        LL, (LH, HL, HH) = pywt.dwt2(img_chw[c], wavelet, mode="symmetric")
        feats.append(LL.ravel()); feats.append(LH.ravel())
        feats.append(HL.ravel()); feats.append(HH.ravel())
    return np.concatenate(feats).astype(np.float32)


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

    # --------------------------------------------------
    # Compute wavelet features

    feats = np.stack([
        wavelet_features_chw(images[i], wavelet=wavelet)
        for i in range(N)
    ])

    return feats

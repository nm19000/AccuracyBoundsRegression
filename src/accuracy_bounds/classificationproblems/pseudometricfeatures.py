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
        LL, (LH, HL, HH) = pywt.dwt2(img_chw[c], wavelet)
        feats.append(LL.ravel()); feats.append(LH.ravel())
        feats.append(HL.ravel()); feats.append(HH.ravel())
    return np.concatenate(feats).astype(np.float32)

import numpy as np

def resample_exposure(expo_raw, E_expo, E_cnt):
    """
    Interpolate exposure(E) onto counts bin centers using log(E).
    expo_raw: (Ne, ny, nx)
    E_expo, E_cnt in GeV
    """
    if expo_raw.shape[0] == len(E_cnt):
        return expo_raw

    order = np.argsort(E_expo)
    expo_raw = expo_raw[order]
    E_expo = E_expo[order]

    logE_src = np.log(E_expo)
    logE_tgt = np.log(E_cnt)
    print(logE_src)
    print(logE_tgt)

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    out = np.empty((len(E_cnt), ny * nx))
    for i, le in enumerate(logE_tgt):
        j = np.searchsorted(logE_src, le)
        j = np.clip(j, 1, ne - 1)
        w = (le - logE_src[j-1]) / (logE_src[j] - logE_src[j-1])
        out[i] = (1 - w) * flat[j-1] + w * flat[j]

    return out.reshape(len(E_cnt), ny, nx)


# Lightweight embeddings for MRI volumes without heavy DL deps.
# Strategy: downsample + intensity histogram + simple spatial stats.
import numpy as np
import nibabel as nib

def _volume_to_embedding(vol: np.ndarray, bins: int = 64) -> np.ndarray:
    # Basic intensity histogram
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        return np.zeros(bins + 6, dtype=np.float32)
    v = np.clip(v, np.percentile(v, 1), np.percentile(v, 99))
    hist, edges = np.histogram(v, bins=bins, density=True)
    # simple spatial stats on central slices
    cx0, cx1 = vol.shape[0]//4, 3*vol.shape[0]//4
    cy0, cy1 = vol.shape[1]//4, 3*vol.shape[1]//4
    cz0, cz1 = vol.shape[2]//4, 3*vol.shape[2]//4
    center = vol[cx0:cx1, cy0:cy1, cz0:cz1]
    stats = np.array([v.mean(), v.std(), v.min(), v.max(),
                      center.mean() if center.size else 0.0,
                      center.std() if center.size else 0.0], dtype=np.float32)
    return np.concatenate([hist.astype(np.float32), stats]).astype(np.float32)

def embed_volume(nifti_path: str) -> np.ndarray:
    # Return a 1D embedding for a NIfTI volume.
    img = nib.load(nifti_path)
    vol = img.get_fdata()
    # fast downsample
    vol = vol[::2, ::2, ::2] if min(vol.shape) >= 32 else vol
    return _volume_to_embedding(vol)

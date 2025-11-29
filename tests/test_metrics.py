import math
import numpy as np
import nibabel as nib
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec_cid.coverage_metrics import k_center_cover_count, diversity_score
from cec_cid.clb import fit_power_loglaw
from cec_cid.embedder import embed_volume


def test_k_center_cover_count_simple():
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    assert k_center_cover_count(pts, eps=0.25) == 3  # tight radius needs all points
    assert k_center_cover_count(pts, eps=1.5) == 2   # middle point covered by others
    assert k_center_cover_count(pts, eps=3.0) == 1   # one ball covers all


def test_diversity_score_mean_distance():
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    mean = diversity_score(pts)
    assert math.isclose(mean, (3.0 + 5.0 + 4.0) / 3.0, rel_tol=1e-6)


def test_fit_power_loglaw_recovers_parameters():
    eps_list = [0.1, 0.05, 0.025, 0.0125]
    d_true, beta_true, c_true = 2.0, 0.25, -0.3
    logN = []
    for eps in eps_list:
        x1 = math.log(1 / eps)
        x2 = math.log(math.log(1 / eps) + 1e-9)
        logN.append(d_true * x1 + beta_true * x2 + c_true)
    d, beta, c = fit_power_loglaw(eps_list, logN)
    assert math.isclose(d, d_true, rel_tol=1e-3)
    assert math.isclose(beta, beta_true, rel_tol=1e-3)
    assert math.isclose(c, c_true, rel_tol=1e-3)


def test_embed_volume_smoke(tmp_path):
    data = np.ones((8, 8, 8), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmp_path / "dummy.nii.gz"
    nib.save(img, path)
    emb = embed_volume(str(path))
    assert emb.ndim == 1
    assert emb.size > 10
    assert np.isfinite(emb).all()

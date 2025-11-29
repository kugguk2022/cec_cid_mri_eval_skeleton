import os
import sys
import numpy as np
import nibabel as nib
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import process_brats as pb


def make_dummy_nifti(shape=(8, 8, 8), fill=1.0):
    data = np.full(shape, fill, dtype=np.float32)
    return nib.Nifti1Image(data, affine=np.eye(4))


def test_extract_slice_handles_3d_axis(tmp_path):
    img = make_dummy_nifti()
    path = tmp_path / "vol.nii.gz"
    nib.save(img, path)

    # Axis 2 (default)
    pb.SLICE_AXIS = 2
    proxy = nib.load(path).dataobj
    s = pb.extract_slice(proxy, 0)
    assert s.shape == (8, 8)
    assert np.allclose(s, 1.0)

    # Axis 0
    pb.SLICE_AXIS = 0
    s0 = pb.extract_slice(proxy, 1)
    assert s0.shape == (8, 8)
    assert np.allclose(s0, 1.0)


def test_brain_slice_dataset_normalizes_and_resizes(tmp_path):
    pb.SLICE_AXIS = 2
    img = make_dummy_nifti(shape=(6, 10, 14), fill=5.0)
    path = tmp_path / "vol.nii.gz"
    nib.save(img, path)

    ds = pb.BrainSliceDataset([(str(path), 2)])
    tensor, pth, idx = ds[0]
    assert tensor.shape == (1, 224, 224)
    assert pth.endswith("vol.nii.gz")
    assert idx == 2
    assert torch.isfinite(tensor).all()
    assert torch.all((0.0 <= tensor) & (tensor <= 1.0))


def test_end_to_end_embedding_smoke(tmp_path):
    pb.SLICE_AXIS = 2
    img = make_dummy_nifti(shape=(8, 8, 8), fill=2.0)
    path = tmp_path / "vol.nii.gz"
    nib.save(img, path)

    ds = pb.BrainSliceDataset([(str(path), 3)])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    model = pb.get_feature_extractor()

    with torch.no_grad():
        for batch, *_ in loader:
            feats = model(batch)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    assert feats.shape[1] == 512
    assert torch.isfinite(feats).all()
    assert torch.allclose(torch.linalg.norm(feats, dim=1), torch.ones(feats.size(0)), atol=1e-4)


def test_scan_dataset_matches_nifti_extensions(tmp_path):
    pb.SLICE_AXIS = 2
    bids = tmp_path / "BIDS_BRATS"
    anat = bids / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)

    img = make_dummy_nifti(shape=(4, 4, 4), fill=3.0)
    nib.save(img, anat / "sub-01_ses-01_T1w.nii")
    nib.save(img, anat / "sub-01_ses-01_T2.nii.gz")

    slices_t1 = pb.scan_dataset(str(bids), target_modality="T1w", slice_axis=2, stride=1)
    slices_t2 = pb.scan_dataset(str(bids), target_modality="T2", slice_axis=2, stride=1)

    assert len(slices_t1) == 4
    assert len(slices_t2) == 4

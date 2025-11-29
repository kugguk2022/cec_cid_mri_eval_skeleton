# CEC–CID MRI Evaluation Skeleton

Serious, non-toy scaffold to compare two generative models (G1 vs G2) on brain MRI using a **Coverage–Efficiency** view:
- Fit the **CEC signature** `(d, beta, c)` of real data via approximate covering numbers in an embedding space.
- Compute the **Capacity Lower Bound (CLB)**: `B >= log N(A, eps)`.
- Log each model's **budget B**, **quality eps**, and **achieved coverage N(A, eps; G, j)**.
- Plot `B` vs `log(1/eps)` against CLB to see who hugs the lower bound.

> Note: Auto-downloading many medical datasets requires registration/DUAs. This repo standardizes **ingest → BIDS** and **embedding/coverage** so you can plug any dataset you have legal access to.

## Layout
```
cec_cid_mri_eval_skeleton/
├─ datasets/
│  ├─ INGEST.md                     # Where to put raw data & how to ingest
│  ├─ ingest_templates/             # Templated download/ingest scripts (edit with your creds/paths)
│  │  ├─ ingest_brats_template.sh
│  │  ├─ ingest_ixi_template.py
│  │  ├─ ingest_isles2022_template.py
│  │  ├─ ingest_msd_task01_template.sh
│  │  ├─ ingest_fastmri_template.py
│  │  └─ ingest_meningioma_template.py
│  └─ raw/                          # Place raw archives or DICOM/NIfTI here (per-dataset subfolders)
├─ utils/
│  └─ bids_repack/
│     ├─ mri_to_bids.py             # Minimal BIDS repacker (NIfTI + JSON → BIDS)
│     └─ rules.json                 # Simple mapping rules
├─ cec_cid/
│  ├─ embedder.py                   # Lightweight embeddings (no heavy deps) + hooks for custom encoders
│  ├─ coverage_metrics.py           # k-center covering number, diversity, mode-collapse probes
│  ├─ clb.py                        # CEC fit & CLB line estimation
│  ├─ schedules.py                  # Budget schedules (log-boost, plateau, bursts)
│  └─ logging_harness.py            # Unified logger → CSVs (budget_trace.csv, coverings.csv, clb.csv)
├─ train/
│  ├─ evaluate_generators.py        # Compare G1 vs G2 against CLB (expects per-checkpoint dumps/embeddings)
│  ├─ g1_diffusion_stub.py          # Budget tracker hooks for diffusion-style training loops
│  └─ g2_gan_stub.py                # Budget tracker hooks for GAN-style loops
├─ outputs/
│  ├─ example/budget_trace.csv
│  ├─ example/coverings.csv
│  └─ example/clb.csv
└─ Makefile
```

## Quickstart
1) Put your data under `datasets/raw/<DATASET>/...` (NIfTI or DICOM).  
2) Run BIDS repack:
```
python utils/bids_repack/mri_to_bids.py   --in_dir datasets/raw/IXI --out_dir datasets/BIDS_IXI --dataset_name IXI
```
3) Build embeddings & CEC/CLB from a **healthy** corpus (e.g., IXI/HCP subset):
```
python cec_cid/clb.py --bids_dir datasets/BIDS_IXI --out_dir outputs/ixi_clb --eps_list 0.05 0.1 0.2
```
4) Hook your training loops (`train/g1_diffusion_stub.py`, `train/g2_gan_stub.py`) to log:
   - `budget_trace.csv`: step, model, B, eps, N, notes
   - `coverings.csv`: eps, logN, etc.
5) Compare:
```
python train/evaluate_generators.py   --clb_csv outputs/ixi_clb/clb.csv   --budget_csv outputs/example/budget_trace.csv
```

## Dependencies
- Python 3.9+
- `numpy`, `nibabel` (for NIfTI), optionally `pydicom` if you ingest DICOM
- No heavy DL libs required for core metrics; you can later plug MONAI/torch encoders.

**You control the encoder:** `cec_cid/embedder.py` exposes a single function `embed_volume()` you can replace with your radiology backbone.

##Citation
If you use this work in your research, please cite:
```
@software{
CEC-CID 2025, title={TheCEC·CIDFramework for Adaptive Generators and Unbounded Sets},
author={kugguk2022},
year={2025},
url={https://github.com/kugguk2022/cec_cid_mri_eval_skeleton}
}
```

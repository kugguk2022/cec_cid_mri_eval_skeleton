# Ingest notes (edit per dataset)

Place raw data under:
```
datasets/raw/
  BRATS/      # Place BraTS ZIPs or extracted NIfTI
  IXI/        # Place IXI NIfTI (T1/T2/PD)
  ISLES2022/  # Place lesion challenge MRIs
  MSD_TASK01/ # Decathlon Task01 (BrainTumour)
  FASTMRI/    # fastMRI brain (raw k-space or recon)
  MENINGIOMA/ # Meningioma cohort (MRIs + masks)
```
Then run BIDS repack (example for IXI):
```
python utils/bids_repack/mri_to_bids.py --in_dir datasets/raw/IXI --out_dir datasets/BIDS_IXI --dataset_name IXI
```

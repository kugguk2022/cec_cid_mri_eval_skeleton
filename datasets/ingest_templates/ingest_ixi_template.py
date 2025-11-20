
#!/usr/bin/env python3
import os, shutil, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='datasets/raw/IXI')
parser.add_argument('--dst', default='datasets/BIDS_IXI')
args = parser.parse_args()

os.makedirs(args.dst, exist_ok=True)
for nifti in glob.glob(os.path.join(args.src, '**', '*.nii*'), recursive=True):
    rel = os.path.basename(nifti)
    shutil.copy2(nifti, os.path.join(args.dst, rel))
print("Copied; now run mri_to_bids.py to normalize structure.")

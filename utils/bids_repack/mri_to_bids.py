
#!/usr/bin/env python3
import os, re, argparse, json, shutil, glob, hashlib, sys
from datetime import datetime

def safe(name):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name)

def guess_modality(fn, rules):
    low = fn.lower()
    for mod, keys in rules["modality_map"].items():
        for k in keys:
            if k.lower() in low:
                return mod
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="raw dataset root (NIfTI or DICOM already converted)")
    ap.add_argument("--out_dir", required=True, help="BIDS output root")
    ap.add_argument("--dataset_name", required=True, help="e.g., IXI, BRATS, ISLES2022")
    ap.add_argument("--subject_prefix", default="sub")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--rules", default=os.path.join(os.path.dirname(__file__), "rules.json"))
    args = ap.parse_args()

    with open(args.rules, "r") as f:
        rules = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    ds_json = {
        "Name": args.dataset_name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name":"mri_to_bids.py","Version":"minimal","CodeURL":""}],
        "Date": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(args.out_dir, "dataset_description.json"), "w") as f:
        json.dump(ds_json, f, indent=2)

    files = glob.glob(os.path.join(args.in_dir, "**", "*.nii*"), recursive=True)
    if not files:
        print("No NIfTI files found under", args.in_dir, file=sys.stderr)
        sys.exit(1)

    for path in files:
        parent = os.path.basename(os.path.dirname(path)) or "unk"
        h = hashlib.sha1(parent.encode()).hexdigest()[:8]
        sub = f"{args.subject_prefix}-{h}"
        modality = guess_modality(os.path.basename(path), rules)
        if modality == "unknown":
            continue
        out_dir = os.path.join(args.out_dir, sub, args.session, "anat")
        os.makedirs(out_dir, exist_ok=True)
        base = f"{sub}_{args.session}_T1w.nii.gz" if modality=="t1" else f"{sub}_{args.session}_{modality.upper()}.nii.gz"
        dst = os.path.join(out_dir, base)
        shutil.copy2(path, dst)

    print("BIDS repack complete →", args.out_dir)

if __name__ == "__main__":
    main()

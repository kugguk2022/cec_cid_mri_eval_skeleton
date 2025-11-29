#!/usr/bin/env python3
import os, re, argparse, json, shutil, glob, hashlib, sys
from datetime import datetime
from collections import Counter

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
    ap.add_argument("--verbose", action="store_true", help="print per-file decisions and summary counts")
    ap.add_argument("--fail_on_unknown", action="store_true", help="raise if a file modality cannot be inferred")
    args = ap.parse_args()

    with open(args.rules, "r") as f:
        rules = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    ds_json = {
        "Name": args.dataset_name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "mri_to_bids.py", "Version": "minimal", "CodeURL": ""}],
        "Date": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "dataset_description.json"), "w") as f:
        json.dump(ds_json, f, indent=2)

    files = glob.glob(os.path.join(args.in_dir, "**", "*.nii*"), recursive=True)
    if not files:
        print("No NIfTI files found under", args.in_dir, file=sys.stderr)
        sys.exit(1)

    stats = Counter()
    modality_counts = Counter()
    for path in files:
        parent = os.path.basename(os.path.dirname(path)) or "unk"
        h = hashlib.sha1(parent.encode()).hexdigest()[:8]
        sub = f"{args.subject_prefix}-{h}"
        modality = guess_modality(os.path.basename(path), rules)
        if modality == "unknown":
            stats["unknown"] += 1
            msg = f"[skip] {path} (modality unknown)"
            if args.fail_on_unknown:
                raise RuntimeError(msg)
            if args.verbose:
                print(msg, file=sys.stderr)
            continue
        modality_counts[modality] += 1
        out_dir = os.path.join(args.out_dir, sub, args.session, "anat")
        os.makedirs(out_dir, exist_ok=True)
        ext = ".nii.gz" if path.lower().endswith(".nii.gz") else ".nii"
        base = f"{sub}_{args.session}_T1w{ext}" if modality == "t1" else f"{sub}_{args.session}_{modality.upper()}{ext}"
        dst = os.path.join(out_dir, base)
        try:
            shutil.copy2(path, dst)
            stats["copied"] += 1
            if args.verbose:
                print(f"[copy] {path} -> {dst}", file=sys.stderr)
        except Exception as exc:
            stats["errors"] += 1
            print(f"[error] {path}: {exc}", file=sys.stderr)

    if args.verbose:
        print("--- summary ---", file=sys.stderr)
        print(f"copied: {stats['copied']}", file=sys.stderr)
        print(f"unknown/skipped: {stats['unknown']}", file=sys.stderr)
        print(f"errors: {stats['errors']}", file=sys.stderr)
        for mod, cnt in sorted(modality_counts.items()):
            print(f"  {mod}: {cnt}", file=sys.stderr)

    print("BIDS repack complete ->", args.out_dir)

if __name__ == "__main__":
    main()

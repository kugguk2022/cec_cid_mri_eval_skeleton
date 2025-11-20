
# Fit CEC signature (d, beta, c) and export CLB as B >= log N(A, eps)
import os, argparse, csv, numpy as np, glob
from .embedder import embed_volume
from .coverage_metrics import k_center_cover_count

def fit_power_loglaw(eps_list, logN_list):
    # Fit: logN ~ d * log(1/eps) + beta*log log(1/eps) + c
    x1 = np.log(1/np.array(eps_list))
    x2 = np.log(np.log(1/np.array(eps_list) + 1e-9) + 1e-9)
    X = np.stack([x1, x2, np.ones_like(x1)], axis=1)
    y = np.array(logN_list)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    d, beta, c = theta.tolist()
    return float(d), float(beta), float(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eps_list", nargs="+", type=float, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = glob.glob(os.path.join(args.bids_dir, "sub-*", "ses-*", "anat", "*.nii*"))
    embs = []
    for p in paths:
        try:
            embs.append(embed_volume(p))
        except Exception:
            continue
    E = np.stack(embs, axis=0) if embs else np.zeros((0, 70), dtype=np.float32)

    rows = []
    for eps in args.eps_list:
        N = k_center_cover_count(E, eps) if len(E) else 0
        rows.append({"eps": eps, "logN": float(np.log(N + 1e-9))})

    cov_path = os.path.join(args.out_dir, "coverings.csv")
    with open(cov_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps","logN"])
        w.writeheader()
        w.writerows(rows)

    d, beta, c = fit_power_loglaw([r["eps"] for r in rows], [r["logN"] for r in rows]) if rows else (0.0,0.0,0.0)
    clb_rows = []
    lo, hi = min(args.eps_list), max(args.eps_list)
    for eps in np.linspace(lo, hi, 20):
        x1 = np.log(1/eps)
        x2 = np.log(np.log(1/eps) + 1e-9)
        logN = d*x1 + beta*x2 + c
        clb_rows.append({"eps": float(eps), "logN": float(logN), "d": d, "beta": beta, "c": c})

    with open(os.path.join(args.out_dir, "clb.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps","logN","d","beta","c"])
        w.writeheader(); w.writerows(clb_rows)

    print(f"Saved: {cov_path} and clb.csv; fitted d={d:.3f}, beta={beta:.3f}, c={c:.3f}")

if __name__ == "__main__":
    main()

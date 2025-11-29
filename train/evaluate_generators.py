# Compare budget traces against CLB line
import csv, argparse, sys


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def require_columns(rows, cols, name):
    if not rows:
        raise ValueError(f"{name} is empty or missing rows")
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clb_csv", required=True)
    ap.add_argument("--budget_csv", required=True)
    ap.add_argument("--out_plot", help="optional path to save eps-vs-B plot")
    args = ap.parse_args()

    clb = load_csv(args.clb_csv)
    trace = load_csv(args.budget_csv)
    require_columns(clb, ["eps", "logN"], "clb_csv")
    require_columns(trace, ["model", "eps", "B"], "budget_csv")

    clb_sorted = sorted(clb, key=lambda r: float(r["eps"]))
    gaps = []
    for rec in trace:
        eps = float(rec["eps"])
        B = float(rec["B"])
        near = min(clb_sorted, key=lambda r: abs(float(r["eps"]) - eps))
        clb_B = float(near["logN"])
        gap = B - clb_B
        gaps.append({"model": rec["model"], "eps": eps, "B": B, "clb": clb_B, "gap": gap})

    for rec in gaps[:10]:
        print(f"model={rec['model']:<12} eps={rec['eps']:.4f} B={rec['B']:.3f}  CLB={rec['clb']:.3f}  gap={rec['gap']:+.3f}")

    if args.out_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plotting skipped (matplotlib unavailable): {exc}", file=sys.stderr)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            eps_vals = [float(r["eps"]) for r in clb_sorted]
            ax.plot(eps_vals, [float(r["logN"]) for r in clb_sorted], label="CLB", color="black")
            models = {}
            for rec in gaps:
                models.setdefault(rec["model"], {"eps": [], "B": []})
                models[rec["model"]]["eps"].append(rec["eps"])
                models[rec["model"]]["B"].append(rec["B"])
            for m, pts in models.items():
                ax.scatter(pts["eps"], pts["B"], s=18, label=m)
            ax.set_xlabel("eps")
            ax.set_ylabel("B / logN")
            ax.set_xscale("log")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, ls="--", alpha=0.4)
            ax.set_title("Budget vs CEC Lower Bound")
            fig.tight_layout()
            fig.savefig(args.out_plot, dpi=200)
            print(f"Plot saved to {args.out_plot}")


if __name__ == "__main__":
    main()

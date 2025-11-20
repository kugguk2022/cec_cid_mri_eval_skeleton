
# Compare budget traces against CLB line
import csv, argparse

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clb_csv", required=True)
    ap.add_argument("--budget_csv", required=True)
    args = ap.parse_args()

    clb = load_csv(args.clb_csv)
    trace = load_csv(args.budget_csv)

    clb_sorted = sorted(clb, key=lambda r: float(r["eps"]))
    for rec in trace[:10]:
        eps = float(rec["eps"])
        B = float(rec["B"])
        near = min(clb_sorted, key=lambda r: abs(float(r["eps"]) - eps))
        clb_B = float(near["logN"])
        gap = B - clb_B
        print(f"model={rec['model']:<8} eps={eps:.3f} B={B:.3f}  CLB={clb_B:.3f}  gap={gap:+.3f}")

if __name__ == "__main__":
    main()

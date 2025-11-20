
# Unified CSV logger for (B, eps, N) per checkpoint
import os, csv
from dataclasses import dataclass, asdict

@dataclass
class BudgetRecord:
    step: int
    model: str
    B: float
    eps: float
    N: float
    notes: str = ""

class ExperimentLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.budget_path = os.path.join(out_dir, "budget_trace.csv")
        if not os.path.exists(self.budget_path):
            with open(self.budget_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step","model","B","eps","N","notes"])
                w.writeheader()

    def log(self, rec: BudgetRecord):
        with open(self.budget_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step","model","B","eps","N","notes"])
            w.writerow(asdict(rec))

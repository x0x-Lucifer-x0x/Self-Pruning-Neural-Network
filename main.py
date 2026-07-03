"""
main.py
-------
Entry point. Sets random seed for reproducibility, defines all config,
then runs training for each lambda value and saves results.

Usage:
    python main.py

Outputs (saved to ./results/):
    gate_distributions.png  — gate histogram per lambda
    training_curves.png     — accuracy + sparsity over epochs
    report.md               — markdown report (spec requirement)
"""

import torch
import numpy as np
import random
import os

from train import (
    get_dataloaders,
    train_model,
    plot_gate_distribution,
    plot_training_curves,
    print_results_table,
    save_report,
)


# ─── Reproducibility seed ──────────────────────────────────────────────────────
SEED = 42

def set_seed(seed):
    """
    Fix all random sources so results are reproducible across runs.
    Covers Python, NumPy, and PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU if ever used
    # makes cuDNN deterministic — slight speed cost, full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed set to {seed} — results are reproducible.")


# ─── Config (no hardcoding in train.py or model.py) ───────────────────────────
CFG = {
    "seed":          SEED,
    "epochs":        60,
    "batch_size":    128,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "warmup_epochs": 0,       # epochs before lambda starts ramping
    "prune_thresh":  1e-2,     # gate below this = pruned
    "num_workers":   2,
    "data_dir":      "./data",
    "results_dir":   "./results",
    "device":        torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # three lambda values: low / medium / high  (spec requires minimum 3)
    "lambdas":       [0.1, 1.0, 5.0],
}


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(CFG["seed"])

    os.makedirs(CFG["results_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print("  Self-Pruning Neural Network — CIFAR-10")
    print("  Tredence AI Engineering Case Study")
    print("=" * 60)
    print(f"  Seed       : {CFG['seed']}")
    print(f"  Epochs     : {CFG['epochs']}  |  Batch: {CFG['batch_size']}  |  LR: {CFG['lr']}")
    print(f"  LR warmup  : {CFG['warmup_epochs']} epochs")
    print(f"  Lambdas    : {CFG['lambdas']}")
    print(f"  Device     : {CFG['device']}")
    if CFG["device"].type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")

    # load CIFAR-10 once, reuse across all lambda runs
    train_loader, test_loader = get_dataloaders(
        data_dir    = CFG["data_dir"],
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
    )

    all_results = []
    for lam in CFG["lambdas"]:
        # re-seed before each run so each lambda starts from identical init
        set_seed(CFG["seed"])
        res = train_model(lam, train_loader, test_loader, CFG)
        all_results.append(res)

    # ── outputs ───────────────────────────────────────────────────────────
    print_results_table(all_results)
    plot_gate_distribution(all_results, CFG["results_dir"], CFG["prune_thresh"])
    plot_training_curves(all_results,   CFG["results_dir"])
    save_report(all_results,            CFG["results_dir"])

    print(f"\nDone. All outputs saved to {CFG['results_dir']}/")


if __name__ == "__main__":
    main()

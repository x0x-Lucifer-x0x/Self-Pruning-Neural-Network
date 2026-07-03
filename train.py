"""
train.py
--------
Contains:
  - get_dataloaders  : CIFAR-10 loaders with augmentation
  - get_lambda_for_epoch : linear warmup schedule for lambda
  - train_one_epoch  : single epoch training with total loss
  - evaluate         : test accuracy
  - train_model      : full training run for one lambda value
  - plot_gate_distribution : gate histogram (bimodal = success)
  - plot_training_curves   : accuracy + sparsity over epochs
  - print_results_table    : console summary table
  - save_report            : markdown report as required by spec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from model import SelfPruningNet, PrunableLinear


# ─── Dataset ───────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size, num_workers=2):
    """
    CIFAR-10 train and test loaders.
    Train: random crop + horizontal flip + color jitter (standard augmentation)
    Test:  only normalize (no augmentation — fair evaluation)
    Official CIFAR-10 channel mean/std for normalization.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ─── Lambda warmup ─────────────────────────────────────────────────────────────
def get_lambda_for_epoch(target_lambda, epoch, warmup_epochs):
    """
    Linearly ramp lambda from 0 → target over warmup_epochs.

    Why warmup:
      Without it, high lambda immediately kills gates before the network
      has learned any useful features → poor accuracy from the start.
      Warmup lets the CNN backbone and gates stabilize first.
    """
    if epoch < warmup_epochs:
        return target_lambda * (epoch / warmup_epochs)
    return target_lambda


# ─── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, target_lambda, epoch,
                    warmup_epochs, device):
    """
    One full pass over training data.

    Total Loss = CrossEntropyLoss + lambda * SparsityLoss   (as per spec)
    Both weight and gate_scores are updated automatically since both
    are nn.Parameter and passed to the same optimizer.
    """
    model.train()
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    correct     = 0
    total       = 0
    lam         = get_lambda_for_epoch(target_lambda, epoch, warmup_epochs)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)

        # classification loss (spec: nn.CrossEntropyLoss)
        ce_loss = criterion(logits, labels)

        # sparsity loss: L1 of gates across ALL prunable layers (spec requirement)
        all_gates = torch.cat([
            torch.sigmoid(layer.gate_scores).reshape(-1)
            for layer in model.get_prunable_layers()
        ])
        sparsity_loss = all_gates.mean()

        # total loss (spec: Total Loss = ClassificationLoss + λ * SparsityLoss)
        loss = ce_loss + lam * sparsity_loss

        loss.backward()


        optimizer.step()

        total_loss += loss.item()
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc      = 100.0 * correct / total
    return avg_loss, acc


# ─── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """Compute test accuracy on the given loader."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits  = model(imgs)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ─── Full training run for one lambda ──────────────────────────────────────────
def train_model(target_lambda, train_loader, test_loader, cfg):
    """
    Train SelfPruningNet for cfg["epochs"] epochs with given lambda.
    Returns dict with final metrics, gate values, history, and model.
    """
    device        = cfg["device"]
    epochs        = cfg["epochs"]
    lr            = cfg["lr"]
    weight_decay  = cfg["weight_decay"]
    warmup_epochs = cfg["warmup_epochs"]

    print(f"\n{'='*60}")
    print(f"  Training with λ = {target_lambda}")
    print(f"{'='*60}")

    model = SelfPruningNet(num_classes=10).to(device)

    # two param groups:
    #   - actual weights: apply weight_decay (standard regularization)
    #   - gate_scores:    NO weight_decay (L1 sparsity loss handles this)
    #     mixing weight_decay with gate_scores would interfere with L1 behavior
    weight_params = []
    gate_params   = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            weight_params.append(param)

    optimizer = optim.AdamW([
        {"params": weight_params, "weight_decay": weight_decay},
        {"params": gate_params,   "weight_decay": 0.0},
    ], lr=lr)

    # cosine annealing: smooth LR decay, avoids sharp cliff of step schedules
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc":  [],
        "test_acc":   [],
        "sparsity":   [],
    }
    best_test_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer,
            target_lambda, epoch, warmup_epochs, device)

        test_acc = evaluate(model, test_loader, device)

        # sparsity: % of gates below threshold
        sparsity_pct, _ = model.compute_sparsity(threshold=cfg["prune_thresh"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity_pct)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lam_eff = get_lambda_for_epoch(target_lambda, epoch, warmup_epochs)
            elapsed = time.time() - t0
            print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
                  f"loss={train_loss:.4f}  "
                  f"train={train_acc:.1f}%  "
                  f"test={test_acc:.1f}%  "
                  f"sparsity={sparsity_pct:.1f}%  "
                  f"λ_eff={lam_eff:.5f}  "
                  f"({elapsed:.0f}s)")

    # final evaluation
    final_test_acc           = evaluate(model, test_loader, device)
    final_sparsity, all_gates = model.compute_sparsity(threshold=cfg["prune_thresh"])
    layer_sparsities          = model.get_layer_sparsities(threshold=cfg["prune_thresh"])

    print(f"\n  Final test accuracy : {final_test_acc:.2f}%")
    print(f"  Final sparsity      : {final_sparsity:.2f}%")
    print(f"  Layer-wise sparsity:")
    for name, pct in layer_sparsities:
        print(f"    {name}: {pct:.1f}%")

    return {
        "lambda":         target_lambda,
        "test_acc":       final_test_acc,
        "sparsity":       final_sparsity,
        "all_gates":      all_gates,
        "history":        history,
        "layer_sparsity": layer_sparsities,
        "model":          model,
    }


# ─── Plotting ──────────────────────────────────────────────────────────────────
def plot_gate_distribution(results_list, results_dir, prune_thresh):
    """
    Gate value histogram for each lambda.
    Successful pruning shows bimodal distribution:
      - large spike near 0   (dead / pruned gates)
      - smaller cluster near 0.8-1.0 (active gates)
    """
    n         = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        gates = res["all_gates"]
        lam   = res["lambda"]
        acc   = res["test_acc"]
        spar  = res["sparsity"]

        ax.hist(gates, bins=80, color="#4C72B0", edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.set_title(f"λ = {lam}\nAcc={acc:.1f}%  Sparsity={spar:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=prune_thresh, color="red", linestyle="--",
                   linewidth=1.5, label=f"Prune threshold ({prune_thresh})")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.suptitle("Gate Value Distributions — Self-Pruning Network (CIFAR-10)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "gate_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved gate distribution plot → {path}")


def plot_training_curves(results_list, results_dir):
    """Test accuracy and sparsity curves over all epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

    for i, res in enumerate(results_list):
        hist = res["history"]
        lam  = res["lambda"]
        col  = colors[i % len(colors)]
        axes[0].plot(hist["test_acc"], color=col, label=f"λ={lam}", linewidth=2)
        axes[1].plot(hist["sparsity"], color=col, label=f"λ={lam}", linewidth=2)

    for ax, title, ylabel in zip(
        axes,
        ["Test Accuracy over Epochs", "Sparsity Level over Epochs"],
        ["Test Accuracy (%)", "Sparsity (%)"]
    ):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training Dynamics — Self-Pruning Network", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves      → {path}")


# ─── Console table ─────────────────────────────────────────────────────────────
def print_results_table(results_list):
    """Print lambda / accuracy / sparsity summary."""
    print("\n" + "=" * 55)
    print("  Results Summary")
    print("=" * 55)
    print(f"  {'Lambda':>10}  {'Test Acc (%)':>14}  {'Sparsity (%)':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
    for res in results_list:
        print(f"  {res['lambda']:>10.4f}  "
              f"{res['test_acc']:>14.2f}  "
              f"{res['sparsity']:>14.2f}")
    print("=" * 55)


# ─── Markdown report ───────────────────────────────────────────────────────────
def save_report(results_list, results_dir):
    """Write the required markdown report: explanation + table + analysis."""
    md = []
    md.append("# Self-Pruning Neural Network — Results Report\n")


    md.append("## Results Table\n")
    md.append("| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |")
    md.append("|:----------:|:-----------------:|:------------------:|")
    for res in results_list:
        md.append(
            f"| {res['lambda']} | {res['test_acc']:.2f} | {res['sparsity']:.2f} |")
    md.append("")

    md.append("## Layer-wise Sparsity (Best Model)\n")
    best = max(results_list, key=lambda r: r["test_acc"])
    md.append(
        f"Best model: λ = {best['lambda']}, Test Acc = {best['test_acc']:.2f}%\n")
    md.append("| Layer | Sparsity (%) |")
    md.append("|:------|:------------:|")
    for name, pct in best["layer_sparsity"]:
        md.append(f"| {name} | {pct:.1f} |")
    md.append("")

    path = os.path.join(results_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved markdown report      → {path}")

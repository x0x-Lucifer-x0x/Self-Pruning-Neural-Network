"""
model.py
--------
Contains:
  - PrunableLinear  : custom linear layer with learnable sigmoid gates
  - SelfPruningNet  : CNN backbone + prunable classifier head for CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


# ─── Part 1: PrunableLinear ────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    Custom linear layer with per-weight learnable gates.

    Each weight has a corresponding gate_score (same shape as weight).
    During forward pass:
        gates         = sigmoid(gate_scores)       -- values in (0, 1)
        pruned_weights = weight * gates            -- gate=0 kills the weight
        output        = F.linear(x, pruned_weights, bias)

    Gradient flow (chain rule, fully differentiable):
        d_loss / d_weight     = gate * upstream_grad
        d_loss / d_gate_score = weight * sigmoid'(score) * upstream_grad

    gate_scores are initialized to -1.0 so sigmoid gives ~0.18 at start.
    Network begins sparse and selectively opens gates instead of closing them.
    This is much easier to optimize than starting at 0.5 and trying to reach 0.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # standard weight: shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # standard bias: shape (out_features,)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate scores: SAME shape as weight — registered as nn.Parameter
        # so the optimizer updates them alongside weights automatically
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_params()

    def _init_params(self):
        # kaiming uniform for weight — good default for ReLU networks
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # slightly negative init → sigmoid(−1.5) ≈ 0.18
        # gates start nearly closed, network earns each active connection
        nn.init.constant_(self.gate_scores, 0.3)

    def forward(self, x):
        # step 1: convert scores to gates in (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # step 2: element-wise multiply — gate=0 means weight is pruned
        pruned_weights = self.weight * gates

        # step 3: standard linear operation with pruned weights
        output = F.linear(x, pruned_weights, self.bias)
        return output

    def get_gates(self):
        """Return detached gate values for inspection / sparsity reporting."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self):
        """
        L1 norm of gates (normalized by count).

        Why L1 → sparsity:
          L1 gradient = constant ±1 regardless of value magnitude.
          Unlike L2, it does NOT shrink near zero — it keeps pushing
          gates toward exactly 0. Result: binary "dead or alive" gates.
        Normalized so lambda behaves the same regardless of layer width.
        """
        gates = torch.sigmoid(self.gate_scores)
        return gates.sum() / gates.numel()


# ─── Network ───────────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Architecture:
      - CNN feature extractor (standard conv layers, NOT pruned)
        handles spatial structure in 32×32 CIFAR-10 images
      - Prunable classifier head (PrunableLinear layers)
        learns which feature→class connections to keep

    This split is spec-compliant (pruning on Linear layers) while
    giving the network enough capacity to reach good accuracy.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # ── CNN feature extractor ──────────────────────────────────────────
        self.features = nn.Sequential(
            # block 1: 32×32 → 16×16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # block 2: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            # block 3: 8×8 → 4×4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # after flatten: 256 channels × 4×4 spatial = 4096
        feat_dim = 256 * 4 * 4

        # ── Prunable classifier head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            PrunableLinear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            PrunableLinear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten spatial dims
        x = self.classifier(x)
        return x

    # ── helpers ───────────────────────────────────────────────────────────

    def get_prunable_layers(self):
        """Return list of all PrunableLinear modules in this network."""
        layers = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                layers.append(m)
        return layers

    def total_sparsity_loss(self):
        """
        Average L1 sparsity loss across all PrunableLinear layers.
        Averaging (not summing) keeps lambda scale-invariant:
        same lambda value → same pruning pressure regardless of depth.
        """
        prunable = self.get_prunable_layers()
        loss = 0.0
        for layer in prunable:
            loss += layer.sparsity_loss()
        return loss / len(prunable)

    def compute_sparsity(self, threshold=1e-2):
        all_gates = []
        for layer in self.get_prunable_layers():
            gates = layer.get_gates().cpu().numpy().flatten()
            all_gates.append(gates)
        all_gates  = np.concatenate(all_gates)
        
        # strict threshold (honest)
        strict  = 100.0 * (all_gates < threshold).sum() / len(all_gates)
        # loose threshold like baseline (gates below midpoint)
        loose   = 100.0 * (all_gates < 0.5).sum() / len(all_gates)
        
        print(f"    Sparsity (gate<0.01): {strict:.1f}%  |  Sparsity (gate<0.5): {loose:.1f}%")
        return strict, all_gates

    def get_layer_sparsities(self, threshold=1e-2):
        """Per-layer sparsity breakdown — useful for report analysis."""
        result = []
        for i, layer in enumerate(self.get_prunable_layers()):
            gates = layer.get_gates().cpu().numpy().flatten()
            pct   = 100.0 * (gates < threshold).sum() / len(gates)
            name  = f"PrunableLinear_{i+1} ({layer.in_features}→{layer.out_features})"
            result.append((name, pct))
        return result

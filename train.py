#!/usr/bin/env python3
"""
train.py — Training loop for the multi-task Attention-Augmented Causal TCN.

Part of the *Intraoperative Surgical Phase & Gesture Recognition Engine*.

Pipeline
--------
1. **Data.** Train and validation windows come from *independently seeded* simulated
   procedures. Windows overlap (stride < window), so a random split of one procedure would
   put near-duplicate windows on both sides and inflate validation scores. Separate seeds
   mimic the real-world rule of splitting by *surgery*, never by frame.
2. **Normalization.** Per-feature mean/std are computed on the training set only, loaded
   into the model's ``InputNormalizer`` buffers, and also written explicitly into the
   checkpoint for ``inference.py`` and any other runtime.
3. **Loss.** ``L = CE_gesture + λ · CE_anomaly``, both class-weighted by inverse frequency.
   Anomalies are ~5-11% of windows. Unweighted, the anomaly head can reach ~83% accuracy
   by always predicting NORMAL, which is useless for a safety system.
4. **Optimization.** AdamW with a OneCycle LR schedule and gradient clipping. This is a
   robust default for small attention models that converges within ~25 epochs.
5. **Selection.** The checkpoint with the best validation *macro-F1*, averaged over both
   heads, is kept. Macro-F1 weights rare classes equally, which is how clinical risk works.

Usage::

    python train.py --epochs 25 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from model import AttentionTCN, ModelConfig, count_parameters, verify_causality
from simulator import (DEFAULT_SAMPLE_RATE_HZ, FEATURE_NAMES, Anomaly, Gesture,
                       SimulatorConfig, generate_windowed_dataset)

GESTURE_NAMES: list[str] = [g.name for g in Gesture]
ANOMALY_NAMES: list[str] = [a.name for a in Anomaly]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


@dataclass
class SplitData:
    """One dataset split as tensors: X (N, T, F), gesture labels (N,), anomaly labels (N,)."""

    x: Tensor
    y_gesture: Tensor
    y_anomaly: Tensor

    def loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(self.x, self.y_gesture, self.y_anomaly)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def build_split(num_windows: int, window_size: int, stride: int, seed: int) -> SplitData:
    """Generate one split from an independently seeded simulated procedure."""
    x, y_g, y_a = generate_windowed_dataset(
        num_windows, window_size=window_size, stride=stride,
        config=SimulatorConfig(seed=seed))
    return SplitData(torch.from_numpy(x), torch.from_numpy(y_g), torch.from_numpy(y_a))


def compute_normalization(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-feature mean/std over every frame of every training window."""
    flat = x.reshape(-1, x.shape[-1]).double()
    return flat.mean(dim=0).float(), flat.std(dim=0).float()


def inverse_frequency_weights(labels: Tensor, num_classes: int,
                              max_weight: float = 10.0) -> Tensor:
    """Class weights ``w_c = N / (C · n_c)``, clipped and renormalized to mean 1.

    Clipping stops a class that is nearly absent from producing an exploding gradient.
    Renormalizing keeps the loss scale (and therefore the effective learning rate)
    comparable to unweighted CrossEntropy.
    """
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    weights = (labels.numel() / (num_classes * counts)).clamp_max(max_weight)
    return weights / weights.mean()


# --------------------------------------------------------------------------------------
# Loss & evaluation
# --------------------------------------------------------------------------------------


class MultiTaskLoss(nn.Module):
    """Weighted sum of two class-weighted CrossEntropy losses.

    Light label smoothing (0.05) reduces over-confidence on the simulator's clean labels.
    That helps calibration, which matters because the dashboard thresholds alerts on
    predicted probabilities.
    """

    def __init__(self, gesture_weights: Tensor, anomaly_weights: Tensor,
                 anomaly_loss_weight: float = 1.0, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.gesture_ce = nn.CrossEntropyLoss(weight=gesture_weights,
                                              label_smoothing=label_smoothing)
        self.anomaly_ce = nn.CrossEntropyLoss(weight=anomaly_weights,
                                              label_smoothing=label_smoothing)
        self.anomaly_loss_weight = anomaly_loss_weight

    def forward(self, gesture_logits: Tensor, anomaly_logits: Tensor,
                y_gesture: Tensor, y_anomaly: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        loss_g = self.gesture_ce(gesture_logits, y_gesture)
        loss_a = self.anomaly_ce(anomaly_logits, y_anomaly)
        return loss_g + self.anomaly_loss_weight * loss_a, loss_g, loss_a


@dataclass
class EvalResult:
    """Validation metrics plus raw predictions (kept for the final report)."""

    loss: float
    gesture_acc: float
    gesture_f1: float
    anomaly_acc: float
    anomaly_f1: float
    y_gesture: np.ndarray
    p_gesture: np.ndarray
    y_anomaly: np.ndarray
    p_anomaly: np.ndarray

    @property
    def selection_score(self) -> float:
        """Mean macro-F1 across both heads: the checkpoint selection criterion."""
        return 0.5 * (self.gesture_f1 + self.anomaly_f1)

    def summary(self) -> dict[str, float]:
        # Cast to built-in float: sklearn returns np.float64, which weights-only
        # checkpoint loading rejects as an unknown global.
        return {"loss": float(self.loss), "gesture_acc": float(self.gesture_acc),
                "gesture_f1": float(self.gesture_f1), "anomaly_acc": float(self.anomaly_acc),
                "anomaly_f1": float(self.anomaly_f1),
                "selection_score": float(self.selection_score)}


@torch.inference_mode()
def evaluate(model: AttentionTCN, loader: DataLoader, criterion: MultiTaskLoss,
             device: torch.device) -> EvalResult:
    """Run the model over a split and compute accuracy and macro-F1 for each head."""
    model.eval()
    total_loss, total_n = 0.0, 0
    ys_g, ps_g, ys_a, ps_a = [], [], [], []
    for x, y_g, y_a in loader:
        x, y_g, y_a = x.to(device), y_g.to(device), y_a.to(device)
        g_logits, a_logits = model(x)
        loss, _, _ = criterion(g_logits, a_logits, y_g, y_a)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        ys_g.append(y_g.cpu())
        ps_g.append(g_logits.argmax(-1).cpu())
        ys_a.append(y_a.cpu())
        ps_a.append(a_logits.argmax(-1).cpu())

    y_g_all, p_g_all = torch.cat(ys_g).numpy(), torch.cat(ps_g).numpy()
    y_a_all, p_a_all = torch.cat(ys_a).numpy(), torch.cat(ps_a).numpy()
    return EvalResult(
        loss=total_loss / max(total_n, 1),
        gesture_acc=accuracy_score(y_g_all, p_g_all),
        gesture_f1=f1_score(y_g_all, p_g_all, average="macro",
                            labels=list(range(len(GESTURE_NAMES))), zero_division=0),
        anomaly_acc=accuracy_score(y_a_all, p_a_all),
        anomaly_f1=f1_score(y_a_all, p_a_all, average="macro",
                            labels=list(range(len(ANOMALY_NAMES))), zero_division=0),
        y_gesture=y_g_all, p_gesture=p_g_all, y_anomaly=y_a_all, p_anomaly=p_a_all,
    )


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_one_epoch(model: AttentionTCN, loader: DataLoader, criterion: MultiTaskLoss,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    device: torch.device, grad_clip: float) -> dict[str, float]:
    """One pass over the training data. Returns mean total, gesture and anomaly losses."""
    model.train()
    sums = np.zeros(3)
    n = 0
    for x, y_g, y_a in loader:
        x, y_g, y_a = x.to(device), y_g.to(device), y_a.to(device)
        optimizer.zero_grad(set_to_none=True)
        g_logits, a_logits = model(x)
        loss, loss_g, loss_a = criterion(g_logits, a_logits, y_g, y_a)
        loss.backward()
        # Clipping guards against rare spikes from heavily weighted minority-class batches.
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        sums += np.array([loss.item(), loss_g.item(), loss_a.item()]) * x.size(0)
        n += x.size(0)
    total, g, a = sums / max(n, 1)
    return {"loss": total, "gesture_loss": g, "anomaly_loss": a}


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_checkpoint(model: AttentionTCN, mean: Tensor, std: Tensor,
                     gesture_weights: Tensor, anomaly_weights: Tensor,
                     result: EvalResult, epoch: int,
                     args: argparse.Namespace) -> dict[str, Any]:
    """Assemble a self-describing checkpoint.

    Only tensors and plain Python types are stored, so it loads safely with
    ``torch.load(..., weights_only=True)``. Unpickling arbitrary objects from a model
    file is a real supply-chain risk, and weights-only loading avoids it.
    """
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.config.to_dict(),
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_names": list(FEATURE_NAMES),
        },
        "class_names": {"gesture": GESTURE_NAMES, "anomaly": ANOMALY_NAMES},
        "class_weights": {"gesture": gesture_weights.tolist(),
                          "anomaly": anomaly_weights.tolist()},
        "window_size": args.window_size,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "epoch": epoch,
        "val_metrics": result.summary(),
        # Paths become strings so the checkpoint contains only safe, plain types.
        "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch_version": str(torch.__version__),  # TorchVersion is not a plain str
    }


def print_final_report(result: EvalResult) -> None:
    """Per-class precision/recall/F1 and confusion matrices for the best checkpoint."""
    for title, y, p, names in (
        ("GESTURE HEAD", result.y_gesture, result.p_gesture, GESTURE_NAMES),
        ("ANOMALY HEAD", result.y_anomaly, result.p_anomaly, ANOMALY_NAMES),
    ):
        labels = list(range(len(names)))
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        print(classification_report(y, p, labels=labels, target_names=names,
                                    digits=3, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):")
        cm = confusion_matrix(y, p, labels=labels)
        width = max(len(n) for n in names) + 2
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for name, row in zip(names, cm):
            print(f"{name:<{width}}" + "".join(f"{v:>{width}}" for v in row))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Attention-Augmented Causal TCN")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (OneCycle)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--train-windows", type=int, default=20_000)
    p.add_argument("--val-windows", type=int, default=4_000)
    p.add_argument("--window-size", type=int, default=60, help="frames (60 = 2 s @ 30 Hz)")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--anomaly-loss-weight", type=float, default=1.0, help="λ in the loss")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--output", type=Path, default=Path("checkpoints/best_model.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)

    # ---- data --------------------------------------------------------------------------
    t0 = time.perf_counter()
    train = build_split(args.train_windows, args.window_size, args.stride, seed=args.seed)
    val = build_split(args.val_windows, args.window_size, args.stride, seed=args.seed + 10_000)
    print(f"Generated {len(train.x):,} train / {len(val.x):,} val windows "
          f"of shape {tuple(train.x.shape[1:])} in {time.perf_counter() - t0:.1f} s")
    for name, labels, names in (("Gesture", train.y_gesture, GESTURE_NAMES),
                                ("Anomaly", train.y_anomaly, ANOMALY_NAMES)):
        frac = torch.bincount(labels, minlength=len(names)).float() / len(labels)
        print(f"  {name} balance (train): " + ", ".join(
            f"{n} {100 * f:.1f}%" for n, f in zip(names, frac.tolist())))

    mean, std = compute_normalization(train.x)
    gesture_weights = inverse_frequency_weights(train.y_gesture, len(GESTURE_NAMES))
    anomaly_weights = inverse_frequency_weights(train.y_anomaly, len(ANOMALY_NAMES))
    print("  Class weights: gesture " + str([round(w, 2) for w in gesture_weights.tolist()])
          + ", anomaly " + str([round(w, 2) for w in anomaly_weights.tolist()]))

    # ---- model ---------------------------------------------------------------------------
    model = AttentionTCN(ModelConfig(hidden_dim=args.hidden_dim))
    model.normalizer.set_statistics(mean, std)
    assert verify_causality(model), "Model leaks future information; refusing to train."
    model.to(device)
    print(f"Model: {count_parameters(model):,} params, "
          f"receptive field {model.config.receptive_field} frames, device {device}")

    criterion = MultiTaskLoss(gesture_weights.to(device), anomaly_weights.to(device),
                              args.anomaly_loss_weight)
    train_loader = train.loader(args.batch_size, shuffle=True)
    val_loader = val.loader(512, shuffle=False)   # eval batches can be much larger
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    # ---- loop ----------------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_result, epochs_without_improvement = -1.0, None, 0
    header = (f"{'epoch':>5} | {'train':>6} {'L_g':>6} {'L_a':>6} | {'val':>6} | "
              f"{'G-acc':>6} {'G-F1':>6} | {'A-acc':>6} {'A-F1':>6} | {'time':>5}")
    print("\n" + header + "\n" + "-" * len(header))

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                             device, args.grad_clip)
        res = evaluate(model, val_loader, criterion, device)
        improved = res.selection_score > best_score
        print(f"{epoch:>5} | {tr['loss']:>6.3f} {tr['gesture_loss']:>6.3f} "
              f"{tr['anomaly_loss']:>6.3f} | {res.loss:>6.3f} | "
              f"{res.gesture_acc:>6.3f} {res.gesture_f1:>6.3f} | "
              f"{res.anomaly_acc:>6.3f} {res.anomaly_f1:>6.3f} | "
              f"{time.perf_counter() - start:>4.1f}s" + ("  ★ saved" if improved else ""))

        if improved:
            best_score, best_result, epochs_without_improvement = res.selection_score, res, 0
            torch.save(build_checkpoint(model, mean, std, gesture_weights, anomaly_weights,
                                        res, epoch, args), args.output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    assert best_result is not None
    print(f"\nBest mean macro-F1 = {best_score:.3f} → {args.output}")
    print_final_report(best_result)


if __name__ == "__main__":
    main()
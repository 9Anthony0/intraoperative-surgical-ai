#!/usr/bin/env python3
"""
model.py — Attention-Augmented Causal Temporal Convolutional Network (multi-task).

Part of the *Intraoperative Surgical Phase & Gesture Recognition Engine*.

Architecture
------------
::

    Input (B, T=60, 9)  raw telemetry, in physical units
       │
       ├─► InputNormalizer        (mean/std buffers, baked into the graph)
       ├─► Input projection       Linear 9 → H
       ├─► CausalTCNBlock × 4     dilations 1, 2, 4, 8 (two causal convs per block)
       ├─► CausalSelfAttention    multi-head, causal mask, residual + LayerNorm
       ├─► Temporal pooling       [last step ‖ attention-weighted mean]  → 2H
       ├─► Gesture head  → 3 logits   (Idle, Suturing, Dissection)
       └─► Anomaly head  → 3 logits   (Normal, Tremor, Hesitation)

Why every layer is causal
-------------------------
The model classifies the *last* frame of each window, so within a window everything is
already "the past". Strict causality still matters, for three reasons:

1. **Streaming consistency.** The representation at step ``t`` depends only on frames
   ``≤ t``. It is identical whether computed now or later, when more frames have arrived.
   That is what makes per-frame outputs trustworthy as live predictions, and it is the
   prerequisite for incremental (cached) inference as a future latency optimization.
2. **No edge artifacts.** Symmetric ("same") padding feeds zeros in place of the future at
   the last timestep, which is exactly the step we classify. Left-only padding avoids it.
3. **Verifiability.** ``verify_causality`` below proves it empirically by perturbing future
   frames and asserting that past outputs do not change.

Latency vs. accuracy trade-offs
-------------------------------
* **TCN over RNN.** All timesteps are processed in parallel with a fixed, data-independent
  cost, so latency is predictable. Residual paths keep gradients healthy across long
  sequences (no vanishing gradients through recurrent time steps).
* **Two convs per block.** The receptive field is ``1 + 2·(k−1)·Σdilations = 61`` frames,
  which covers the full 2 s window with only 4 blocks. One conv per block would stop at
  31 frames (~1 s) and leave the far half of the window to attention alone.
* **Attention on top, not instead.** Convolutions extract local motion primitives cheaply,
  and a single attention layer then compares every step against the full context. At
  T = 60 the O(T²) cost is negligible (3,600 scores per head).
* **Small hidden size (H = 64).** About 0.14 M parameters. Batch-1 CPU forward passes run
  in single-digit milliseconds, well inside the 15 ms model budget in the README.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyperparameters. Serialized into every checkpoint so the model rebuilds exactly."""

    num_features: int = 9
    hidden_dim: int = 64
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_heads: int = 4
    dropout: float = 0.1
    num_gestures: int = 3
    num_anomalies: int = 3
    max_seq_len: int = 256   # upper bound for the precomputed causal attention mask

    @property
    def receptive_field(self) -> int:
        """Frames of history visible to the TCN stack (two convs per block)."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**data)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


class InputNormalizer(nn.Module):
    """Per-feature standardization with statistics stored as (non-trainable) buffers.

    Baking normalization into the module makes it part of the model's ``state_dict``, so a
    checkpoint cannot be loaded with the wrong statistics and there is no separate
    preprocessing step for the serving path to get wrong (a classic train/serve skew bug).
    ``train.py`` also writes the raw statistics into the checkpoint explicitly, for
    transparency and for runtimes such as ONNX that consume the model differently.
    """

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))

    @torch.no_grad()
    def set_statistics(self, mean: Tensor, std: Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.eps))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class CausalConv1d(nn.Module):
    """1-D convolution that sees only the present and past.

    Standard ``Conv1d`` padding is symmetric. Here we pad ``(k − 1)·d`` zeros on the
    **left only**, so output ``t`` is a function of inputs ``t − (k−1)d … t``. The output
    length equals the input length, with no trimming needed.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, C, T)
        return self.conv(F.pad(x, (self.left_pad, 0)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, T) tensor.

    LayerNorm normalizes each timestep independently, so unlike BatchNorm it never mixes
    information across time (causality holds) and behaves identically at batch size 1
    during inference and at batch size 64 during training.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalTCNBlock(nn.Module):
    """Residual block: [CausalConv → LayerNorm → GELU → Dropout] × 2, plus skip connection.

    The identity skip gives gradients a direct path through the stack, which is what keeps
    deep dilated TCNs trainable on long sequences.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(2):
            layers += [
                CausalConv1d(channels, channels, kernel_size, dilation),
                ChannelLayerNorm(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, T) -> (B, C, T)
        return x + self.body(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, residual connection and LayerNorm.

    The mask is precomputed once as a buffer and sliced per call, so the hot path never
    allocates it. ``need_weights=False`` lets PyTorch use its fused attention kernels.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # True = "may not attend": everything strictly above the diagonal (the future).
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, C) -> (B, T, C)
        t = x.size(1)
        out, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:t, :t], need_weights=False)
        return self.norm(x + self.dropout(out))


class TemporalPooling(nn.Module):
    """Summarize (B, T, C) into (B, 2C) as [last step ‖ attention-weighted mean].

    The last step is the model's view of "right now", which gives the sharpest onset
    detection. The learned weighted mean adds a stable context summary, which reduces
    frame-to-frame flicker in the predictions. Both depend only on frames ≤ T, so the pooled
    vector stays causal with respect to the frame being classified.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.score(x), dim=1)          # (B, T, 1)
        context = (weights * x).sum(dim=1)                     # (B, C)
        return torch.cat((x[:, -1], context), dim=-1)          # (B, 2C)


def _make_head(in_dim: int, hidden: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------


class AttentionTCN(nn.Module):
    """Attention-augmented causal TCN with separate gesture and anomaly heads.

    Why two heads instead of one 9-way softmax over (gesture × anomaly)? The two tasks are
    orthogonal: a surgeon can be suturing *and* hesitating at the same time. Factorized
    heads share one motion representation (multi-task regularization), keep each output
    calibrated on its own label space, and let the anomaly loss be class-weighted without
    distorting the gesture classifier.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or ModelConfig()
        h = cfg.hidden_dim
        self.normalizer = InputNormalizer(cfg.num_features)
        self.input_proj = nn.Linear(cfg.num_features, h)
        self.tcn = nn.Sequential(*[
            CausalTCNBlock(h, cfg.kernel_size, d, cfg.dropout) for d in cfg.dilations
        ])
        self.attention = CausalSelfAttention(h, cfg.num_heads, cfg.dropout, cfg.max_seq_len)
        self.pool = TemporalPooling(h)
        self.gesture_head = _make_head(2 * h, h, cfg.num_gestures, cfg.dropout)
        self.anomaly_head = _make_head(2 * h, h, cfg.num_anomalies, cfg.dropout)

    def encode(self, x: Tensor) -> Tensor:
        """Map raw telemetry (B, T, F) to per-timestep causal features (B, T, H)."""
        z = self.input_proj(self.normalizer(x))               # (B, T, H)
        z = self.tcn(z.transpose(1, 2)).transpose(1, 2)        # convs run on (B, H, T)
        return self.attention(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Classify the last frame of each window.

        Args:
            x: raw (un-normalized) telemetry, shape (B, T, num_features).

        Returns:
            ``(gesture_logits, anomaly_logits)``, shapes (B, num_gestures) and
            (B, num_anomalies). A plain tuple (not a dict) keeps the model export-friendly.
        """
        pooled = self.pool(self.encode(x))
        return self.gesture_head(pooled), self.anomaly_head(pooled)


# --------------------------------------------------------------------------------------
# Utilities: parameter count, causality proof, checkpoint I/O, latency probe
# --------------------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def verify_causality(model: AttentionTCN, seq_len: int = 60, trials: int = 5,
                     atol: float = 1e-5) -> bool:
    """Empirically prove there is no future leakage in ``encode``.

    For random cut points ``t``, overwrite frames ``t+1 …`` with fresh noise and check that
    per-step features ``0 … t`` are unchanged. Any non-causal op (symmetric padding, an
    unmasked attention, BatchNorm across time) would fail this test.
    """
    was_training = model.training
    model.eval()
    try:
        x = torch.randn(2, seq_len, model.config.num_features)
        reference = model.encode(x)
        for t in torch.randint(0, seq_len - 1, (trials,)).tolist():
            corrupted = x.clone()
            corrupted[:, t + 1:] = torch.randn_like(corrupted[:, t + 1:]) * 10.0
            if not torch.allclose(reference[:, :t + 1], model.encode(corrupted)[:, :t + 1],
                                  atol=atol):
                return False
        return True
    finally:
        model.train(was_training)


def load_checkpoint(path: str | Path,
                    device: str | torch.device = "cpu") -> tuple[AttentionTCN, dict[str, Any]]:
    """Rebuild a model from a ``train.py`` checkpoint and put it in eval mode.

    Returns:
        ``(model, checkpoint_dict)``. The dict carries the normalization statistics,
        class names, window size and validation metrics for downstream consumers.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = AttentionTCN(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def measure_latency_ms(model: AttentionTCN, seq_len: int = 60, iterations: int = 200,
                       warmup: int = 20) -> dict[str, float]:
    """Batch-1 CPU forward-pass latency percentiles (the streaming workload)."""
    model.eval()
    x = torch.randn(1, seq_len, model.config.num_features)
    for _ in range(warmup):
        model(x)
    samples = torch.empty(iterations)
    for i in range(iterations):
        start = time.perf_counter()
        model(x)
        samples[i] = (time.perf_counter() - start) * 1e3
    return {f"p{q}": float(samples.quantile(q / 100)) for q in (50, 95, 99)}


if __name__ == "__main__":
    net = AttentionTCN()
    cfg = net.config
    print(f"AttentionTCN: {count_parameters(net):,} trainable parameters")
    print(f"TCN receptive field: {cfg.receptive_field} frames "
          f"({cfg.receptive_field / 30:.2f} s at 30 Hz)")
    g_logits, a_logits = net(torch.randn(8, 60, cfg.num_features))
    print(f"Output shapes: gesture {tuple(g_logits.shape)}, anomaly {tuple(a_logits.shape)}")
    print(f"Causality check (no future leakage): "
          f"{'PASS' if verify_causality(net) else 'FAIL'}")
    lat = measure_latency_ms(net)
    print(f"Batch-1 CPU latency ({torch.get_num_threads()} thread(s)): "
          + ", ".join(f"{k} {v:.2f} ms" for k, v in lat.items()))
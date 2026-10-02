#!/usr/bin/env python3
"""
inference.py — Low-latency streaming inference engine.

Part of the *Intraoperative Surgical Phase & Gesture Recognition Engine*.

Runtime topology
----------------
::

    ┌──────────────── background thread: "surgical-inference" ────────────────┐
    │  asyncio loop                                                           │
    │   AsyncTelemetryStreamer ──(30 Hz queue)──► consumer coroutine          │
    │          │                                    │                         │
    │          └──► TelemetryRingBuffer ◄───────────┤ latest 60-frame window  │
    │                                               ▼                         │
    │                          InferenceEngine (PyTorch, batch 1)             │
    │                                               ▼                         │
    │                    EMA smoothing → hysteresis alerts → phase tracker    │
    └───────────────────────────────────────────────┬─────────────────────────┘
                                                    │ lock-protected snapshot()
                                     Streamlit script thread (app.py)

Why a dedicated thread with its own event loop? Streamlit re-executes the whole script on
every interaction and does not host a persistent asyncio loop. Running the streamer on a
background thread keeps the 30 Hz clock independent of UI reruns. The UI then pulls a
consistent, copied ``snapshot()`` whenever it redraws (~4 Hz), so rendering never stalls
data acquisition and inference never waits on rendering.

Latency design
--------------
* **Preallocated input tensor.** Each window is copied into one reused ``(1, T, 9)``
  tensor, so there is no per-frame allocation on the hot path.
* **``torch.inference_mode``** disables autograd bookkeeping entirely.
* **Few intra-op threads (default 2).** For a 0.13 M-parameter model at batch 1, extra
  threads add synchronization overhead and contend with the UI thread. Fewer threads gives
  *lower and more stable* tail latency.
* **Staleness guard.** If a frame waited in the queue longer than two sample periods, it is
  skipped instead of inferred. The engine always works on the freshest data ("latest frame
  wins") and never builds a backlog.

Signal post-processing (why raw probabilities never drive alerts directly)
-------------------------------------------------------------------------
1. **EMA smoothing** of class probabilities: ``p̂_t = α·p_t + (1−α)·p̂_{t−1}``. At 30 Hz with
   α = 0.35 the effective time constant is ~70 ms. That suppresses single-frame flicker
   while adding far less delay than a human reaction time.
2. **Hysteresis.** An alert switches on at ``on_threshold`` but only switches off below
   ``0.7 × on_threshold``, so a probability hovering at the boundary cannot strobe.
3. **Debounce.** The smoothed probability must exceed the threshold for ``min_on_frames``
   consecutive frames (default 3, i.e. 100 ms) before an alert fires.

Important: the model contains its own ``InputNormalizer``. Feed it **raw** telemetry. The
normalization statistics stored in the checkpoint are informational here, and applying
them a second time would silently corrupt every prediction.

Usage::

    python inference.py --benchmark --iterations 1000
    python inference.py --live --duration 20
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Final

import numpy as np
import torch

from model import AttentionTCN, load_checkpoint
from simulator import (AsyncTelemetryStreamer, Gesture, KinematicSimulator, SimulatorConfig,
                       generate_windowed_dataset)

DEFAULT_CHECKPOINT: Final[Path] = Path("checkpoints/best_model.pt")
LATENCY_BUDGET_MS: Final[float] = 50.0


class AlertLevel(IntEnum):
    """Dashboard alert state, ordered by severity (higher value = higher priority)."""

    NOMINAL = 0
    TREMOR = 1       # yellow: fatigue / steadiness warning
    HESITATION = 2   # red: interrupted motor plan mid-task


# --------------------------------------------------------------------------------------
# Configuration & difficulty presets
# --------------------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Runtime knobs. All of them can be changed live from the dashboard."""

    ema_alpha: float = 0.35          # higher = more responsive, less smoothing
    tremor_threshold: float = 0.60   # smoothed P(tremor) needed to raise the alert
    hesitation_threshold: float = 0.60
    hysteresis_ratio: float = 0.70   # release alert below ratio × threshold
    min_on_frames: int = 3           # debounce: consecutive frames above threshold
    phase_debounce_frames: int = 8   # frames a new gesture must persist to start a phase
    num_threads: int = 2             # torch intra-op threads (see module docstring)
    history_size: int = 600          # predictions kept for plotting (20 s at 30 Hz)

    @staticmethod
    def thresholds_for_sensitivity(sensitivity: float) -> float:
        """Map a 0-1 "sensitivity" slider to a probability threshold (0.80 → 0.30).

        More sensitive means earlier alerts and more false alarms. In a safety net, recall
        is usually worth some precision, so the dashboard defaults to the middle.
        """
        s = float(np.clip(sensitivity, 0.0, 1.0))
        return 0.80 - 0.50 * s


#: Simulator settings for each difficulty level. Only anomaly *frequency and mix* change.
#: Sensor noise stays at the training value on purpose: raising it would create a
#: train/serve distribution shift and make the demo misleading rather than harder.
DIFFICULTY_PRESETS: Final[dict[str, dict[str, float]]] = {
    "Routine case": {"anomaly_mean_interval_s": 9.0, "hesitation_probability": 0.4},
    "Standard case": {"anomaly_mean_interval_s": 5.0, "hesitation_probability": 0.5},
    "Complex case": {"anomaly_mean_interval_s": 3.0, "hesitation_probability": 0.6},
    "Fatigued surgeon": {"anomaly_mean_interval_s": 2.0, "hesitation_probability": 0.35},
}


# --------------------------------------------------------------------------------------
# Output records
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Prediction:
    """One inference result, plus the context the dashboard needs to draw it."""

    frame_idx: int
    sim_time: float
    position: np.ndarray            # (3,) tool tip, mm
    velocity: float                 # mm/s
    grip: float                     # rad
    gesture_probs: np.ndarray       # (3,) EMA-smoothed
    anomaly_probs: np.ndarray       # (3,) EMA-smoothed
    raw_anomaly_probs: np.ndarray   # (3,) before smoothing (for diagnostics)
    gesture: Gesture                # smoothed argmax
    alert: AlertLevel
    model_latency_ms: float         # forward pass only
    e2e_latency_ms: float           # frame emission → alert decision
    true_gesture: int               # simulator ground truth (demo / evaluation only)
    true_anomaly: int


@dataclass(slots=True)
class PhaseSegment:
    """A contiguous block of one gesture on the workflow timeline."""

    gesture: int
    start: float
    end: float


@dataclass(slots=True)
class AlertEvent:
    """One alert episode, for the dashboard's event log."""

    level: AlertLevel
    start: float
    end: float | None = None
    peak_prob: float = 0.0

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


@dataclass
class ServiceSnapshot:
    """Immutable, copied view of service state for the UI thread."""

    running: bool
    predictions: list[Prediction]
    predicted_phases: list[PhaseSegment]
    true_phases: list[PhaseSegment]
    alert_events: list[AlertEvent]
    frames_received: int
    frames_skipped: int
    stream_hz: float
    error: str | None = None

    @property
    def latest(self) -> Prediction | None:
        return self.predictions[-1] if self.predictions else None


# --------------------------------------------------------------------------------------
# Core components
# --------------------------------------------------------------------------------------


class InferenceEngine:
    """Thin, allocation-free wrapper around the trained model for batch-1 inference."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, num_threads: int = 2,
                 warmup_iterations: int = 20) -> None:
        torch.set_num_threads(max(1, num_threads))
        self.model: AttentionTCN
        self.model, self.checkpoint = load_checkpoint(checkpoint, device="cpu")
        self.window_size: int = int(self.checkpoint["window_size"])
        self.num_features: int = self.model.config.num_features
        self.gesture_names: list[str] = self.checkpoint["class_names"]["gesture"]
        self.anomaly_names: list[str] = self.checkpoint["class_names"]["anomaly"]
        self._input = torch.zeros(1, self.window_size, self.num_features)
        # Warm-up: the first calls pay one-off costs (kernel selection, lazy allocations)
        # that must not show up as latency spikes during a live procedure.
        for _ in range(warmup_iterations):
            self.predict(np.zeros((self.window_size, self.num_features), dtype=np.float32))

    @torch.inference_mode()
    def predict(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Classify the last frame of a raw ``(T, 9)`` window.

        Returns:
            ``(gesture_probs, anomaly_probs, forward_latency_ms)``.
        """
        start = time.perf_counter()
        self._input[0].copy_(torch.from_numpy(window))   # reuse the preallocated buffer
        g_logits, a_logits = self.model(self._input)
        g = torch.softmax(g_logits[0], dim=-1).numpy()
        a = torch.softmax(a_logits[0], dim=-1).numpy()
        return g, a, (time.perf_counter() - start) * 1e3


class EMASmoother:
    """Exponential moving average over probability vectors (stays a valid distribution)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = probs.copy()
        else:
            self._state = self.alpha * probs + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


class HysteresisAlert:
    """Debounced on/off detector for one anomaly class."""

    def __init__(self) -> None:
        self.active = False
        self._above = 0

    def update(self, prob: float, on_threshold: float, off_ratio: float,
               min_on_frames: int) -> bool:
        if self.active:
            if prob < on_threshold * off_ratio:
                self.active, self._above = False, 0
        else:
            self._above = self._above + 1 if prob >= on_threshold else 0
            if self._above >= min_on_frames:
                self.active = True
        return self.active


class PhaseTracker:
    """Turns a per-frame label stream into timeline segments, with optional debounce.

    Without debounce a single misclassified frame would split one suturing phase into three
    segments. A new phase is only committed after it persists for ``debounce`` frames, and
    its start time is then backdated to the first frame of the run, so the timeline stays
    accurate rather than lagging.
    """

    def __init__(self, debounce: int = 0, max_segments: int = 200) -> None:
        self.debounce = debounce
        self.segments: deque[PhaseSegment] = deque(maxlen=max_segments)
        self._candidate: int | None = None
        self._candidate_start = 0.0
        self._candidate_count = 0

    def update(self, label: int, t: float) -> None:
        if not self.segments:
            self.segments.append(PhaseSegment(label, t, t))
            return
        current = self.segments[-1]
        if label == current.gesture:
            current.end = t
            self._candidate, self._candidate_count = None, 0
            return
        if label != self._candidate:
            self._candidate, self._candidate_start, self._candidate_count = label, t, 0
        self._candidate_count += 1
        if self._candidate_count > self.debounce:
            current.end = self._candidate_start
            self.segments.append(PhaseSegment(label, self._candidate_start, t))
            self._candidate, self._candidate_count = None, 0
        else:
            current.end = t


# --------------------------------------------------------------------------------------
# Background service
# --------------------------------------------------------------------------------------


class LiveInferenceService:
    """Owns the simulator, streamer, model and post-processing on a background thread.

    Thread-safety contract: the background thread is the only *writer* of history state.
    The UI thread only calls ``snapshot()`` (which copies under a lock) and the ``set_*``
    methods (which swap small config values; attribute assignment is atomic in CPython).
    """

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 config: InferenceConfig | None = None,
                 difficulty: str = "Standard case", seed: int | None = None) -> None:
        self.config = config or InferenceConfig()
        self.engine = InferenceEngine(checkpoint, num_threads=self.config.num_threads)
        self.simulator = KinematicSimulator(SimulatorConfig(seed=seed))
        self.set_difficulty(difficulty)
        self._gesture_ema = EMASmoother(self.config.ema_alpha)
        self._anomaly_ema = EMASmoother(self.config.ema_alpha)
        self._tremor_alert = HysteresisAlert()
        self._hesitation_alert = HysteresisAlert()
        self._predicted_phases = PhaseTracker(self.config.phase_debounce_frames)
        self._true_phases = PhaseTracker(debounce=0)
        self._predictions: deque[Prediction] = deque(maxlen=self.config.history_size)
        self._events: deque[AlertEvent] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_received = 0
        self._frames_skipped = 0
        self._arrivals: deque[float] = deque(maxlen=90)
        self._error: str | None = None
        self.difficulty = difficulty

    # ---- lifecycle --------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._thread_main, name="surgical-inference",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._consume())
        except Exception as exc:  # surface failures in the UI instead of dying silently
            self._error = f"{type(exc).__name__}: {exc}"

    # ---- live settings ------------------------------------------------------------------

    def set_difficulty(self, name: str) -> None:
        """Adjust anomaly frequency/mix on the running simulator (read each step)."""
        for key, value in DIFFICULTY_PRESETS[name].items():
            setattr(self.simulator.config, key, value)
        self.difficulty = name

    def set_sensitivity(self, sensitivity: float) -> None:
        threshold = InferenceConfig.thresholds_for_sensitivity(sensitivity)
        self.config.tremor_threshold = threshold
        self.config.hesitation_threshold = threshold

    def set_smoothing(self, alpha: float) -> None:
        self.config.ema_alpha = alpha
        self._gesture_ema.alpha = alpha
        self._anomaly_ema.alpha = alpha

    # ---- hot path -------------------------------------------------------------------------

    async def _consume(self) -> None:
        streamer = AsyncTelemetryStreamer(self.simulator, buffer_capacity=300)
        period, window = self.simulator.dt, self.engine.window_size
        async with streamer:
            async with aclosing(streamer.frames()) as frames:
                async for frame in frames:
                    if self._stop_flag.is_set():
                        break
                    now = time.perf_counter()
                    self._frames_received += 1
                    self._arrivals.append(now)
                    self._true_phases.update(int(frame.gesture), frame.sim_time)
                    if len(streamer.ring_buffer) < window:
                        continue                      # still filling the first 2 s
                    if now - frame.wall_time > 2 * period:
                        self._frames_skipped += 1     # stale: latest frame wins
                        continue
                    features, *_ = streamer.ring_buffer.latest_window(window)
                    self._process(frame.frame_idx, frame.sim_time, frame.wall_time,
                                  features, int(frame.gesture), int(frame.anomaly))

    def _process(self, frame_idx: int, sim_time: float, wall_time: float,
                 features: np.ndarray, true_gesture: int, true_anomaly: int) -> None:
        cfg = self.config
        g_raw, a_raw, model_ms = self.engine.predict(features)
        g = self._gesture_ema(g_raw)
        a = self._anomaly_ema(a_raw)

        tremor = self._tremor_alert.update(float(a[1]), cfg.tremor_threshold,
                                           cfg.hysteresis_ratio, cfg.min_on_frames)
        hesitation = self._hesitation_alert.update(float(a[2]), cfg.hesitation_threshold,
                                                   cfg.hysteresis_ratio, cfg.min_on_frames)
        # Hesitation outranks tremor: an interrupted motor plan mid-task is the more
        # immediate risk, so it wins the (single) banner when both are active.
        alert = (AlertLevel.HESITATION if hesitation
                 else AlertLevel.TREMOR if tremor else AlertLevel.NOMINAL)
        gesture = Gesture(int(np.argmax(g)))
        last = features[-1]
        prediction = Prediction(
            frame_idx=frame_idx, sim_time=sim_time, position=last[:3].copy(),
            velocity=float(last[8]), grip=float(last[7]), gesture_probs=g,
            anomaly_probs=a, raw_anomaly_probs=a_raw, gesture=gesture, alert=alert,
            model_latency_ms=model_ms,
            e2e_latency_ms=(time.perf_counter() - wall_time) * 1e3,
            true_gesture=true_gesture, true_anomaly=true_anomaly,
        )
        with self._lock:
            self._predictions.append(prediction)
            self._predicted_phases.update(int(gesture), sim_time)
            self._update_events(alert, sim_time, a)

    def _update_events(self, alert: AlertLevel, t: float, probs: np.ndarray) -> None:
        """Open, extend, or close alert episodes for the event log (caller holds lock)."""
        current = self._events[-1] if self._events and self._events[-1].end is None else None
        if current is not None and current.level != alert:
            current.end = t
            current = None
        if alert is not AlertLevel.NOMINAL:
            if current is None:
                current = AlertEvent(alert, t)
                self._events.append(current)
            current.peak_prob = max(current.peak_prob, float(probs[int(alert)]))

    # ---- UI access --------------------------------------------------------------------------

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            predictions = list(self._predictions)
            predicted = [PhaseSegment(s.gesture, s.start, s.end)
                         for s in self._predicted_phases.segments]
            events = [AlertEvent(e.level, e.start, e.end, e.peak_prob) for e in self._events]
        true = [PhaseSegment(s.gesture, s.start, s.end) for s in self._true_phases.segments]
        arrivals = list(self._arrivals)
        hz = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
        return ServiceSnapshot(self.running, predictions, predicted, true, events,
                               self._frames_received, self._frames_skipped, hz, self._error)


# --------------------------------------------------------------------------------------
# CLI: benchmark and headless live run
# --------------------------------------------------------------------------------------


def _percentiles(samples: list[float]) -> str:
    arr = np.asarray(samples)
    return (f"p50 {np.percentile(arr, 50):6.2f} ms | p95 {np.percentile(arr, 95):6.2f} ms | "
            f"p99 {np.percentile(arr, 99):6.2f} ms | max {arr.max():6.2f} ms")


def run_benchmark(checkpoint: Path, iterations: int, num_threads: int) -> None:
    """Time the model alone and the full per-frame pipeline on realistic windows."""
    engine = InferenceEngine(checkpoint, num_threads=num_threads)
    windows, _, _ = generate_windowed_dataset(iterations, window_size=engine.window_size,
                                              stride=1, config=SimulatorConfig(seed=123))
    smoother, alert = EMASmoother(0.35), HysteresisAlert()
    model_ms, pipeline_ms = [], []
    for window in windows:
        start = time.perf_counter()
        _, a, ms = engine.predict(window)
        alert.update(float(smoother(a)[2]), 0.6, 0.7, 3)
        pipeline_ms.append((time.perf_counter() - start) * 1e3)
        model_ms.append(ms)
    p99 = float(np.percentile(pipeline_ms, 99))
    print(f"\nInference benchmark — {iterations} windows of {engine.window_size} frames, "
          f"{torch.get_num_threads()} thread(s)")
    print(f"  Model forward : {_percentiles(model_ms)}")
    print(f"  Full pipeline : {_percentiles(pipeline_ms)}")
    verdict = "PASS" if p99 < LATENCY_BUDGET_MS else "FAIL"
    print(f"  Budget        : p99 {p99:.2f} ms vs {LATENCY_BUDGET_MS:.0f} ms → {verdict}")


def run_live(checkpoint: Path, duration: float, difficulty: str, seed: int) -> None:
    """Headless live run: stream, infer, and score against simulator ground truth."""
    service = LiveInferenceService(checkpoint, difficulty=difficulty, seed=seed)
    service.start()
    print(f"\nLive inference for {duration:.0f} s ({difficulty}) … warming up the 2 s window")
    t_end = time.time() + duration
    last_idx = -1
    while time.time() < t_end:
        time.sleep(1.0)
        latest = service.snapshot().latest
        if latest is not None and latest.frame_idx != last_idx:
            last_idx = latest.frame_idx
            print(f"  t={latest.sim_time:6.1f}s  pred={latest.gesture.name:<10} "
                  f"(true {Gesture(latest.true_gesture).name:<10})  "
                  f"alert={latest.alert.name:<10}  P(hes)={latest.anomaly_probs[2]:.2f}  "
                  f"e2e={latest.e2e_latency_ms:5.2f} ms")
    service.stop()
    snap = service.snapshot()
    if snap.error:
        print(f"Service error: {snap.error}")
    preds = snap.predictions
    if not preds:
        print("No predictions were produced.")
        return
    gesture_acc = np.mean([p.gesture == p.true_gesture for p in preds])
    print(f"\nPredictions: {len(preds)} | stream {snap.stream_hz:.1f} Hz | "
          f"stale frames skipped: {snap.frames_skipped}")
    print(f"Live gesture accuracy (smoothed): {100 * gesture_acc:.1f}%")
    print(f"Model latency : {_percentiles([p.model_latency_ms for p in preds])}")
    print(f"End-to-end    : {_percentiles([p.e2e_latency_ms for p in preds])}")
    print(f"Alert episodes: {len(snap.alert_events)} → " + ", ".join(
        f"{e.level.name}@{e.start:.1f}s" for e in snap.alert_events[-8:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-latency surgical inference engine")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--benchmark", action="store_true", help="measure latency")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--live", action="store_true", help="headless live stream demo")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS),
                        default="Standard case")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found at {args.checkpoint}. Run `python train.py`.")
    if not (args.benchmark or args.live):
        args.benchmark = True
    if args.benchmark:
        run_benchmark(args.checkpoint, args.iterations, args.threads)
    if args.live:
        run_live(args.checkpoint, args.duration, args.difficulty, args.seed)


if __name__ == "__main__":
    main()
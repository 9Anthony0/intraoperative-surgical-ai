#!/usr/bin/env python3
"""
simulator.py — High-Frequency 8-DOF Surgical Kinematic Telemetry Generator.

Part of the *Intraoperative Surgical Phase & Gesture Recognition Engine*.

Architecture overview
---------------------
The simulator is split into four layers, each independently testable:

1. **Math primitives** — quaternion algebra, SLERP, and minimum-jerk profiles.
2. **Gesture motion models** (``IdleModel``, ``SuturingModel``, ``DissectionModel``) —
   generators that map a *gesture-local clock* ``tau`` to a pose offset relative to the
   pose at which the gesture began (its "anchor").
3. **Anomaly injectors** (``inject_tremor``, ``inject_hesitation``) — pure functions that
   return a ``Perturbation`` layered *on top of* the clean gesture pose. Because anomalies
   are an independent layer, they co-occur with any base gesture, which is exactly the
   structure the multi-task (gesture + anomaly) model is designed to learn.
4. **Streaming** (``TelemetryRingBuffer``, ``AsyncTelemetryStreamer``) — a drift-free
   30 Hz asyncio producer with a bounded, drop-oldest queue and a preallocated ring buffer.

Key modeling decision: hesitation as *time dilation*
----------------------------------------------------
A micro-hesitation is not "the arm stops and teleports". It is the surgeon's motor plan
*slowing down mid-gesture* and then resuming exactly where it left off. We model it by
scaling the rate at which the gesture-local clock ``tau`` advances (down to ~0-12% speed),
wrapped in a smooth envelope. The pose therefore stays on the gesture's trajectory while its
velocity collapses. That is harder (and more realistic) for a classifier than a pose freeze,
because the network must reason about velocity *relative to the surrounding motion context*,
which is what the self-attention layer in ``model.py`` is there for.

Telemetry schema (9 model features)
-----------------------------------
``[x, y, z, qw, qx, qy, qz, grip, velocity]`` — 8 pose DOF plus derived tip speed.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import IntEnum
from typing import AsyncIterator, ClassVar, Final

import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------------------
# Constants & type aliases
# --------------------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

DEFAULT_SAMPLE_RATE_HZ: Final[float] = 30.0
NUM_FEATURES: Final[int] = 9
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "x", "y", "z", "qw", "qx", "qy", "qz", "grip", "velocity",
)

#: Neutral instrument-tip position in the robot workspace frame (mm).
HOME_POSITION: Final[FloatArray] = np.array([0.0, 0.0, 100.0])
#: Identity rotation quaternion in (w, x, y, z) convention.
IDENTITY_QUAT: Final[FloatArray] = np.array([1.0, 0.0, 0.0, 0.0])
#: Mechanical limit of the jaw opening (rad).
GRIP_MAX_RAD: Final[float] = 1.2


class Gesture(IntEnum):
    """Base surgical gestures (label space of the model's gesture head)."""

    IDLE = 0
    SUTURING = 1
    DISSECTION = 2


class Anomaly(IntEnum):
    """Motor anomalies (label space of the model's anomaly head)."""

    NORMAL = 0
    TREMOR = 1
    HESITATION = 2


# --------------------------------------------------------------------------------------
# Layer 1: math primitives
# --------------------------------------------------------------------------------------


def minimum_jerk(tau: float) -> float:
    """Return the normalized minimum-jerk position profile ``s(tau)`` for ``tau`` in [0, 1].

    Human reaching movements closely follow the minimum-jerk model (Flash & Hogan, 1985):
    ``s = 10 tau^3 - 15 tau^4 + 6 tau^5``. It has zero velocity *and* zero acceleration at
    both endpoints, so chaining segments never produces velocity spikes that a model could
    exploit as a trivial "segment boundary" shortcut.
    """
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def quat_normalize(q: FloatArray) -> FloatArray:
    """Project a 4-vector back onto the unit-quaternion manifold."""
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else IDENTITY_QUAT.copy()


def quat_multiply(q1: FloatArray, q2: FloatArray) -> FloatArray:
    """Hamilton product ``q1 ⊗ q2`` in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: FloatArray) -> FloatArray:
    """Conjugate (= inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis: FloatArray, angle: float) -> FloatArray:
    """Build a unit quaternion rotating by ``angle`` radians about ``axis``."""
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], np.sin(half) * axis))


def quat_slerp(q0: FloatArray, q1: FloatArray, s: float) -> FloatArray:
    """Spherical linear interpolation between unit quaternions (shortest path)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # q and -q encode the same rotation; take the short arc
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: fall back to normalized lerp for stability
        return quat_normalize(q0 + s * (q1 - q0))
    theta = np.arccos(dot) * s
    ortho = quat_normalize(q1 - q0 * dot)
    return q0 * np.cos(theta) + ortho * np.sin(theta)


def _random_unit_vector(rng: np.random.Generator, planar: bool = False) -> FloatArray:
    """Sample a direction uniformly on the sphere (or on the xy unit circle)."""
    if planar:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0.0])
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------------------
# Layer 2: gesture motion models
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class GesturePose:
    """Clean gesture output, expressed relative to the gesture's anchor pose."""

    offset: FloatArray    # (3,) translation offset from anchor position, mm
    rotation: FloatArray  # (4,) rotation applied after the anchor orientation
    grip: float           # target jaw angle, rad (low-pass filtered downstream)


class MotionModel(ABC):
    """Abstract gesture generator driven by a gesture-local clock ``tau``.

    Contract: ``pose(0, ·)`` returns (approximately) zero offset and identity rotation, so a
    new segment always starts exactly where the previous one ended (C0 continuity).
    Structured gestures also end on a cycle boundary where minimum-jerk velocity is zero.
    """

    gesture: ClassVar[Gesture]

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    @property
    @abstractmethod
    def duration(self) -> float:
        """Gesture-local duration in seconds (``tau`` at which the segment ends)."""

    @abstractmethod
    def pose(self, tau: float, dt: float) -> GesturePose:
        """Return the clean pose at local time ``tau``; ``dt`` is the effective step."""


class IdleModel(MotionModel):
    """Idle / repositioning: slow return toward neutral plus Ornstein-Uhlenbeck drift.

    Between tasks surgeons typically retract toward a neutral working position. The homing
    term also keeps long simulations from random-walking out of the workspace. The OU
    process (mean-reverting Brownian motion) mimics the small, aimless postural drift of a
    hand that is holding but not executing.
    """

    gesture = Gesture.IDLE
    GRIP_REST: ClassVar[float] = 0.35

    def __init__(
        self,
        rng: np.random.Generator,
        anchor_position: FloatArray,
        anchor_orientation: FloatArray,
    ) -> None:
        super().__init__(rng)
        self._duration = float(rng.uniform(1.5, 4.0))
        self._return_time = min(1.2, 0.6 * self._duration)
        self._home_delta = 0.6 * (HOME_POSITION - anchor_position)
        # Rotation that would take the anchor orientation back to neutral.
        self._to_neutral = quat_conjugate(anchor_orientation)
        self._drift = np.zeros(3)
        self._theta = 1.5   # OU mean-reversion rate, 1/s
        self._sigma = 0.6   # OU diffusion, mm/sqrt(s)

    @property
    def duration(self) -> float:
        return self._duration

    def pose(self, tau: float, dt: float) -> GesturePose:
        # Euler-Maruyama step of the OU process.
        self._drift += (-self._theta * self._drift * dt
                        + self._sigma * np.sqrt(max(dt, 0.0)) * self._rng.standard_normal(3))
        s = minimum_jerk(tau / self._return_time)
        rotation = quat_slerp(IDENTITY_QUAT, self._to_neutral, 0.5 * s)
        return GesturePose(s * self._home_delta + self._drift, rotation, self.GRIP_REST)


class SuturingModel(MotionModel):
    """Needle-driving cycles: circular arc bite → pull-through → release & reposition.

    Geometry: a curved needle of radius ``R`` is driven by *wrist rotation*, so the tip
    traces a semicircle in the plane spanned by ``u`` (across the wound) and ``w`` (into
    tissue), while the instrument rotates by the same angle about the plane normal
    ``n = u × w``. Rodrigues' formula confirms the pose and position are mutually consistent:
    rotating ``-u`` by ``theta`` about ``n`` yields ``-(cos θ u + sin θ w)``.

    Each cycle advances one stitch spacing along the wound direction ``v``.
    """

    gesture = Gesture.SUTURING
    GRIP_CLOSED: ClassVar[float] = 0.05
    GRIP_OPEN: ClassVar[float] = 0.65
    DRIVE_END: ClassVar[float] = 0.55   # fraction of the cycle spent driving the needle
    PULL_END: ClassVar[float] = 0.80    # fraction at which pull-through completes

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._radius = float(rng.uniform(6.0, 9.0))      # needle radius, mm
        self._period = float(rng.uniform(2.4, 3.6))      # s per stitch
        self._n_cycles = int(rng.integers(2, 5))
        self._spacing = float(rng.uniform(3.0, 5.0))     # stitch spacing, mm
        self._lift = float(rng.uniform(4.0, 8.0))        # pull-through height, mm
        heading = rng.uniform(0.0, 2.0 * np.pi)
        self._v = np.array([np.cos(heading), np.sin(heading), 0.0])   # along wound
        self._u = np.array([-np.sin(heading), np.cos(heading), 0.0])  # across wound
        self._w = np.array([0.0, 0.0, -1.0])                          # into tissue
        self._needle_axis = np.cross(self._u, self._w)
        self._tilt = quat_from_axis_angle(
            _random_unit_vector(rng, planar=True), float(rng.uniform(0.2, 0.5)))

    @property
    def duration(self) -> float:
        return self._period * self._n_cycles

    def pose(self, tau: float, dt: float) -> GesturePose:
        cycle, c = divmod(tau / self._period, 1.0)
        base = cycle * self._spacing * self._v
        r, u, w = self._radius, self._u, self._w
        drive_end = 2.0 * r * u
        lift_end = drive_end - self._lift * w
        next_start = self._spacing * self._v

        if c < self.DRIVE_END:            # 1) needle bite: semicircular arc via wrist roll
            theta = np.pi * minimum_jerk(c / self.DRIVE_END)
            offset = r * (1.0 - np.cos(theta)) * u - r * np.sin(theta) * w
            angle, grip = theta, self.GRIP_CLOSED
        elif c < self.PULL_END:           # 2) pull-through: lift while unwinding the wrist
            s = minimum_jerk((c - self.DRIVE_END) / (self.PULL_END - self.DRIVE_END))
            offset = drive_end + s * (lift_end - drive_end)
            angle, grip = np.pi * (1.0 - s), self.GRIP_CLOSED
        else:                             # 3) release and reposition for the next bite
            s = minimum_jerk((c - self.PULL_END) / (1.0 - self.PULL_END))
            offset = lift_end + s * (next_start - lift_end)
            angle, grip = 0.0, self.GRIP_OPEN

        tilt = quat_slerp(IDENTITY_QUAT, self._tilt, minimum_jerk(tau / 0.5))
        rotation = quat_multiply(tilt, quat_from_axis_angle(self._needle_axis, angle))
        return GesturePose(base + offset, rotation, grip)


class DissectionModel(MotionModel):
    """Blunt dissection cycles: push closed → spread jaws → retract, with slow advance.

    The spreading grip pattern (closed → wide → closed) is the dominant discriminative cue
    versus suturing, whose grip stays closed for most of the cycle. A slow lateral sweep
    and a small instrument twist add realistic variability.
    """

    gesture = Gesture.DISSECTION
    GRIP_CLOSED: ClassVar[float] = 0.05
    GRIP_SPREAD: ClassVar[float] = 0.9
    PUSH_END: ClassVar[float] = 0.4
    SPREAD_END: ClassVar[float] = 0.6

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._period = float(rng.uniform(1.0, 1.6))
        self._n_cycles = int(rng.integers(4, 9))
        self._stroke = float(rng.uniform(3.0, 6.0))     # push depth, mm
        self._advance = float(rng.uniform(0.6, 1.5))    # net progress per cycle, mm
        heading = rng.uniform(0.0, 2.0 * np.pi)
        d = np.array([np.cos(heading), np.sin(heading), -0.3])
        self._d = d / np.linalg.norm(d)                 # push direction (slightly downward)
        self._lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        self._sweep_amp = float(rng.uniform(1.0, 3.0))
        self._sweep_period = self._period * float(rng.uniform(4.0, 6.0))
        self._twist_amp = float(rng.uniform(0.1, 0.25))
        self._tilt = quat_from_axis_angle(
            _random_unit_vector(rng, planar=True), float(rng.uniform(0.1, 0.3)))

    @property
    def duration(self) -> float:
        return self._period * self._n_cycles

    def pose(self, tau: float, dt: float) -> GesturePose:
        cycle, c = divmod(tau / self._period, 1.0)
        closed, spread, stroke = self.GRIP_CLOSED, self.GRIP_SPREAD, self._stroke

        if c < self.PUSH_END:             # 1) push into the tissue plane, jaws closed
            along, grip = stroke * minimum_jerk(c / self.PUSH_END), closed
        elif c < self.SPREAD_END:         # 2) spread the jaws in place
            s = minimum_jerk((c - self.PUSH_END) / (self.SPREAD_END - self.PUSH_END))
            along, grip = stroke, closed + s * (spread - closed)
        else:                             # 3) retract (net advance remains), jaws close
            s = minimum_jerk((c - self.SPREAD_END) / (1.0 - self.SPREAD_END))
            along = stroke - s * (stroke - self._advance)
            grip = spread + s * (closed - spread)

        sweep = self._sweep_amp * np.sin(2.0 * np.pi * tau / self._sweep_period)
        offset = (cycle * self._advance + along) * self._d + sweep * self._lateral
        twist = quat_from_axis_angle(self._d, self._twist_amp * np.sin(2.0 * np.pi * c))
        tilt = quat_slerp(IDENTITY_QUAT, self._tilt, minimum_jerk(tau / 0.5))
        return GesturePose(offset, quat_multiply(tilt, twist), grip)


# --------------------------------------------------------------------------------------
# Layer 3: anomaly events and explicit injection functions
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Perturbation:
    """Additive anomaly layer applied on top of the clean gesture pose."""

    time_scale: float = 1.0  # multiplier on the gesture clock (hesitation < 1)
    position_offset: FloatArray = field(default_factory=lambda: np.zeros(3))
    rotation: FloatArray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    grip_offset: float = 0.0
    intensity: float = 0.0   # envelope value in [0, 1]; drives ground-truth labeling


@dataclass(slots=True)
class AnomalyEvent:
    """Base class: a time-bounded anomaly with a raised-cosine onset/offset envelope."""

    kind: ClassVar[Anomaly]
    start: float
    duration: float
    ramp: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def envelope(self, t: float) -> float:
        """Smooth [0, 1] activation. Raised-cosine ramps avoid step artifacts in the data."""
        local = t - self.start
        if local < 0.0 or local > self.duration:
            return 0.0
        ramp = min(self.ramp, 0.5 * self.duration)
        if ramp <= 0.0:
            return 1.0
        if local < ramp:
            return 0.5 * (1.0 - np.cos(np.pi * local / ramp))
        if local > self.duration - ramp:
            return 0.5 * (1.0 - np.cos(np.pi * (self.duration - local) / ramp))
        return 1.0

    def truncate(self, t: float) -> None:
        """End the event at time ``t`` (used when the context makes it meaningless)."""
        self.duration = max(0.0, t - self.start)


@dataclass(slots=True)
class TremorEvent(AnomalyEvent):
    """Fatigue-enhanced physiological tremor (6-12 Hz, sub-millimeter amplitude).

    Sampling note: at 30 Hz the Nyquist limit is 15 Hz, so tremor up to 12 Hz is captured
    without aliasing. That constraint is one reason the stream runs at 30 Hz, not 10 Hz.
    """

    kind: ClassVar[Anomaly] = Anomaly.TREMOR
    freqs: FloatArray = field(default_factory=lambda: np.array([9.0]))     # (k,) Hz
    amps: FloatArray = field(default_factory=lambda: np.full(3, 0.4))      # (3,) mm per axis
    phases: FloatArray = field(default_factory=lambda: np.zeros((1, 3)))   # (k, 3) rad
    rot_axis: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    rot_amp: float = 0.01        # rad
    am_freq: float = 0.7         # slow amplitude modulation (tremor waxes and wanes), Hz


@dataclass(slots=True)
class HesitationEvent(AnomalyEvent):
    """Micro-hesitation: gesture clock slows sharply, with small corrective search motions."""

    kind: ClassVar[Anomaly] = Anomaly.HESITATION
    min_rate: float = 0.05           # residual gesture speed at full hesitation (0-1)
    jitter_freq: float = 2.0         # Hz, low-frequency "searching" corrections
    jitter_amp: float = 0.25         # mm
    jitter_dir: FloatArray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    grip_jitter_freq: float = 2.5    # Hz, indecisive grip micro-adjustments
    grip_jitter_amp: float = 0.05    # rad


def sample_tremor_event(rng: np.random.Generator, start: float) -> TremorEvent:
    """Draw a randomized tremor episode starting at simulation time ``start``."""
    f0 = rng.uniform(6.0, 12.0)
    freqs = np.array([f0, min(f0 * rng.uniform(1.05, 1.2), 13.5)])
    return TremorEvent(
        start=start,
        duration=float(rng.uniform(1.0, 3.0)),
        ramp=0.3,
        freqs=freqs,
        amps=rng.uniform(0.2, 0.8, size=3),
        phases=rng.uniform(0.0, 2.0 * np.pi, size=(freqs.size, 3)),
        rot_axis=_random_unit_vector(rng),
        rot_amp=float(rng.uniform(0.005, 0.02)),
        am_freq=float(rng.uniform(0.4, 1.0)),
    )


def sample_hesitation_event(rng: np.random.Generator, start: float) -> HesitationEvent:
    """Draw a randomized micro-hesitation starting at simulation time ``start``."""
    return HesitationEvent(
        start=start,
        duration=float(rng.uniform(0.6, 2.0)),
        ramp=float(rng.uniform(0.1, 0.2)),   # sharp onset: hesitations begin abruptly
        min_rate=float(rng.uniform(0.0, 0.12)),
        jitter_freq=float(rng.uniform(1.0, 3.0)),
        jitter_amp=float(rng.uniform(0.1, 0.4)),
        jitter_dir=_random_unit_vector(rng),
        grip_jitter_freq=float(rng.uniform(1.5, 4.0)),
        grip_jitter_amp=float(rng.uniform(0.02, 0.08)),
    )


def inject_tremor(event: TremorEvent, t: float) -> Perturbation:
    """Compute the tremor perturbation at time ``t``.

    Translation: per-axis sum of two detuned sinusoids (a narrow-band tremor peak rather
    than a pure tone), amplitude-modulated and gated by the event envelope. Orientation:
    a small oscillatory rotation about a random axis at the dominant frequency. The gesture
    clock is untouched, so the underlying task continues at normal speed. That is exactly
    what makes tremor *co-occur* with a gesture instead of replacing it.
    """
    env = event.envelope(t)
    if env <= 0.0:
        return Perturbation()
    local = t - event.start
    am = 1.0 + 0.3 * np.sin(2.0 * np.pi * event.am_freq * local)
    osc = np.sin(2.0 * np.pi * event.freqs[:, None] * local + event.phases).mean(axis=0)
    angle = env * am * event.rot_amp * np.sin(
        2.0 * np.pi * event.freqs[0] * local + event.phases[0, 0])
    return Perturbation(
        position_offset=env * am * event.amps * osc,
        rotation=quat_from_axis_angle(event.rot_axis, angle),
        intensity=env,
    )


def inject_hesitation(event: HesitationEvent, t: float) -> Perturbation:
    """Compute the micro-hesitation perturbation at time ``t``.

    The core effect is ``time_scale``: the gesture clock slows to ``min_rate`` of normal
    speed, so the tool *stays on its planned trajectory* while velocity collapses. Small
    low-frequency corrective motions and grip micro-oscillations are layered on top, the
    kinematic signature of a surgeon who has paused to re-plan rather than simply stopped.
    """
    env = event.envelope(t)
    if env <= 0.0:
        return Perturbation()
    local = t - event.start
    jitter = event.jitter_amp * np.sin(2.0 * np.pi * event.jitter_freq * local)
    grip = event.grip_jitter_amp * np.sin(2.0 * np.pi * event.grip_jitter_freq * local)
    return Perturbation(
        time_scale=1.0 - (1.0 - event.min_rate) * env,
        position_offset=env * jitter * event.jitter_dir,
        grip_offset=env * grip,
        intensity=env,
    )


# --------------------------------------------------------------------------------------
# Simulator core
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryFrame:
    """One sample of robot telemetry plus ground-truth labels."""

    frame_idx: int
    sim_time: float          # simulated time since reset, s
    wall_time: float         # time.perf_counter() at emission (for latency accounting)
    features: Float32Array   # (9,) [x, y, z, qw, qx, qy, qz, grip, velocity]
    gesture: Gesture
    anomaly: Anomaly

    @property
    def position(self) -> Float32Array:
        return self.features[0:3]

    @property
    def quaternion(self) -> Float32Array:
        return self.features[3:7]

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat, JSON-friendly representation (useful for logging and dashboards)."""
        record: dict[str, float | int | str] = {
            name: float(v) for name, v in zip(FEATURE_NAMES, self.features)
        }
        record.update(frame_idx=self.frame_idx, sim_time=self.sim_time,
                      gesture=self.gesture.name, anomaly=self.anomaly.name)
        return record


@dataclass(slots=True)
class SimulatorConfig:
    """Tunable simulation parameters. Defaults approximate a 30 Hz RAS telemetry bus."""

    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    seed: int | None = None
    position_noise_mm: float = 0.05     # encoder / kinematic-chain noise
    quat_noise: float = 0.002
    grip_noise_rad: float = 0.01
    grip_time_constant_s: float = 0.08  # jaw actuator lag
    velocity_smoothing: float = 0.5     # EMA alpha for derived speed
    enable_anomalies: bool = True
    anomaly_mean_interval_s: float = 5.0
    hesitation_probability: float = 0.5
    anomaly_label_threshold: float = 0.5


#: First-order Markov chain over gestures: current -> (next gestures, probabilities).
_TRANSITIONS: Final[dict[Gesture, tuple[tuple[Gesture, ...], tuple[float, ...]]]] = {
    Gesture.IDLE: ((Gesture.SUTURING, Gesture.DISSECTION), (0.5, 0.5)),
    Gesture.SUTURING: ((Gesture.IDLE, Gesture.DISSECTION), (0.6, 0.4)),
    Gesture.DISSECTION: ((Gesture.IDLE, Gesture.SUTURING), (0.5, 0.5)),
}


class KinematicSimulator:
    """Synchronous, deterministic (given a seed) 8-DOF kinematic telemetry generator.

    ``step()`` is pure CPU work with no sleeping, so the same class powers both the
    real-time stream (wrapped by ``AsyncTelemetryStreamer``) and fast offline dataset
    generation for ``train.py``. Training and inference therefore see *identical* signal
    statistics, which avoids a whole class of train/serve skew bugs.
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.dt = 1.0 / self.config.sample_rate_hz
        self._rng = np.random.default_rng(self.config.seed)
        self._grip_alpha = self.dt / (self.config.grip_time_constant_s + self.dt)
        self.reset()

    # ---- state management ------------------------------------------------------------

    def reset(self) -> None:
        """Return the arm to the home pose and clear all gesture and anomaly state."""
        self._t = 0.0
        self._frame_idx = 0
        self._clean_position = HOME_POSITION.copy()
        self._clean_orientation = IDENTITY_QUAT.copy()
        self._grip = IdleModel.GRIP_REST
        self._prev_noisy_position: FloatArray | None = None
        self._prev_quat = IDENTITY_QUAT.copy()
        self._velocity = 0.0
        self._active_anomaly: AnomalyEvent | None = None
        self._next_anomaly_time = self._sample_anomaly_gap()
        self._start_segment(Gesture.IDLE)

    @property
    def current_gesture(self) -> Gesture:
        return self._model.gesture

    def _start_segment(self, gesture: Gesture) -> None:
        """Begin a new gesture anchored at the current *clean* pose.

        Anchoring on the clean pose (not the anomaly-perturbed one) guarantees tremor or
        hesitation offsets are never "baked into" the trajectory after the event ends.
        """
        self._anchor_position = self._clean_position.copy()
        self._anchor_orientation = self._clean_orientation.copy()
        self._tau = 0.0
        if gesture is Gesture.IDLE:
            self._model: MotionModel = IdleModel(
                self._rng, self._anchor_position, self._anchor_orientation)
        elif gesture is Gesture.SUTURING:
            self._model = SuturingModel(self._rng)
        else:
            self._model = DissectionModel(self._rng)

    def _next_gesture(self) -> Gesture:
        choices, probs = _TRANSITIONS[self.current_gesture]
        return choices[int(self._rng.choice(len(choices), p=probs))]

    def _sample_anomaly_gap(self) -> float:
        """Inter-arrival time of anomalies: Poisson process with a 1 s refractory floor."""
        return self._t + 1.0 + float(self._rng.exponential(self.config.anomaly_mean_interval_s))

    def _update_anomalies(self, t: float) -> None:
        """Retire finished events and schedule new ones."""
        event = self._active_anomaly
        if event is not None and t >= event.end:
            self._active_anomaly = None
            self._next_anomaly_time = self._sample_anomaly_gap()

        if (self._active_anomaly is None and self.config.enable_anomalies
                and t >= self._next_anomaly_time):
            if self._rng.random() < self.config.hesitation_probability:
                if self.current_gesture is Gesture.IDLE:
                    # A hesitation needs an ongoing motor plan to interrupt; retry shortly.
                    self._next_anomaly_time = t + 0.5
                    return
                self._active_anomaly = sample_hesitation_event(self._rng, t)
            else:
                self._active_anomaly = sample_tremor_event(self._rng, t)

    def _perturbation(self, t: float) -> Perturbation:
        """Dispatch to the explicit injection function for the active anomaly."""
        event = self._active_anomaly
        if isinstance(event, TremorEvent):
            return inject_tremor(event, t)
        if isinstance(event, HesitationEvent):
            return inject_hesitation(event, t)
        return Perturbation()

    # ---- main step -------------------------------------------------------------------

    def step(self) -> TelemetryFrame:
        """Advance the simulation by one sample period and return the emitted frame."""
        cfg, rng, t = self.config, self._rng, self._t
        self._update_anomalies(t)
        pert = self._perturbation(t)

        # 1) Advance the gesture-local clock (slowed during hesitation).
        effective_dt = self.dt * pert.time_scale
        self._tau += effective_dt
        if self._tau >= self._model.duration:
            self._start_segment(self._next_gesture())
            if (isinstance(self._active_anomaly, HesitationEvent)
                    and self.current_gesture is Gesture.IDLE):
                self._active_anomaly.truncate(t)
                pert = Perturbation()

        # 2) Clean pose = anchor ⊕ gesture offset.
        gp = self._model.pose(self._tau, effective_dt)
        self._clean_position = self._anchor_position + gp.offset
        self._clean_orientation = quat_normalize(
            quat_multiply(self._anchor_orientation, gp.rotation))

        # 3) Anomaly layer.
        position = self._clean_position + pert.position_offset
        orientation = quat_multiply(self._clean_orientation, pert.rotation)
        self._grip += self._grip_alpha * (gp.grip + pert.grip_offset - self._grip)

        # 4) Sensor noise.
        noisy_pos = position + rng.normal(0.0, cfg.position_noise_mm, 3)
        noisy_quat = quat_normalize(orientation + rng.normal(0.0, cfg.quat_noise, 4))
        if np.dot(noisy_quat, self._prev_quat) < 0.0:
            # q and -q are the same rotation. Enforcing hemisphere continuity prevents
            # sign flips that look like huge discontinuities to a neural network.
            noisy_quat = -noisy_quat
        self._prev_quat = noisy_quat
        noisy_grip = float(np.clip(self._grip + rng.normal(0.0, cfg.grip_noise_rad),
                                   0.0, GRIP_MAX_RAD))

        # 5) Derived feature: tip speed from finite differences, EMA-smoothed.
        if self._prev_noisy_position is not None:
            raw_speed = float(np.linalg.norm(noisy_pos - self._prev_noisy_position)) / self.dt
            self._velocity += cfg.velocity_smoothing * (raw_speed - self._velocity)
        self._prev_noisy_position = noisy_pos

        # 6) Ground truth: label only where the anomaly is materially present.
        anomaly = Anomaly.NORMAL
        if self._active_anomaly is not None and pert.intensity >= cfg.anomaly_label_threshold:
            anomaly = self._active_anomaly.kind

        features = np.concatenate(
            (noisy_pos, noisy_quat, (noisy_grip, self._velocity))).astype(np.float32)
        frame = TelemetryFrame(self._frame_idx, t, time.perf_counter(), features,
                               self.current_gesture, anomaly)
        self._t += self.dt
        self._frame_idx += 1
        return frame

    # ---- offline generation ----------------------------------------------------------

    def generate_sequence(self, num_frames: int) -> tuple[Float32Array, IntArray, IntArray]:
        """Generate ``num_frames`` contiguous frames as arrays (no real-time pacing).

        Returns:
            features ``(N, 9)`` float32, gesture labels ``(N,)``, anomaly labels ``(N,)``.
        """
        features = np.empty((num_frames, NUM_FEATURES), dtype=np.float32)
        gestures = np.empty(num_frames, dtype=np.int64)
        anomalies = np.empty(num_frames, dtype=np.int64)
        for i in range(num_frames):
            frame = self.step()
            features[i], gestures[i], anomalies[i] = frame.features, frame.gesture, frame.anomaly
        return features, gestures, anomalies


def generate_windowed_dataset(
    num_windows: int,
    window_size: int = 60,
    stride: int = 5,
    config: SimulatorConfig | None = None,
) -> tuple[Float32Array, IntArray, IntArray]:
    """Build a sliding-window dataset for ``train.py``.

    Each window is labeled with the labels of its **last** frame. That matches causal
    streaming inference exactly: at runtime the model sees the past 2 s and must classify
    *now*. Overlapping windows (stride < window) multiply the effective sample count cheaply.

    Returns:
        X ``(num_windows, window_size, 9)``, y_gesture ``(num_windows,)``,
        y_anomaly ``(num_windows,)``.
    """
    total = window_size + (num_windows - 1) * stride
    features, gestures, anomalies = KinematicSimulator(config).generate_sequence(total)
    windows = np.lib.stride_tricks.sliding_window_view(
        features, (window_size, NUM_FEATURES))[::stride, 0]
    last = np.arange(num_windows) * stride + window_size - 1
    return np.ascontiguousarray(windows), gestures[last], anomalies[last]


# --------------------------------------------------------------------------------------
# Layer 4: streaming infrastructure
# --------------------------------------------------------------------------------------


class TelemetryRingBuffer:
    """Fixed-capacity, preallocated circular buffer of recent telemetry.

    Preallocation matters for latency: appending to a Python list or deque of arrays
    triggers allocations and GC pressure on the hot path, while writing into a fixed NumPy
    block is O(1) with no allocation. Reads return chronologically ordered *copies*, so
    consumers (inference, dashboard) never observe a half-written window. A lock makes it
    safe to share between the asyncio thread and Streamlit's script thread.
    """

    def __init__(self, capacity: int, num_features: int = NUM_FEATURES) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._features = np.zeros((capacity, num_features), dtype=np.float32)
        self._gestures = np.zeros(capacity, dtype=np.int64)
        self._anomalies = np.zeros(capacity, dtype=np.int64)
        self._times = np.zeros(capacity, dtype=np.float64)
        self._head = 0   # index of the next write
        self._size = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def append(self, frame: TelemetryFrame) -> None:
        """Write one frame, overwriting the oldest when full. O(1), allocation-free."""
        with self._lock:
            i = self._head
            self._features[i] = frame.features
            self._gestures[i] = frame.gesture
            self._anomalies[i] = frame.anomaly
            self._times[i] = frame.sim_time
            self._head = (i + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def latest_window(
        self, n: int | None = None,
    ) -> tuple[Float32Array, IntArray, IntArray, FloatArray]:
        """Return the most recent ``n`` frames, oldest first.

        Returns:
            features ``(n, 9)``, gestures ``(n,)``, anomalies ``(n,)``, sim times ``(n,)``.
        """
        with self._lock:
            n = self._size if n is None else n
            if n > self._size:
                raise ValueError(f"Requested {n} frames but only {self._size} buffered")
            idx = (self._head - n + np.arange(n)) % self.capacity
            return (self._features[idx], self._gestures[idx],
                    self._anomalies[idx], self._times[idx])


@dataclass(slots=True)
class StreamStats:
    """Producer-side timing health metrics."""

    frames_emitted: int = 0
    frames_dropped: int = 0
    deadline_overruns: int = 0
    _jitter_sum_s: float = 0.0
    max_jitter_s: float = 0.0

    def record_jitter(self, lateness_s: float) -> None:
        self._jitter_sum_s += lateness_s
        self.max_jitter_s = max(self.max_jitter_s, lateness_s)

    @property
    def mean_jitter_ms(self) -> float:
        return 1e3 * self._jitter_sum_s / max(self.frames_emitted, 1)


class AsyncTelemetryStreamer:
    """Real-time ~30 Hz telemetry producer built on ``asyncio``.

    Design choices:

    * **Absolute-deadline scheduling.** Each tick targets ``t0 + k * period`` rather than
      sleeping a fixed ``period`` after the last one, so per-tick jitter never accumulates
      into rate drift. If the loop falls more than one period behind, it resynchronizes
      instead of bursting catch-up frames (stale frames are worthless in the OR).
    * **Bounded queue with drop-oldest backpressure.** A safety monitor should always act
      on the *freshest* data. If the consumer stalls, old frames are discarded and counted
      rather than letting latency grow without bound.
    * **Ring buffer side-channel.** Every frame is also written to a ``TelemetryRingBuffer``
      so the dashboard can pull a history window without competing for queue items.

    Platform note: Windows' default timer resolution (~15.6 ms) adds per-tick jitter, but
    absolute scheduling still holds the *average* rate at 30 Hz.
    """

    def __init__(
        self,
        simulator: KinematicSimulator,
        queue_maxsize: int = 64,
        buffer_capacity: int = 300,
    ) -> None:
        self.simulator = simulator
        self.period = simulator.dt
        self.queue_maxsize = queue_maxsize
        self.ring_buffer = TelemetryRingBuffer(buffer_capacity)
        self.stats = StreamStats()
        self._queue: asyncio.Queue[TelemetryFrame] | None = None
        self._stopped: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create loop-bound primitives and launch the producer task."""
        if self._task is not None and not self._task.done():
            return
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)
        self._stopped = asyncio.Event()
        self._task = asyncio.create_task(self._produce(), name="telemetry-producer")

    async def stop(self) -> None:
        """Signal the producer to exit and wait for it (cancelling if it hangs)."""
        if self._stopped is not None:
            self._stopped.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def __aenter__(self) -> AsyncTelemetryStreamer:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    def _publish(self, frame: TelemetryFrame) -> None:
        """Non-blocking enqueue with drop-oldest policy."""
        assert self._queue is not None
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.stats.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(frame)

    async def _produce(self) -> None:
        assert self._stopped is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while not self._stopped.is_set():
            delay = deadline - loop.time()
            if delay > 0.0:
                await asyncio.sleep(delay)
            lateness = loop.time() - deadline
            if lateness > self.period:
                # More than one tick behind: resync rather than burst stale frames.
                self.stats.deadline_overruns += 1
                deadline = loop.time()
                lateness = 0.0
            self.stats.record_jitter(max(lateness, 0.0))

            frame = self.simulator.step()
            self.ring_buffer.append(frame)
            self._publish(frame)
            self.stats.frames_emitted += 1
            deadline += self.period

    async def frames(self) -> AsyncIterator[TelemetryFrame]:
        """Async iterator over live frames; terminates cleanly after ``stop()``."""
        if self._queue is None or self._stopped is None:
            raise RuntimeError("Streamer not started; use 'async with streamer:' first")
        while True:
            if self._stopped.is_set() and self._queue.empty():
                return
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=5 * self.period)
            except asyncio.TimeoutError:
                continue
            yield frame


# --------------------------------------------------------------------------------------
# CLI smoke test:  python simulator.py
# --------------------------------------------------------------------------------------

_ANSI = {Anomaly.TREMOR: "\033[93m", Anomaly.HESITATION: "\033[91m"}  # yellow / red
_RESET = "\033[0m"


def _format_row(frame: TelemetryFrame, queue_latency_ms: float, color: bool) -> str:
    x, y, z, qw, qx, qy, qz, grip, vel = frame.features
    row = (f"{frame.frame_idx:>6} {frame.sim_time:>7.2f}s | "
           f"{x:>7.2f} {y:>7.2f} {z:>7.2f} | "
           f"{qw:>6.3f} {qx:>6.3f} {qy:>6.3f} {qz:>6.3f} | "
           f"{grip:>5.2f} {vel:>6.1f} | "
           f"{frame.gesture.name:<10} {frame.anomaly.name:<10} | {queue_latency_ms:>5.2f}")
    if color and frame.anomaly in _ANSI:
        return f"{_ANSI[frame.anomaly]}{row}{_RESET}"
    return row


async def _run_stream_demo(duration_s: float, seed: int | None,
                           print_every: int, color: bool) -> None:
    """Stream live telemetry, print a sample of frames, and report timing health."""
    sim = KinematicSimulator(SimulatorConfig(seed=seed))
    streamer = AsyncTelemetryStreamer(sim, buffer_capacity=int(10 * sim.config.sample_rate_hz))
    gesture_counts: Counter[Gesture] = Counter()
    anomaly_counts: Counter[Anomaly] = Counter()
    arrivals: list[float] = []
    latencies_ms: list[float] = []

    header = (f"{'frame':>6} {'t':>8} | {'x':>7} {'y':>7} {'z':>7} | "
              f"{'qw':>6} {'qx':>6} {'qy':>6} {'qz':>6} | {'grip':>5} {'vel':>6} | "
              f"{'gesture':<10} {'anomaly':<10} | {'q-lat':>5}")
    print(f"\nStreaming 8-DOF telemetry at {sim.config.sample_rate_hz:.0f} Hz "
          f"for {duration_s:.0f} s (units: mm, rad, mm/s, ms)\n")
    print(header)
    print("-" * len(header))

    async with streamer:
        async with aclosing(streamer.frames()) as frames:
            async for frame in frames:
                now = time.perf_counter()
                arrivals.append(now)
                latencies_ms.append(1e3 * (now - frame.wall_time))
                gesture_counts[frame.gesture] += 1
                anomaly_counts[frame.anomaly] += 1
                if frame.frame_idx % print_every == 0 or frame.anomaly is not Anomaly.NORMAL:
                    print(_format_row(frame, latencies_ms[-1], color))
                if now - arrivals[0] >= duration_s:
                    break

    n = len(arrivals)
    intervals_ms = 1e3 * np.diff(arrivals)
    window, *_ = streamer.ring_buffer.latest_window(min(60, len(streamer.ring_buffer)))
    quat_err = float(np.abs(np.linalg.norm(window[:, 3:7], axis=1) - 1.0).max())

    print("\n" + "=" * 60 + "\nSTREAM HEALTH SUMMARY\n" + "=" * 60)
    print(f"Frames received        : {n}")
    print(f"Effective rate         : {(n - 1) / (arrivals[-1] - arrivals[0]):.2f} Hz "
          f"(target {sim.config.sample_rate_hz:.0f} Hz)")
    print(f"Inter-arrival          : mean {intervals_ms.mean():.2f} ms, "
          f"std {intervals_ms.std():.2f} ms")
    print(f"Producer jitter        : mean {streamer.stats.mean_jitter_ms:.2f} ms, "
          f"max {1e3 * streamer.stats.max_jitter_s:.2f} ms")
    print(f"Queue latency          : p50 {np.percentile(latencies_ms, 50):.3f} ms, "
          f"p99 {np.percentile(latencies_ms, 99):.3f} ms")
    print(f"Dropped / overruns     : {streamer.stats.frames_dropped} / "
          f"{streamer.stats.deadline_overruns}")
    print(f"Ring buffer window     : {window.shape}, max |‖q‖-1| = {quat_err:.2e}")
    print("Gesture distribution   : " + ", ".join(
        f"{g.name} {100 * gesture_counts[g] / n:.0f}%" for g in Gesture))
    print("Anomaly distribution   : " + ", ".join(
        f"{a.name} {100 * anomaly_counts[a] / n:.0f}%" for a in Anomaly))


def _run_dataset_check(num_windows: int, seed: int | None) -> None:
    """Generate an offline windowed dataset and report shapes, balance, and throughput."""
    start = time.perf_counter()
    X, y_g, y_a = generate_windowed_dataset(num_windows, config=SimulatorConfig(seed=seed))
    elapsed = time.perf_counter() - start
    print(f"\nOffline dataset: X {X.shape} {X.dtype}, y_gesture {y_g.shape}, "
          f"y_anomaly {y_a.shape} in {elapsed:.2f} s")
    print("Gesture balance : " + ", ".join(
        f"{g.name} {100 * np.mean(y_g == g):.1f}%" for g in Gesture))
    print("Anomaly balance : " + ", ".join(
        f"{a.name} {100 * np.mean(y_a == a):.1f}%" for a in Anomaly))
    print(f"Finite values   : {bool(np.isfinite(X).all())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="8-DOF surgical telemetry simulator")
    parser.add_argument("--duration", type=float, default=10.0, help="stream length, s")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (reproducible runs)")
    parser.add_argument("--print-every", type=int, default=6,
                        help="print every Nth frame (anomalous frames always print)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--dataset-check", type=int, default=0, metavar="N",
                        help="also generate N offline training windows and report stats")
    args = parser.parse_args()

    asyncio.run(_run_stream_demo(args.duration, args.seed,
                                 max(1, args.print_every), not args.no_color))
    if args.dataset_check > 0:
        _run_dataset_check(args.dataset_check, args.seed)


if __name__ == "__main__":
    main()
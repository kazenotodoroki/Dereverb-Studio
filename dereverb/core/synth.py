"""Générateurs de signaux de synthèse : IR artificielles et matière sèche de
secours pour l'entraînement (quand aucun dossier de sons secs n'est fourni)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def synthetic_ir(sr: int, rt60: float = 1.5, predelay: float = 0.012,
                 seed: int = 0, length: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int((length or rt60 * 1.2) * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    # amortissement des aigus plus rapide (deux bandes)
    sos = butter(2, 4000, "low", fs=sr, output="sos")
    lowpart = sosfilt(sos, tail)
    hi = (tail - lowpart) * np.exp(-6.91 * t / (rt60 * 0.5))
    tail = lowpart + hi
    tail[: int(predelay * sr)] = 0
    h = 0.35 * tail / np.abs(tail).max()
    h[0] = 1.0
    for k in range(8):  # réflexions précoces
        d = int(rng.uniform(0.003, predelay) * sr)
        h[d] += rng.uniform(-0.5, 0.5)
    return h


def synthetic_dry(sr: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Matière sèche variée : notes harmoniques, percussions, bruit, transitoires."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n)
    t_cursor = 0
    while t_cursor < n:
        kind = rng.choice(["note", "perc", "noise", "chord", "click"], p=[.35, .25, .1, .2, .1])
        dur = int(rng.uniform(0.05, 1.2) * sr)
        dur = min(dur, n - t_cursor)
        t = np.arange(dur) / sr
        if kind in ("note", "chord"):
            f0s = [55 * 2 ** (rng.integers(0, 48) / 12)]
            if kind == "chord":
                f0s += [f0s[0] * r for r in (1.26, 1.5)]
            sig = np.zeros(dur)
            for f0 in f0s:
                for k in range(1, 16):
                    if f0 * k > sr * 0.45:
                        break
                    sig += np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k ** rng.uniform(0.8, 2)
            att = rng.uniform(0.002, 0.08)
            env = np.minimum(t / att, 1) * np.exp(-t * rng.uniform(1, 8))
        elif kind == "perc":
            sig = rng.standard_normal(dur)
            sig += np.sin(2 * np.pi * rng.uniform(40, 200) * t * np.exp(-t * 20)) * 3
            env = np.exp(-t * rng.uniform(10, 60))
        elif kind == "noise":
            sos = butter(2, sorted(rng.uniform(100, sr * 0.4, 2)), "band", fs=sr, output="sos")
            sig = sosfilt(sos, rng.standard_normal(dur))
            env = np.sin(np.pi * t / t[-1]) if dur > 1 else np.ones(dur)
        else:
            sig = np.zeros(dur)
            sig[: min(3, dur)] = rng.choice([-1, 1])
            env = np.ones(dur)
        seg = sig * env
        seg /= np.abs(seg).max() + 1e-9
        out[t_cursor: t_cursor + dur] += seg * 10 ** (rng.uniform(-24, 0) / 20)
        # silences ou chevauchements
        t_cursor += int(dur * rng.uniform(0.3, 1.5)) + int(rng.exponential(0.1) * sr)
    return out / (np.abs(out).max() + 1e-9) * 0.7

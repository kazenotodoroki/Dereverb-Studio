"""Utilitaires audio : E/S, rééchantillonnage, convolution FFT."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import fft as sfft
from scipy.signal import resample_poly

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".w64", ".caf"}


@dataclass
class Audio:
    """Signal multicanal (samples, channels) en float64."""
    data: np.ndarray
    sr: int
    path: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.data.shape[0]

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n / self.sr

    def mono(self) -> np.ndarray:
        return self.data.mean(axis=1)


def load(path: str | Path) -> Audio:
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    info = sf.info(str(path))
    return Audio(data=data, sr=int(sr), path=Path(path), meta={"subtype": info.subtype})


def save(path: str | Path, data: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Rééchantillonnage polyphase de haute qualité (axe 0)."""
    if sr_in == sr_out:
        return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=0,
                         window=("kaiser", 12.0))


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolution linéaire 1D complète via FFT."""
    n = len(x) + len(h) - 1
    nf = sfft.next_fast_len(n, real=True)
    return sfft.irfft(sfft.rfft(x, nf) * sfft.rfft(h, nf), nf)[:n]


def db(x: float, floor: float = -200.0) -> float:
    return float(max(20 * np.log10(max(abs(x), 1e-30)), floor))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-30))


def match_channels(ir: np.ndarray, channels: int) -> np.ndarray:
    """Adapte une IR (n, c) au nombre de canaux d'une piste."""
    if ir.shape[1] == channels:
        return ir
    if ir.shape[1] == 1:
        return np.repeat(ir, channels, axis=1)
    if channels == 1:
        return ir.mean(axis=1, keepdims=True)
    # plus de canaux dans l'IR que dans la piste : on prend les premiers
    return ir[:, :channels] if ir.shape[1] > channels else np.resize(ir, (ir.shape[0], channels))


def list_audio(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)

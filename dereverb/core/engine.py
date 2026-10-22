"""Orchestration : applique une méthode à une piste multicanal."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import Audio, match_channels, resample, save
from .inverse import InverseParams, apply_target, deconvolve, direct_part, reconvolve_residual_db
from .iterative import IterativeParams, refine
from .statistical import RoomParams
from .statistical import process as room_process

METHODS = {
    "inverse": "Inverse exacte",
    "iterative": "Itérative (L-BFGS)",
    "room": "Salle (réverb tardive + hybride)",
    "neural": "Neuronale (modèle entraîné)",
}


@dataclass
class ProcessSettings:
    method: str = "inverse"
    inverse: InverseParams = field(default_factory=InverseParams)
    iterative: IterativeParams = field(default_factory=IterativeParams)
    room: RoomParams = field(default_factory=RoomParams)
    target_ms: float = 0.0      # 0 = cible impulsion (IR exacte) ; >0 = garder le son direct de l'IR
    model_path: str | None = None
    output_dir: str | None = None
    suffix: str = "_dry"
    subtype: str = "FLOAT"


@dataclass
class ProcessResult:
    data: np.ndarray
    sr: int
    residual_db: list[float]
    peak: float
    metric: str = "Résidu de reconstruction"
    path: Path | None = None


_model_cache: dict = {}


def _get_model(path: str):
    from .neural import NeuralModel
    if path not in _model_cache:
        _model_cache.clear()
        _model_cache[path] = NeuralModel(path)
    return _model_cache[path]


def process(track: Audio, ir: Audio | None, s: ProcessSettings,
            progress: Callable[[float], None] = lambda f: None,
            cancelled: Callable[[], bool] = lambda: False) -> ProcessResult:
    y = track.data
    C = y.shape[1]
    model = None
    if s.method == "neural":
        if not s.model_path:
            raise ValueError("Aucun modèle neuronal sélectionné.")
        model = _get_model(s.model_path)
        h_all = resample(model.ir, model.cfg.sr, track.sr)[:, None]
        dry, wet = 0.0, 1.0
    else:
        if ir is None:
            raise ValueError("Aucune IR assignée à cette piste.")
        h_all = resample(ir.data, ir.sr, track.sr)
        dry, wet = s.inverse.dry, s.inverse.wet
    h_all = match_channels(h_all, C)

    if s.method == "room":
        rp = s.room
        if rp.crossover_hz < 0:   # automatique : cohérence mesurée sur plusieurs positions
            rp = RoomParams(**{**rp.__dict__, "crossover_hz": float(ir.meta.get("coherent_hz", 0.0))})
        out = room_process(y, h_all, track.sr, rp, progress=progress)
        change = [10 * np.log10(np.sum(out[:, c] ** 2) / (np.sum(y[:, c] ** 2) + 1e-30) + 1e-30)
                  for c in range(C)]
        return ProcessResult(out, track.sr, change, float(np.abs(out).max()), metric="Énergie après traitement")

    out = np.zeros_like(y)
    res = []
    for c in range(C):
        if cancelled():
            raise InterruptedError
        sub = lambda f, c=c: progress((c + f) / C)  # noqa: E731
        yc, hc = y[:, c], h_all[:, c]
        if s.method == "inverse":
            xc = deconvolve(yc, hc, track.sr, s.inverse)
        elif s.method == "iterative":
            xc = refine(yc, hc, track.sr, s.inverse, s.iterative, progress=sub, cancelled=cancelled)
        elif s.method == "neural":
            xc = model.process(yc, track.sr, progress=sub)
        else:
            raise ValueError(s.method)
        res.append(reconvolve_residual_db(yc, xc, hc, dry, wet))
        if s.target_ms > 0 and s.method != "neural":
            xc = apply_target(xc, direct_part(hc, track.sr, s.target_ms))
        out[:, c] = xc
        sub(1.0)
    return ProcessResult(out, track.sr, res, float(np.abs(out).max()))


def output_path(track: Audio, s: ProcessSettings) -> Path:
    src = track.path or Path("piste.wav")
    folder = Path(s.output_dir) if s.output_dir else src.parent
    return folder / f"{src.stem}{s.suffix}.wav"


def write(result: ProcessResult, path: Path, subtype: str) -> Path:
    save(path, result.data, result.sr, subtype=subtype)
    result.path = path
    return path

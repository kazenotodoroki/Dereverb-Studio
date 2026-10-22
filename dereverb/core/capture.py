"""Signal de capture (sweeps sinus exponentiels, méthode Farina) et extraction d'IR.

Le stimulus est composé de N sweeps identiques séparés par des silences.
On joue ce fichier à travers la réverbération (salle, plugin, hardware), on
l'enregistre, puis on le déconvolue : chaque sweep donne une IR, on les
moyenne pour gagner en rapport signal/bruit (+3 dB par doublement), et la
distorsion harmonique est rejetée en « temps négatif » avant chaque IR.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import fft as sfft


@dataclass
class SweepParams:
    sr: int = 48000
    f1: float = 20.0
    f2: float = 22000.0
    duration: float = 10.0      # durée d'un sweep (s)
    repeats: int = 3            # nombre de sweeps
    gap: float = 6.0            # silence après chaque sweep (s) >= RT60 attendu
    pre_silence: float = 1.0    # silence initial (s)
    level_db: float = -6.0      # niveau crête (dBFS)
    fade_in: float = 0.05
    fade_out: float = 0.005

    @property
    def L(self) -> float:
        return self.duration / np.log(self.f2 / self.f1)

    @property
    def period(self) -> int:
        return int(round((self.duration + self.gap) * self.sr))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "SweepParams":
        return cls(**json.loads(text))


def sweep(p: SweepParams) -> np.ndarray:
    """Un sweep sinus exponentiel avec fondus, amplitude 1."""
    f2 = min(p.f2, 0.49 * p.sr)
    n = int(round(p.duration * p.sr))
    t = np.arange(n) / p.sr
    L = p.duration / np.log(f2 / p.f1)
    s = np.sin(2 * np.pi * p.f1 * L * (np.exp(t / L) - 1.0))
    fi, fo = int(p.fade_in * p.sr), int(p.fade_out * p.sr)
    if fi > 0:
        s[:fi] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(fi) / fi)
    if fo > 0:
        s[-fo:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fo) / fo)
    return s


def stimulus(p: SweepParams) -> np.ndarray:
    """Fichier complet à jouer : silence + N × (sweep + silence)."""
    s = sweep(p) * 10 ** (p.level_db / 20)
    gap = np.zeros(int(round(p.gap * p.sr)))
    parts = [np.zeros(int(round(p.pre_silence * p.sr)))]
    for _ in range(p.repeats):
        parts += [s, gap]
    return np.concatenate(parts)


def write_stimulus(path: str | Path, p: SweepParams) -> Path:
    from .audio import save
    path = Path(path)
    save(path, stimulus(p), p.sr, subtype="PCM_24")
    path.with_suffix(".sweep.json").write_text(p.to_json())
    return path


def _deconvolve_sweep(rec: np.ndarray, s: np.ndarray, sr: int, f1: float, f2: float) -> np.ndarray:
    """Déconvolution spectrale régularisée hors bande (rec: (n, c))."""
    n = rec.shape[0] + len(s)
    nf = sfft.next_fast_len(n, real=True)
    S = sfft.rfft(s, nf)
    f = np.fft.rfftfreq(nf, 1 / sr)
    p2 = np.abs(S) ** 2
    lo, hi = f1, min(f2, 0.49 * sr)
    inband = (f >= lo) & (f <= hi)
    eps = np.where(inband, 1e-8, 1.0) * p2[inband].max()
    inv = np.conj(S) / (p2 + eps)
    R = sfft.rfft(rec, nf, axis=0)
    return sfft.irfft(R * inv[:, None], nf, axis=0)[: rec.shape[0]]


@dataclass
class IRResult:
    ir: np.ndarray          # (n, c)
    sr: int
    segments: int
    snr_db: float
    t30: float | None
    edt: float | None


def extract_ir(rec: np.ndarray, p: SweepParams, max_len: float | None = None,
               trim_onset: bool = True, trim_tail: bool = True,
               normalize: bool = True, pre_ms: float = 1.0) -> IRResult:
    """Extrait l'IR moyennée d'un enregistrement du stimulus.

    rec doit être au même taux d'échantillonnage que p.sr.
    """
    if rec.ndim == 1:
        rec = rec[:, None]
    s = sweep(p)
    d = _deconvolve_sweep(rec, s, p.sr, p.f1, p.f2)
    env = np.abs(d).sum(axis=1)

    P = p.period
    peak = int(np.argmax(env))
    t0 = peak % P
    harm_lead = int(p.L * np.log(8) * p.sr)   # harmoniques jusqu'à l'ordre 8 rejetées
    guard = int(0.010 * p.sr)                  # marge avant le pic
    seg_len = P - harm_lead - 2 * guard
    if max_len:
        seg_len = min(seg_len, int(max_len * p.sr))
    seg_len = max(seg_len, int(0.05 * p.sr))

    segs = []
    pk = env[peak]
    k = 0
    while True:
        c = t0 + k * P
        a = c - guard
        if a + seg_len > d.shape[0]:
            break
        if a >= 0:
            local = env[max(c - guard, 0): c + guard + 1].max()
            if local > pk * 0.5:               # sweep effectivement présent
                segs.append(d[a: a + seg_len])
        k += 1
    if not segs:
        raise ValueError("Aucun sweep détecté dans l'enregistrement. "
                         "Vérifiez les paramètres de capture et le taux d'échantillonnage.")
    # alignement fin de chaque segment sur le premier (dérive d'horloge éventuelle)
    ref = segs[0]
    aligned = [ref]
    for sgt in segs[1:]:
        xc = sfft.irfft(sfft.rfft(ref.sum(1), 2 * seg_len) *
                        np.conj(sfft.rfft(sgt.sum(1), 2 * seg_len)), 2 * seg_len)
        lag = int(np.argmax(np.r_[xc[-8:], xc[:9]])) - 8
        aligned.append(np.roll(sgt, lag, axis=0))
    ir = np.mean(aligned, axis=0)

    # bruit de fond estimé sur les 10 % finaux
    tail = ir[int(0.9 * len(ir)):]
    noise = np.sqrt(np.mean(tail ** 2) + 1e-30)
    snr = 20 * np.log10(np.abs(ir).max() / noise)

    if trim_onset:
        e = np.abs(ir).max(axis=1)
        on = int(np.argmax(e > e.max() * 0.1))
        start = max(on - int(pre_ms * 1e-3 * p.sr), 0)
        ir = ir[start:]
    if trim_tail:
        ir = trim_ir_tail(ir, p.sr)
    if normalize:
        ir = ir / np.abs(ir).max()

    t30, edt = decay_times(ir.mean(axis=1), p.sr)
    return IRResult(ir=ir, sr=p.sr, segments=len(segs), snr_db=float(snr), t30=t30, edt=edt)


def trim_ir_tail(ir: np.ndarray, sr: int, margin_db: float = 3.0) -> np.ndarray:
    """Coupe la queue là où l'enveloppe rejoint le bruit de fond, avec fondu."""
    m = (ir ** 2).mean(axis=1)
    w = max(int(0.05 * sr), 1)
    env = np.convolve(m, np.ones(w) / w, mode="same")
    env_db = 10 * np.log10(env + 1e-30)
    noise_db = float(np.median(env_db[int(0.7 * len(m)):]))
    pk = int(np.argmax(env))
    below = np.nonzero(env_db[pk:] < noise_db + margin_db)[0]
    if len(below) == 0:
        return ir
    cut = min(pk + int(below[0]), len(ir))
    out = ir[:cut].copy()
    fl = max(int(0.15 * (cut - pk)), 1)
    out[-fl:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(fl) / fl))[:, None]
    return out


def edc_db(h: np.ndarray) -> np.ndarray:
    e = np.cumsum((h ** 2)[::-1])[::-1]
    return 10 * np.log10(e / (e[0] + 1e-30) + 1e-30)


def decay_times(h: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    """T30 et EDT (s) par intégration de Schroeder."""
    edc = edc_db(h)
    t = np.arange(len(h)) / sr

    def fit(a, b):
        idx = np.nonzero((edc <= a) & (edc >= b))[0]
        if len(idx) < 10:
            return None
        slope = np.polyfit(t[idx], edc[idx], 1)[0]
        return float(-60.0 / slope) if slope < 0 else None

    return fit(-5, -35), fit(0, -10)


# ---------------------------------------------------------------- positions multiples

def _align_on_peak(irs: list[np.ndarray], pre: int) -> list[np.ndarray]:
    out = []
    n = 0
    for h in irs:
        p = int(np.argmax(np.abs(h).max(axis=1)))
        a = h[max(p - pre, 0):]
        if p < pre:
            a = np.vstack([np.zeros((pre - p, h.shape[1])), a])
        out.append(a)
        n = max(n, len(a))
    return [np.vstack([a, np.zeros((n - len(a), a.shape[1]))]) for a in out]


def band_coherence(irs: list[np.ndarray], sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Cohérence spectrale moyenne (paires d'IR) par tiers d'octave.

    1 = les IR sont identiques dans cette bande (la position exacte de la source
    n'y compte pas) ; vers 0 = la structure fine change d'une position à l'autre.
    """
    from scipy import fft as sfft
    hs = [h.mean(axis=1) for h in _align_on_peak(irs, int(0.002 * sr))]
    nf = sfft.next_fast_len(len(hs[0]), real=True)
    H = [sfft.rfft(h, nf) for h in hs]
    f = np.fft.rfftfreq(nf, 1 / sr)
    centers = 1000 * 2.0 ** (np.arange(-17, 14) / 3)
    centers = centers[(centers > 20) & (centers < 0.45 * sr)]
    coh = []
    for fc in centers:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        vals = []
        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                a, b = H[i][m], H[j][m]
                vals.append(np.abs(np.sum(a * np.conj(b))) /
                            np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2) + 1e-30))
        coh.append(np.mean(vals))
    return centers, np.array(coh)


def coherent_limit(centers: np.ndarray, coh: np.ndarray, threshold: float = 0.7) -> float:
    """Plus haute fréquence en dessous de laquelle toutes les bandes restent cohérentes."""
    bad = np.nonzero(coh < threshold)[0]
    if len(bad) == 0:
        return float(centers[-1])
    if bad[0] == 0:
        return 0.0
    return float(centers[bad[0] - 1] * 2 ** (1 / 6))


def combine_positions(irs: list[np.ndarray], sr: int) -> tuple[np.ndarray, float, tuple]:
    """Combine plusieurs captures (source déplacée de quelques cm entre chacune).

    Retourne l'IR combinée, la fréquence de cohérence et (bandes, cohérence).
    Sous la fréquence de cohérence : moyenne des IR (réduit bruit et erreurs).
    Au-dessus : première IR telle quelle (garde la bonne énergie de queue,
    qu'une moyenne de queues décorrélées sous-estimerait).
    """
    from .statistical import crossover
    al = _align_on_peak(irs, int(0.002 * sr))
    centers, coh = band_coherence(irs, sr)
    fc = coherent_limit(centers, coh)
    avg = np.mean(al, axis=0)
    out = crossover(avg, al[0], sr, fc) if fc > 0 else al[0]
    return out, fc, (centers, coh)

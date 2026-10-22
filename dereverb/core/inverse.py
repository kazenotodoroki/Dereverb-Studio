"""Méthode 1 : déconvolution spectrale régularisée sur toute la piste.

Si la piste a été produite par convolution avec exactement cette IR
(reverb à convolution, rendu numérique), cette méthode est quasi exacte :
X = Y·H* / (|H|² + β(f)), FFT de la longueur de la piste entière, donc
aucune approximation de fenêtrage. β(f) protège les fréquences où |H| est
faible (creux, hors bande) contre l'explosion du bruit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft


@dataclass
class InverseParams:
    reg_db: float = -60.0       # régularisation relative à la puissance moyenne de H
    fmin: float = 0.0           # bande de confiance (Hz) ; 0 = pas de limite
    fmax: float = 0.0           # 0 = Nyquist
    outband_db: float = 40.0    # régularisation supplémentaire hors bande
    dry: float = 0.0            # part de signal direct mélangée dans la piste
    wet: float = 1.0            # gain de l'IR dans la piste


def effective_ir(h: np.ndarray, dry: float, wet: float) -> np.ndarray:
    """IR vue par la piste : dry·δ + wet·h (h : 1D)."""
    he = wet * h.astype(np.float64)
    if dry:
        he = he.copy()
        he[0] += dry
    return he


def beta(H: np.ndarray, sr: int, nf: int, p: InverseParams) -> np.ndarray:
    p2 = np.abs(H) ** 2
    b = np.full(p2.shape, 10 ** (p.reg_db / 10) * p2.mean())
    f = np.fft.rfftfreq(nf, 1 / sr)
    out = np.zeros_like(f, dtype=bool)
    if p.fmin > 0:
        out |= f < p.fmin
    if p.fmax > 0:
        out |= f > p.fmax
    if out.any():
        # transition douce d'un tiers d'octave
        b = b * np.where(out, 10 ** (p.outband_db / 10), 1.0)
    return b


def deconvolve(y: np.ndarray, h: np.ndarray, sr: int, p: InverseParams) -> np.ndarray:
    """y : (n,) piste réverbérée ; h : (m,) IR. Retourne (n,)."""
    he = effective_ir(h, p.dry, p.wet)
    n = len(y)
    # marge pour la partie anticausale (IR à phase non minimale)
    nf = sfft.next_fast_len(n + 2 * len(he), real=True)
    H = sfft.rfft(he, nf)
    Y = sfft.rfft(y, nf)
    X = Y * np.conj(H) / (np.abs(H) ** 2 + beta(H, sr, nf, p))
    x = sfft.irfft(X, nf)
    return x[:n]


def reconvolve_residual_db(y: np.ndarray, x: np.ndarray, h: np.ndarray, dry: float, wet: float) -> float:
    """Erreur de reconstruction ||x*h − y|| / ||y|| en dB : plus c'est bas, plus
    l'estimation sèche explique la piste avec cette IR."""
    he = effective_ir(h, dry, wet)
    nf = sfft.next_fast_len(len(x) + len(he), real=True)
    yr = sfft.irfft(sfft.rfft(x, nf) * sfft.rfft(he, nf), nf)[: len(y)]
    return float(10 * np.log10(np.sum((yr - y) ** 2) / (np.sum(y ** 2) + 1e-30) + 1e-30))


def direct_part(h: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Son direct de l'IR : du début jusqu'à `ms` après le pic, fondu demi-Hann.

    Avec une IR mesurée (haut-parleur → salle → micro), déconvoluer puis
    reconvoluer par ce son direct donne l'instrument « tel que le micro
    l'entend sans la salle » : la couleur du haut-parleur de mesure s'annule
    au lieu d'être inversée (sinon un HP faible dans le grave donnerait un
    résultat boomy), et le résultat reste calé dans le temps sur la piste.
    """
    p = int(np.argmax(np.abs(h)))
    n = p + max(int(ms * 1e-3 * sr), 1)
    d = h[:n].astype(np.float64).copy()
    fl = max(int(0.5 * ms * 1e-3 * sr), 1)
    d[-fl:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fl) / fl)
    return d


def apply_target(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    nf = sfft.next_fast_len(len(x) + len(d), real=True)
    return sfft.irfft(sfft.rfft(x, nf) * sfft.rfft(d, nf), nf)[: len(x)]

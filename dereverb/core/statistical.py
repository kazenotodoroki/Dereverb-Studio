"""Méthode « Salle » : suppression de la réverbération tardive guidée par l'IR,
avec inversion exacte optionnelle dans le grave (hybride).

Pourquoi : l'inversion exacte suppose que le son vient exactement de l'endroit
où l'IR a été mesurée. Un orchestre (plusieurs sources), un musicien qui bouge,
un instrument qui ne rayonne pas comme un haut-parleur : la structure fine de
l'IR (phases, réflexions précises) change de quelques centimètres en quelques
centimètres dans l'aigu. Inverser une IR qui ne correspond pas crée des
pré-échos et des artefacts.

Ce qui, en revanche, ne dépend presque pas de la position : l'ENVELOPPE
d'énergie de la réverbération, bande par bande (temps de décroissance,
niveau relatif au son direct). On s'en sert pour prédire, trame par trame,
l'énergie de réverbération tardive contenue dans la piste :

    λ(k, t) = Σ_{l ≥ L0} E(k, l) · |S(k, t − l)|²

E est l'enveloppe temps-fréquence de l'IR (lissée en tiers d'octave,
normalisée par le son direct), S le son direct estimé. Un filtre de Wiener à
décision dirigée retire cette énergie ; le son direct et les premières
réflexions (avant L0) sont conservés, ce qui reste naturel.

Hybride : sous une fréquence de transition (grave, modes de salle), les
longueurs d'onde sont grandes et l'IR reste valable même si la source bouge
un peu ; on y applique la déconvolution exacte (avec cible « son direct »),
ce qui corrige aussi la coloration modale qu'un gain spectral ne peut pas
corriger. Typique : guitare dans une pièce qui sonne mal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import fft as sfft
from scipy.signal import oaconvolve, istft, stft

from .inverse import InverseParams, apply_target, deconvolve, direct_part


@dataclass
class RoomParams:
    late_ms: float = 50.0        # début de la réverbération retirée (après le son direct)
    strength: float = 1.0        # surestimation de la réverb (0,5 = doux, 2 = fort)
    floor_db: float = -24.0      # atténuation maximale
    smoothing: float = 0.85      # décision dirigée (moins de « bruit musical »)
    crossover_hz: float = 0.0    # 0 = pas d'hybride ; sinon inversion exacte en dessous
    inverse_reg_db: float = -40.0
    direct_ms: float = 2.0       # fenêtre du son direct pour la cible de l'inversion
    link: bool = True            # même gain sur tous les canaux (paires stéréo)


def _fft_size(sr: int) -> tuple[int, int]:
    n = 1024 if sr <= 50000 else 2048
    return n, n // 4


def _third_octave_smooth(P: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Lissage en 1/3 d'octave le long de l'axe 0 (fréquences)."""
    out = np.empty_like(P)
    cs = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    df = freqs[1] - freqs[0]
    for k, f in enumerate(freqs):
        lo = max(f * 2 ** (-1 / 6), 0.0)
        hi = f * 2 ** (1 / 6)
        a = int(lo / df)
        b = max(int(np.ceil(hi / df)) + 1, a + 1)
        b = min(b, len(freqs))
        a = min(a, b - 1)
        out[k] = (cs[b] - cs[a]) / (b - a)
    return out


def late_kernel(h: np.ndarray, sr: int, late_ms: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Retourne (E_tardive (K, L), gain_précoce (K,), L0) à partir de l'IR."""
    n_fft, hop = _fft_size(sr)
    p = int(np.argmax(np.abs(h)))
    # première trame centrée sur le son direct
    hh = np.r_[np.zeros(max(n_fft // 2 - p, 0)), h[max(p - n_fft // 2, 0):]]
    f, _, H = stft(hh, sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                   boundary=None, padded=True)
    E = np.abs(H) ** 2
    E = _third_octave_smooth(E, f)
    E = E / (E[:, :1] + 1e-20)
    # trames qui se recouvrent : chaque échantillon de l'IR compte Σw²/hop fois
    win = np.hanning(n_fft + 2)[1:-1]
    E[:, 1:] *= hop / np.sum(win ** 2)
    L0 = max(int(round(late_ms * 1e-3 * sr / hop)), 1)
    early = E[:, :L0].sum(axis=1)
    Ek = E.copy()
    Ek[:, :L0] = 0
    return Ek, early, L0


def suppress_late(Y: np.ndarray, Ek: np.ndarray, early: np.ndarray, p: RoomParams,
                  progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Y : STFT (C, K, T). Retourne les gains (C, K, T)."""
    P = np.abs(Y) ** 2 + 1e-20
    C, K, T = P.shape
    Pl = P.mean(axis=0, keepdims=True) if p.link else P
    # estimation itérative du son direct S² (point fixe)
    S2 = Pl / early[None, :, None]
    for _ in range(6):   # point fixe amorti (évite l'oscillation sur/sous-estimation)
        lam = oaconvolve(S2, Ek[None], axes=2)[..., :T]
        G2 = np.clip(1 - p.strength * lam / Pl, 10 ** (p.floor_db / 10), 1)
        S2 = 0.5 * S2 + 0.5 * G2 * Pl / early[None, :, None]
    lam = p.strength * oaconvolve(S2, Ek[None], axes=2)[..., :T] + 1e-20
    # Wiener à décision dirigée
    gmin = 10 ** (p.floor_db / 20)
    G = np.empty_like(Pl)
    prev = np.zeros(Pl.shape[:2])
    b = p.smoothing
    for t in range(T):
        ml = np.maximum(Pl[..., t] / lam[..., t] - 1, 0)
        xi = b * prev / lam[..., t] + (1 - b) * ml
        g = np.maximum(xi / (1 + xi), gmin)
        G[..., t] = g
        prev = g ** 2 * Pl[..., t]
        if progress and t % 2000 == 0:
            progress(0.2 + 0.6 * t / T)
    return np.broadcast_to(G, P.shape)


def process(y: np.ndarray, h: np.ndarray, sr: int, p: RoomParams,
            progress: Callable[[float], None] | None = None) -> np.ndarray:
    """y : (n, C) ; h : (m, C). Retourne (n, C)."""
    n, C = y.shape
    n_fft, hop = _fft_size(sr)
    hm = h.mean(axis=1)
    Ek, early, _ = late_kernel(hm, sr, p.late_ms)
    _, _, Y = stft(y.T, sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    if progress:
        progress(0.15)
    G = suppress_late(Y, Ek, early, p, progress)
    _, x = istft(G * Y, sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    x = x.T[:n]
    if x.shape[0] < n:
        x = np.vstack([x, np.zeros((n - x.shape[0], C))])

    if p.crossover_hz > 0:
        lo = np.empty_like(x)
        for c in range(C):
            hc = h[:, min(c, h.shape[1] - 1)]
            s = deconvolve(y[:, c], hc, sr, InverseParams(reg_db=p.inverse_reg_db))
            lo[:, c] = apply_target(s, direct_part(hc, sr, p.direct_ms))
        x = crossover(lo, x, sr, p.crossover_hz)
    if progress:
        progress(1.0)
    return x


def crossover(low: np.ndarray, high: np.ndarray, sr: int, fc: float) -> np.ndarray:
    """Transition complémentaire (somme = 1) en cosinus sur une octave, phase nulle."""
    n = low.shape[0]
    nf = sfft.next_fast_len(n, real=True)
    f = np.fft.rfftfreq(nf, 1 / sr)
    u = np.clip((np.log2(np.maximum(f, 1e-6) / fc) + 0.5), 0, 1)
    w_hi = 0.5 - 0.5 * np.cos(np.pi * u)
    L = sfft.rfft(low, nf, axis=0)
    Hh = sfft.rfft(high, nf, axis=0)
    out = sfft.irfft(L * (1 - w_hi)[:, None] + Hh * w_hi[:, None], nf, axis=0)
    return out[:n]

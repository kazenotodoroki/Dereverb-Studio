"""Méthode 2 : déconvolution par optimisation (L-BFGS préconditionné, PyTorch).

On cherche le signal sec x qui minimise
    ||(x ∗ h)[0:n] − y||² / ||y||²  +  λ · mean |STFT(x)|
- le terme de données est une convolution LINÉAIRE tronquée à la durée de la
  piste : contrairement à la méthode 1, une piste coupée avant la fin de la
  queue de réverbération est modélisée correctement ;
- le terme de parcimonie (L1 lissée dans le plan temps-fréquence) exploite le
  fait qu'un signal sec est plus « creux » qu'un signal réverbéré : il comble
  les creux spectraux de l'IR et rejette le bruit au lieu de l'amplifier.
x est paramétré par x = G·z où G est l'inverse régularisé (méthode 1) : le
problème devient bien conditionné et L-BFGS converge bien plus vite. Au départ
z = y, donc on part exactement de la solution de la méthode 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy import fft as sfft

from .inverse import InverseParams, beta, effective_ir


@dataclass
class IterativeParams:
    iterations: int = 200
    sparsity: float = 0.01      # λ ; 0 = pure moindres carrés
    n_fft: int = 1024
    hop: int = 256
    precond_reg_db: float = -40.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def refine(y: np.ndarray, h: np.ndarray, sr: int, inv: InverseParams, p: IterativeParams,
           progress: Callable[[float], None] | None = None,
           cancelled: Callable[[], bool] | None = None,
           device: torch.device | None = None) -> np.ndarray:
    device = device or pick_device()
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    n = len(y)
    scale = float(np.sqrt(np.mean(y ** 2)) + 1e-12)

    he = effective_ir(h, inv.dry, inv.wet)
    nf = sfft.next_fast_len(n + 2 * len(he), real=True)
    Hn = sfft.rfft(he, nf)
    pre = InverseParams(**{**inv.__dict__, "reg_db": max(inv.reg_db, p.precond_reg_db)})
    Gn = np.conj(Hn) / (np.abs(Hn) ** 2 + beta(Hn, sr, nf, pre))
    H = torch.as_tensor(Hn, device=device).to(torch.complex128 if dtype == torch.float64 else torch.complex64)
    G = torch.as_tensor(Gn, device=device).to(H.dtype)

    yt = torch.as_tensor(y / scale, dtype=dtype, device=device)
    yy = torch.sum(yt ** 2)
    win = torch.hann_window(p.n_fft, dtype=dtype, device=device)
    with torch.no_grad():
        cmean = torch.stft(yt, p.n_fft, p.hop, window=win, return_complex=True).abs().mean()
        eps2 = (1e-3 * cmean) ** 2

    z = torch.nn.Parameter(yt.clone())

    def dry_of(zz):
        return torch.fft.irfft(torch.fft.rfft(zz, nf) * G, nf)[:n]

    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=p.iterations, history_size=20,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")
    count = [0]

    class _Stop(Exception):
        pass

    def closure():
        if cancelled and cancelled():
            raise _Stop
        opt.zero_grad()
        x = dry_of(z)
        r = torch.fft.irfft(torch.fft.rfft(x, nf) * H, nf)[:n] - yt
        loss = torch.sum(r ** 2) / yy
        if p.sparsity > 0:
            C = torch.stft(x, p.n_fft, p.hop, window=win, return_complex=True)
            loss = loss + p.sparsity * torch.sqrt(C.real ** 2 + C.imag ** 2 + eps2).mean() / cmean
        loss.backward()
        count[0] += 1
        if progress:
            progress(min(count[0] / (p.iterations * 1.25), 0.99))
        return loss

    try:
        opt.step(closure)
    except _Stop:
        pass
    with torch.no_grad():
        return dry_of(z).cpu().numpy().astype(np.float64) * scale

"""Méthode 3 : réseau de neurones entraîné pour UNE IR (philosophie NAM).

Comme NAM apprend un ampli précis, on apprend ici une réverbération précise :
- les données d'entraînement sont générées à la volée : matière sèche
  (votre dossier de sons secs, sinon un générateur de synthèse) convoluée avec
  l'IR, avec de légères perturbations de l'IR (gain/décroissance de la queue,
  bruit) pour rendre le modèle robuste aux écarts entre l'IR mesurée et la
  réverbération réelle de la piste ;
- architecture hybride : le réseau reçoit la piste réverbérée Y ET l'estimation
  X1 de la méthode 1 (inverse régularisée), et prédit deux masques complexes :
      Ŝ = M1 · X1 + M2 · Y
  Il est initialisé pour sortir exactement X1, puis apprend à corriger les
  artefacts (pré-échos, bruit amplifié, écarts d'IR).
- réseau temporel convolutif dilaté (contexte ≈ ±11 s), non causal (offline).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import list_audio, load, resample
from .inverse import InverseParams, deconvolve
from .iterative import pick_device
from .synth import synthetic_dry


@dataclass
class NeuralConfig:
    sr: int = 48000
    n_fft: int = 2048
    hop: int = 512
    channels: int = 192
    dilations: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 1])
    seg_seconds: float = 4.0
    batch: int = 8
    steps: int = 3000
    lr: float = 3e-4
    spectral_weight: float = 0.1
    inv_reg_db: float = -30.0
    robustness: float = 0.5     # 0 = IR exacte (reverb numérique) ; 1 = forte variabilité (salle réelle)
    noise_db: float = -70.0     # bruit de fond typique des pistes


# ---------------------------------------------------------------- modèle

class _Block(nn.Module):
    def __init__(self, ch: int, d: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch)
        self.conv = nn.Conv1d(ch, 2 * ch, 3, dilation=d, padding=d)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        a, b = self.conv(self.norm(x)).chunk(2, dim=1)
        return x + self.proj(torch.tanh(a) * torch.sigmoid(b))   # unité à porte (WaveNet / NAM)


class DereverbNet(nn.Module):
    def __init__(self, n_bins: int, ch: int, dilations: list[int]):
        super().__init__()
        self.n_bins = n_bins
        self.inp = nn.Conv1d(4 * n_bins, ch, 1)
        self.blocks = nn.ModuleList(_Block(ch, d) for d in dilations)
        self.out = nn.Conv1d(ch, 4 * n_bins, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, Y: torch.Tensor, X1: torch.Tensor) -> torch.Tensor:
        """Y, X1 : STFT complexes (B, F, T). Retourne Ŝ complexe (B, F, T)."""
        # magnitudes log + X1 complexe compressé (information de phase)
        c = X1 / (X1.abs() + 1e-4) ** 0.7
        feats = torch.cat([torch.log(Y.abs() + 1e-4), torch.log(X1.abs() + 1e-4), c.real, c.imag], dim=1)
        h = self.inp(feats)
        for b in self.blocks:
            h = b(h)
        r1, i1, r2, i2 = self.out(h).chunk(4, dim=1)
        M1 = torch.complex(1 + 2 * torch.tanh(r1), 2 * torch.tanh(i1))
        M2 = torch.complex(2 * torch.tanh(r2), 2 * torch.tanh(i2))
        return M1 * X1 + M2 * Y


# ---------------------------------------------------------------- données

class DryCorpus:
    def __init__(self, sr: int, folder: str | Path | None = None, max_minutes: float = 60,
                 log: Callable[[str], None] = print):
        self.items: list[np.ndarray] = []
        total = 0
        if folder:
            for p in list_audio(folder):
                try:
                    a = load(p)
                except Exception as e:  # noqa: BLE001
                    log(f"ignoré : {p.name} ({e})")
                    continue
                x = resample(a.mono(), a.sr, sr).astype(np.float32)
                if len(x) < sr:
                    continue
                self.items.append(x)
                total += len(x)
                if total > max_minutes * 60 * sr:
                    break
            log(f"Matière sèche : {len(self.items)} fichiers, {total / sr / 60:.1f} min")
        if not self.items:
            log("Aucun son sec fourni : génération d'un corpus de synthèse (3 min).")
            self.items = [synthetic_dry(sr, 30, seed=s).astype(np.float32) for s in range(6)]
        self.weights = np.array([len(x) for x in self.items], dtype=float)
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng()

    def window(self, n: int) -> np.ndarray:
        for _ in range(20):
            x = self.items[self.rng.choice(len(self.items), p=self.weights)]
            if len(x) <= n:
                w = np.zeros(n, np.float32)
                off = self.rng.integers(0, n - len(x) + 1)
                w[off: off + len(x)] = x
            else:
                s = self.rng.integers(0, len(x) - n)
                w = x[s: s + n].copy()
            tail = w[-n // 3:]
            if np.sqrt(np.mean(tail ** 2)) > 1e-4:
                break
        return w * (1 if self.rng.random() < .5 else -1)


def _perturb_ir(h: torch.Tensor, sr: int, k: float, g: torch.Generator) -> torch.Tensor:
    """Variantes plausibles de l'IR : gain et vitesse de décroissance de la queue."""
    if k <= 0:
        return h.clone()
    t = torch.arange(len(h), device=h.device, dtype=h.dtype) / sr
    gain_db = (torch.rand(1, generator=g).item() * 2 - 1) * 3 * k
    dec = (torch.rand(1, generator=g).item() * 2 - 1) * 0.8 * k        # 1/s
    ramp = torch.clamp(t / 0.005, 0, 1)                                 # son direct préservé
    curve = 1 + ramp * (10 ** (gain_db / 20) * torch.exp(dec * t) - 1)
    out = h * curve
    out = out + torch.randn(len(h), generator=g, device="cpu").to(h) * h.abs().max() * 1e-3 * k * torch.exp(-t * 2)
    return out


def _inv_torch(y: torch.Tensor, h: torch.Tensor, nf: int, reg_db: float) -> torch.Tensor:
    H = torch.fft.rfft(h, nf)
    p2 = H.abs() ** 2
    Gi = H.conj() / (p2 + 10 ** (reg_db / 10) * p2.mean())
    return torch.fft.irfft(torch.fft.rfft(y, nf) * Gi, nf)[..., : y.shape[-1]]


# ---------------------------------------------------------------- pertes

def _mrstft(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for n in (512, 1024, 2048):
        w = torch.hann_window(n, device=a.device)
        A = torch.stft(a, n, n // 4, window=w, return_complex=True).abs()
        B = torch.stft(b, n, n // 4, window=w, return_complex=True).abs()
        sc = torch.linalg.norm(A - B) / (torch.linalg.norm(B) + 1e-8)
        lm = F.l1_loss(torch.log(A + 1e-5), torch.log(B + 1e-5))
        loss = loss + sc + lm
    return loss / 3


def _snr_db(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10((ref ** 2).sum(-1) / (((est - ref) ** 2).sum(-1) + 1e-10) + 1e-10)


# ---------------------------------------------------------------- entraînement

@dataclass
class TrainReport:
    step: int
    loss: float
    val_snr: float | None = None
    base_snr: float | None = None
    elapsed: float = 0.0


class Trainer:
    def __init__(self, ir: np.ndarray, ir_sr: int, cfg: NeuralConfig, dry_folder: str | None = None,
                 log: Callable[[str], None] = print, device: torch.device | None = None):
        self.cfg = cfg
        self.log = log
        self.device = device or pick_device()
        h = resample(ir.astype(np.float64), ir_sr, cfg.sr)
        self.ir = h
        self.h = torch.tensor(h, dtype=torch.float32, device=self.device)
        self.corpus = DryCorpus(cfg.sr, dry_folder, log=log)
        self.model = DereverbNet(cfg.n_fft // 2 + 1, cfg.channels, cfg.dilations).to(self.device)
        # la couche de sortie (initialisée à zéro = identité sur X1) apprend plus lentement :
        # le modèle ne s'éloigne de la méthode 1 que lorsqu'il fait mieux.
        out_p = list(self.model.out.parameters())
        ids = {id(p) for p in out_p}
        body = [p for p in self.model.parameters() if id(p) not in ids]
        self.opt = torch.optim.AdamW([{"params": body, "lr": cfg.lr},
                                      {"params": out_p, "lr": cfg.lr * 0.05}], weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, [cfg.lr, cfg.lr * 0.05],
                                                         total_steps=cfg.steps, pct_start=0.1)
        self.win = torch.hann_window(cfg.n_fft, device=self.device)
        self.gen = torch.Generator().manual_seed(0)
        self.history: list[TrainReport] = []
        n_par = sum(p.numel() for p in self.model.parameters())
        log(f"Modèle : {n_par / 1e6:.2f} M paramètres — calcul sur {self.device.type.upper()}")
        self.val = self._batch(4, seed=1234)

    def _batch(self, B: int, seed: int | None = None):
        cfg = self.cfg
        g = torch.Generator().manual_seed(seed) if seed is not None else self.gen
        if seed is not None:
            self.corpus.rng = np.random.default_rng(seed)
        T = int(cfg.seg_seconds * cfg.sr)
        Lh = len(self.h)
        ctx = min(Lh, int(6 * cfg.sr))
        N = ctx + T
        nf = int(2 ** math.ceil(math.log2(N + Lh)))
        dry = torch.tensor(np.stack([self.corpus.window(N) for _ in range(B)]), device=self.device)
        hs = torch.stack([_perturb_ir(self.h, cfg.sr, cfg.robustness, g) for _ in range(B)])
        wet = torch.fft.irfft(torch.fft.rfft(dry, nf) * torch.fft.rfft(hs, nf), nf)[:, :N]
        sc = wet.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8
        nl = cfg.noise_db + (torch.rand(B, 1, generator=g) * 20 - 10).to(self.device)
        wet = wet + torch.randn(wet.shape, generator=g).to(self.device) * sc * 10 ** (nl / 20)
        x1 = _inv_torch(wet, self.h, nf, cfg.inv_reg_db)
        wet, x1, dry = wet[:, -T:], x1[:, -T:], dry[:, -T:]
        # normalisation + gain aléatoire (invariance de niveau)
        sc = wet.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8
        gain = 10 ** ((torch.rand(B, 1, generator=g) * 20 - 10).to(self.device) / 20)
        f = gain / sc
        return wet * f, x1 * f, dry * f

    def _forward(self, wet, x1):
        cfg = self.cfg
        Y = torch.stft(wet, cfg.n_fft, cfg.hop, window=self.win, return_complex=True)
        X1 = torch.stft(x1, cfg.n_fft, cfg.hop, window=self.win, return_complex=True)
        S = self.model(Y, X1)
        return torch.istft(S, cfg.n_fft, cfg.hop, window=self.win, length=wet.shape[-1])

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            wet, x1, dry = self.val
            est = self._forward(wet, x1)
            v = _snr_db(est, dry).mean().item()
            b = _snr_db(x1, dry).mean().item()
        self.model.train()
        return v, b

    def train(self, progress: Callable[[TrainReport], None] | None = None,
              cancelled: Callable[[], bool] | None = None, val_every: int = 50):
        cfg = self.cfg
        t0 = time.time()
        v, b = self.validate()
        self.log(f"Départ : SNR méthode 1 = {b:.2f} dB")
        for step in range(1, cfg.steps + 1):
            if cancelled and cancelled():
                self.log("Entraînement interrompu.")
                break
            wet, x1, dry = self._batch(cfg.batch)
            est = self._forward(wet, x1)
            loss = -_snr_db(est, dry).mean() / 10 + self.cfg.spectral_weight * _mrstft(est, dry)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.sched.step()
            rep = TrainReport(step, float(loss.item()), elapsed=time.time() - t0)
            if step % val_every == 0 or step == cfg.steps:
                rep.val_snr, rep.base_snr = self.validate()
            self.history.append(rep)
            if progress:
                progress(rep)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        torch.save({"format": "dereverb-studio/1", "config": asdict(self.cfg),
                    "state": self.model.state_dict(), "ir": self.ir.astype(np.float32),
                    "history": [asdict(r) for r in self.history]}, path)
        return path


# ---------------------------------------------------------------- inférence

class NeuralModel:
    def __init__(self, path: str | Path, device: torch.device | None = None):
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        self.cfg = NeuralConfig(**d["config"])
        self.ir = np.asarray(d["ir"], dtype=np.float64)
        self.device = device or pick_device()
        self.model = DereverbNet(self.cfg.n_fft // 2 + 1, self.cfg.channels, self.cfg.dilations)
        self.model.load_state_dict(d["state"])
        self.model.to(self.device).eval()
        self.path = Path(path)
        self.history = d.get("history", [])

    @torch.no_grad()
    def process(self, y: np.ndarray, sr: int, progress: Callable[[float], None] | None = None,
                chunk: int = 1500, ctx: int = 600) -> np.ndarray:
        cfg = self.cfg
        yr = resample(y, sr, cfg.sr)
        x1 = deconvolve(yr, self.ir, cfg.sr, InverseParams(reg_db=cfg.inv_reg_db))
        sc = float(np.sqrt(np.mean(yr ** 2)) + 1e-12)
        win = torch.hann_window(cfg.n_fft, device=self.device)
        tY = torch.tensor(yr / sc, dtype=torch.float32, device=self.device)
        tX = torch.tensor(x1 / sc, dtype=torch.float32, device=self.device)
        Y = torch.stft(tY, cfg.n_fft, cfg.hop, window=win, return_complex=True)[None]
        X1 = torch.stft(tX, cfg.n_fft, cfg.hop, window=win, return_complex=True)[None]
        T = Y.shape[-1]
        out = torch.zeros_like(Y)
        for s in range(0, T, chunk):
            a, b = max(s - ctx, 0), min(s + chunk + ctx, T)
            S = self.model(Y[..., a:b], X1[..., a:b])
            out[..., s: min(s + chunk, T)] = S[..., s - a: s - a + min(chunk, T - s)]
            if progress:
                progress(min((s + chunk) / T, 1.0))
        est = torch.istft(out[0], cfg.n_fft, cfg.hop, window=win, length=len(yr)).cpu().numpy()
        return resample(est.astype(np.float64) * sc, cfg.sr, sr)[: len(y)]

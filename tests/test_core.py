"""Vérifications numériques sur signaux de synthèse : python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dereverb.core.audio import fft_convolve  # noqa: E402
from dereverb.core.capture import SweepParams, extract_ir, stimulus  # noqa: E402
from dereverb.core.inverse import InverseParams, deconvolve  # noqa: E402
from dereverb.core.iterative import IterativeParams, refine  # noqa: E402
from dereverb.core.synth import synthetic_dry, synthetic_ir  # noqa: E402

SR = 48000


def sdr(est, ref):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((est - ref) ** 2))


def main():
    h = synthetic_ir(SR, 1.8, seed=1)
    x = synthetic_dry(SR, 10, seed=2)
    full = fft_convolve(x, h)
    xp = np.r_[x, np.zeros(len(h) - 1)]
    print(f"piste réverbérée vs sec      : {sdr(full[:len(x)], x):7.1f} dB")

    e = deconvolve(full, h, SR, InverseParams(reg_db=-140))
    print(f"inverse, IR exacte           : {sdr(e, xp):7.1f} dB")
    assert sdr(e, xp) > 100

    rng = np.random.default_rng(0)
    noisy = full + rng.standard_normal(len(full)) * full.std() * 10 ** (-50 / 20)
    e1 = deconvolve(noisy, h, SR, InverseParams(reg_db=-40))
    e2 = refine(noisy, h, SR, InverseParams(), IterativeParams(iterations=80, sparsity=0.01))
    print(f"bruit −50 dB, inverse        : {sdr(e1, xp):7.1f} dB")
    print(f"bruit −50 dB, itérative      : {sdr(e2, xp):7.1f} dB")

    trunc = full[:len(x)]
    m = len(x) - len(h)
    e1 = deconvolve(trunc, h, SR, InverseParams(reg_db=-60))
    e2 = refine(trunc, h, SR, InverseParams(), IterativeParams(iterations=150, sparsity=0))
    print(f"queue coupée, inverse        : {sdr(e1[:m], x[:m]):7.1f} dB (hors dernière IR)")
    print(f"queue coupée, itérative      : {sdr(e2[:m], x[:m]):7.1f} dB (hors dernière IR)")

    p = SweepParams(sr=SR, duration=4, repeats=3, gap=2.5)
    rec = fft_convolve(stimulus(p), h)
    rec = rec + rng.standard_normal(len(rec)) * 1e-4
    r = extract_ir(rec[:, None], p, trim_onset=False, trim_tail=False, normalize=False)
    print(f"capture : {r.segments} sweeps, T30 = {r.t30:.2f} s (attendu ≈ 1.8 s), dynamique {r.snr_db:.0f} dB")
    assert r.segments == 3 and abs(r.t30 - 1.8) < 0.15
    # méthode Salle : la vraie réverb (autre position de source) ≠ IR mesurée au-dessus de 300 Hz
    from scipy.signal import butter, sosfiltfilt, stft, istft
    from dereverb.core import statistical as st
    sos = butter(4, 300, "low", fs=SR, output="sos")
    hc = synthetic_ir(SR, 1.8, seed=7)
    hb = sosfiltfilt(sos, h) + (hc - sosfiltfilt(sos, hc))
    yb = fft_convolve(x, hb)[:len(x)]
    L = int(0.05 * SR)
    early = fft_convolve(x, hb[:L])[:len(x)]
    late = yb - early
    nf, hop = st._fft_size(SR)
    kw = dict(window="hann", nperseg=nf, noverlap=nf - hop)
    Ek, eg, _ = st.late_kernel(h, SR, 50)
    G = st.suppress_late(stft(yb, SR, **kw)[2][None], Ek, eg, st.RoomParams())[0]
    ge = istft(G * stft(early, SR, **kw)[2], SR, **kw)[1][:len(x)]
    gl = istft(G * stft(late, SR, **kw)[2], SR, **kw)[1][:len(x)]
    before = 10 * np.log10(np.sum(early ** 2) / np.sum(late ** 2))
    after = 10 * np.log10(np.sum(ge ** 2) / np.sum(gl ** 2))
    print(f"salle, IR d'une autre position : rapport direct/réverb tardive {before:.1f} → {after:.1f} dB")
    assert after - before > 4
    print("OK")


if __name__ == "__main__":
    main()

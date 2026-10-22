"""Traitement en ligne de commande (lots, scripts).

Exemples :
  python cli.py voix.wav --ir hall.wav
  python cli.py *.wav --ir hall.wav --method iterative --reg -80 --sparsity 0.01 -o sorties/
  python cli.py voix.wav --model hall.pt
  python cli.py --sweep capture.wav                      # exporter le signal de capture
  python cli.py --extract enregistrement.wav -o ir.wav   # extraire une IR
"""
import argparse
from pathlib import Path

from dereverb.core import engine
from dereverb.core.audio import load, resample, save
from dereverb.core.capture import SweepParams, extract_ir, write_stimulus
from dereverb.core.inverse import InverseParams
from dereverb.core.iterative import IterativeParams
from dereverb.core.statistical import RoomParams


def main():
    ap = argparse.ArgumentParser(description="Dereverb Studio — ligne de commande")
    ap.add_argument("tracks", nargs="*")
    ap.add_argument("--ir")
    ap.add_argument("--model")
    ap.add_argument("--method", choices=list(engine.METHODS), default=None)
    ap.add_argument("--reg", type=float, default=-60)
    ap.add_argument("--dry", type=float, default=0.0)
    ap.add_argument("--wet", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--sparsity", type=float, default=0.01)
    ap.add_argument("--late-ms", type=float, default=60.0, help="méthode room : durée conservée")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=-18.0)
    ap.add_argument("--crossover", type=float, default=0.0, help="méthode room : grave exact sous cette fréquence")
    ap.add_argument("--direct-ms", type=float, default=2.0, help="cible son direct (0 = impulsion)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--sweep", help="exporter le signal de capture vers ce fichier")
    ap.add_argument("--extract", help="extraire une IR de cet enregistrement")
    ap.add_argument("--sweep-json", help="paramètres du sweep (.sweep.json)")
    a = ap.parse_args()

    p = SweepParams.from_json(Path(a.sweep_json).read_text()) if a.sweep_json else SweepParams()
    if a.sweep:
        write_stimulus(a.sweep, p)
        print(f"Signal de capture : {a.sweep}")
        return
    if a.extract:
        rec = load(a.extract)
        data = resample(rec.data, rec.sr, p.sr)
        r = extract_ir(data, p)
        out = a.output or str(Path(a.extract).with_name(Path(a.extract).stem + "_IR.wav"))
        save(out, r.ir, r.sr)
        print(f"IR : {out}  (T30 {r.t30}, {r.segments} sweeps, dynamique {r.snr_db:.0f} dB)")
        return

    method = a.method or ("neural" if a.model else "inverse")
    s = engine.ProcessSettings(method=method,
                               inverse=InverseParams(reg_db=a.reg, dry=a.dry, wet=a.wet),
                               iterative=IterativeParams(iterations=a.iterations, sparsity=a.sparsity),
                               room=RoomParams(late_ms=a.late_ms, strength=a.strength, floor_db=a.floor,
                                               crossover_hz=a.crossover),
                               target_ms=a.direct_ms, model_path=a.model, output_dir=a.output)
    ir = load(a.ir) if a.ir else None
    for t in a.tracks:
        tr = load(t)
        res = engine.process(tr, ir, s)
        path = engine.write(res, engine.output_path(tr, s), s.subtype)
        print(f"{path}   {res.metric.lower()} {max(res.residual_db):.1f} dB")


if __name__ == "__main__":
    main()

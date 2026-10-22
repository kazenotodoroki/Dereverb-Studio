"""Page « Capture » : signal de mesure, extraction des IR."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QVBoxLayout, QWidget)

from ..core.audio import Audio, load, resample, save
from ..core.capture import (IRResult, SweepParams, combine_positions, decay_times, extract_ir,
                            trim_ir_tail, write_stimulus)
from .state import IRLibrary
from .widgets import Card, Form, IRView, Worker, button, dspin, ispin, label

FILTER = "Audio (*.wav *.aif *.aiff *.flac *.w64 *.caf)"


@dataclass
class Capture:
    path: Path
    result: IRResult | None = None
    lib_name: str | None = None
    coherent_hz: float | None = None
    label: str | None = None


class CapturePage(QWidget):
    def __init__(self, lib: IRLibrary, status):
        super().__init__()
        self.lib = lib
        self.status = status
        self.caps: list[Capture] = []
        self.worker: Worker | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(20)

        # ---- colonne gauche : signal
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(label("Capture", "H1"))
        left.addWidget(label("Jouez ce signal à travers la réverbération (salle, plugin, matériel), "
                             "enregistrez le retour, puis importez l'enregistrement. "
                             "Une capture par piste si chaque piste a sa propre réverbération.",
                             "Hint", True))
        sc = Card("Signal de capture")
        f = Form()
        self.sr = QComboBox()
        self.sr.addItems(["44100", "48000", "88200", "96000"])
        self.sr.setCurrentText("48000")
        f.row("Fréquence", self.sr)
        self.f1 = f.row("Début", dspin(20, 5, 200, 1, 0, "Hz"))
        self.f2 = f.row("Fin", dspin(22000, 2000, 48000, 500, 0, "Hz"))
        self.dur = f.row("Durée d'un sweep", dspin(10, 1, 60, 1, 1, "s"),
                         "Plus long = meilleur rapport signal/bruit.")
        self.rep = f.row("Répétitions", ispin(3, 1, 16), "Les IR de chaque sweep sont moyennées.")
        self.gap = f.row("Silence après sweep", dspin(6, 0.5, 30, 0.5, 1, "s"),
                         "Doit dépasser la durée de la réverbération (RT60).")
        self.lvl = f.row("Niveau crête", dspin(-6, -40, 0, 1, 1, "dBFS"))
        sc.add(f)
        self.total = sc.add(label("", "Hint", True))
        for w in (self.dur, self.rep, self.gap):
            w.valueChanged.connect(self.update_total)
        sc.add(button("Exporter le signal…", "Primary", self.export))
        left.addWidget(sc)

        oc = Card("Extraction")
        self.trim_on = oc.add(QCheckBox("Recaler sur le son direct"))
        self.trim_on.setChecked(True)
        self.trim_on.setToolTip("Supprime la latence d'enregistrement avant le son direct (1 ms conservée).\n"
                                "Décochez pour garder le pré-délai exact d'un plugin.")
        self.trim_tail = oc.add(QCheckBox("Couper la queue au bruit de fond"))
        self.trim_tail.setChecked(True)
        self.norm = oc.add(QCheckBox("Normaliser (crête = 0 dBFS)"))
        self.norm.setChecked(True)
        self.norm.setToolTip("Décochez pour conserver le gain absolu du système "
                             "(utile si le niveau sec ↔ réverbéré doit être exact).")
        left.addWidget(oc)
        left.addStretch()
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(330)
        root.addWidget(lw)

        # ---- colonne droite : enregistrements
        right = QVBoxLayout()
        right.setSpacing(12)
        h = QHBoxLayout()
        h.addWidget(label("Enregistrements", "H2"))
        h.addStretch()
        h.addWidget(button("Importer…", None, self.import_recs))
        self.extract_b = button("Extraire les IR", "Primary", self.extract)
        h.addWidget(self.extract_b)
        self.comb_b = button("Combiner les positions", None, self.combine)
        self.comb_b.setToolTip("Sélectionnez plusieurs captures faites en déplaçant la source de quelques\n"
                               "centimètres (même micro). Mesure jusqu'à quelle fréquence l'IR reste\n"
                               "valable malgré les mouvements du musicien, et moyenne le grave.")
        h.insertWidget(2, self.comb_b)
        right.addLayout(h)
        self.list = QListWidget()
        self.list.setObjectName("List")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentRowChanged.connect(self.on_select)
        right.addWidget(self.list, 1)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setVisible(False)
        right.addWidget(self.prog)

        ic = Card("Réponse impulsionnelle")
        mh = QHBoxLayout()
        self.m_t30 = self._metric(mh, "T30")
        self.m_edt = self._metric(mh, "EDT")
        self.m_snr = self._metric(mh, "Dynamique")
        self.m_len = self._metric(mh, "Longueur")
        self.m_coh = self._metric(mh, "Grave exact")
        self.m_coh.setToolTip("Fréquence sous laquelle les captures combinées concordent :\n"
                              "la méthode Salle y inverse l'IR exactement (mode « Auto »).")
        mh.addStretch()
        ic.add(mh)
        self.irview = IRView()
        ic.add(self.irview)
        bh = QHBoxLayout()
        bh.addWidget(label("<span style='color:#a78bfa'>●</span> énergie&nbsp;&nbsp;&nbsp;"
                           "<span style='color:#5eead4'>●</span> décroissance (Schroeder)", "Hint"))
        bh.addStretch()
        bh.addWidget(button("Enregistrer l'IR…", None, self.save_ir))
        ic.add(bh)
        right.addWidget(ic, 2)
        root.addLayout(right, 1)
        self.update_total()

    def _metric(self, lay, title):
        box = QVBoxLayout()
        box.setSpacing(2)
        box.addWidget(label(title, "H2"))
        v = label("—", "Metric")
        box.addWidget(v)
        lay.addLayout(box)
        lay.addSpacing(26)
        return v

    def params(self) -> SweepParams:
        return SweepParams(sr=int(self.sr.currentText()), f1=self.f1.value(), f2=self.f2.value(),
                           duration=self.dur.value(), repeats=self.rep.value(), gap=self.gap.value(),
                           level_db=self.lvl.value())

    def set_params(self, p: SweepParams):
        self.sr.setCurrentText(str(p.sr))
        self.f1.setValue(p.f1)
        self.f2.setValue(p.f2)
        self.dur.setValue(p.duration)
        self.rep.setValue(p.repeats)
        self.gap.setValue(p.gap)
        self.lvl.setValue(p.level_db)

    def update_total(self):
        p = self.params()
        t = p.pre_silence + p.repeats * (p.duration + p.gap)
        self.total.setText(f"Durée totale : {t:.1f} s — les mêmes réglages doivent servir à l'extraction "
                           "(enregistrés aussi dans un fichier .sweep.json).")

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter le signal de capture", "capture_sweep.wav",
                                              "WAV (*.wav)")
        if not path:
            return
        write_stimulus(path, self.params())
        self.status(f"Signal exporté : {path}")

    # ------------------------------------------------------------ enregistrements
    def import_recs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Importer des enregistrements", "", FILTER)
        for p in files:
            p = Path(p)
            js = p.with_suffix(".sweep.json")
            if js.exists():
                self.set_params(SweepParams.from_json(js.read_text()))
            c = Capture(p)
            self.caps.append(c)
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, len(self.caps) - 1)
            self.list.addItem(it)
        if files:
            self.list.setCurrentRow(self.list.count() - 1)

    def extract(self):
        todo = [i for i, c in enumerate(self.caps) if c.result is None]
        if not todo:
            self.status("Importez d'abord des enregistrements du signal de capture.")
            return
        p = self.params()
        opts = dict(trim_onset=self.trim_on.isChecked(), trim_tail=self.trim_tail.isChecked(),
                    normalize=self.norm.isChecked())

        def task(w: Worker):
            out = []
            for k, i in enumerate(todo):
                c = self.caps[i]
                a = load(c.path)
                rec = resample(a.data, a.sr, p.sr) if a.sr != p.sr else a.data
                try:
                    r = extract_ir(rec, p, **opts)
                    out.append((i, r, None))
                except Exception as e:  # noqa: BLE001
                    out.append((i, None, str(e)))
                w.progress.emit((k + 1) / len(todo))
            return out

        self.worker = Worker(task, self)
        self.worker.progress.connect(lambda f: self.prog.setValue(int(f * 1000)))
        self.worker.done.connect(self.on_extracted)
        self.worker.failed.connect(lambda m: (self.busy(False), QMessageBox.warning(self, "Extraction", m)))
        self.busy(True)
        self.worker.start()

    def on_extracted(self, out):
        self.busy(False)
        errs = []
        for i, r, err in out:
            c = self.caps[i]
            item = self.list.item(i)
            if r is None:
                errs.append(f"{c.path.name} : {err}")
                item.setText(f"{c.path.name}   ·   échec")
                continue
            c.result = r
            c.lib_name = self.add_to_library(c.path.stem, r.ir, r.sr, c.path, {})
            t30 = f"{r.t30:.2f} s" if r.t30 else "—"
            item.setText(f"{c.path.name}   ·   T30 {t30}   ·   {r.segments} sweep(s)   ·   → bibliothèque")
        if errs:
            QMessageBox.warning(self, "Extraction", "\n".join(errs))
        self.status("IR ajoutées à la bibliothèque — assignez-les dans « Pistes ».")
        self.on_select(self.list.currentRow())

    def add_to_library(self, name, ir, sr, path, meta) -> str:
        """Une entrée par micro si la capture est multicanal (> 2 canaux)."""
        if ir.shape[1] <= 2:
            return self.lib.add(name, Audio(ir, sr, path, dict(meta, source="capture")))
        first = None
        for c in range(ir.shape[1]):
            m = dict(meta, source="capture")
            if "coherent_hz_ch" in meta:
                m["coherent_hz"] = meta["coherent_hz_ch"][c]
            n = self.lib.add(f"{name} · micro {c + 1}", Audio(ir[:, c:c + 1], sr, path, m))
            first = first or n
        return first

    def combine(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()})
        caps = [self.caps[r] for r in rows if self.caps[r].result is not None]
        if len(caps) < 2:
            QMessageBox.information(self, "Combiner", "Sélectionnez au moins deux captures extraites "
                                    "(même micro, source déplacée de quelques centimètres).")
            return
        irs = [c.result.ir for c in caps]
        chans = {ir.shape[1] for ir in irs}
        if len(chans) != 1 or len({c.result.sr for c in caps}) != 1:
            QMessageBox.warning(self, "Combiner", "Les captures doivent avoir le même nombre de canaux "
                                "et la même fréquence d'échantillonnage.")
            return
        sr = caps[0].result.sr
        C = chans.pop()
        parts, fcs = [], []
        for ch in range(C):
            out, fc, _ = combine_positions([ir[:, ch:ch + 1] for ir in irs], sr)
            parts.append(out)
            fcs.append(fc)
        n = min(len(p) for p in parts)
        ir = np.hstack([p[:n] for p in parts])
        if self.trim_tail.isChecked():
            ir = trim_ir_tail(ir, sr)
        t30, edt = decay_times(ir.mean(axis=1), sr)
        res = IRResult(ir, sr, sum(c.result.segments for c in caps),
                       min(c.result.snr_db for c in caps), t30, edt)
        base = caps[0].path.stem
        meta = {"coherent_hz": float(min(fcs)), "coherent_hz_ch": fcs}
        c = Capture(caps[0].path, res, coherent_hz=float(min(fcs)), label=f"{base} · {len(caps)} positions")
        c.lib_name = self.add_to_library(c.label, ir, sr, caps[0].path, meta)
        self.caps.append(c)
        it = QListWidgetItem(f"{c.label}   ·   grave exact ≤ {min(fcs):.0f} Hz   ·   → bibliothèque")
        self.list.addItem(it)
        self.list.setCurrentRow(self.list.count() - 1)
        self.status(f"Combinaison : IR valable exactement jusqu'à {min(fcs):.0f} Hz.")

    def on_select(self, row: int):
        c = self.caps[row] if 0 <= row < len(self.caps) else None
        r = c.result if c else None
        self.irview.display(r.ir if r else None, r.sr if r else 48000)
        self.m_t30.setText(f"{r.t30:.2f} s" if r and r.t30 else "—")
        self.m_edt.setText(f"{r.edt:.2f} s" if r and r.edt else "—")
        self.m_snr.setText(f"{r.snr_db:.0f} dB" if r else "—")
        self.m_len.setText(f"{len(r.ir) / r.sr:.2f} s" if r else "—")
        self.m_coh.setText(f"≤ {c.coherent_hz:.0f} Hz" if c and c.coherent_hz is not None else "—")

    def save_ir(self):
        row = self.list.currentRow()
        if row < 0 or self.caps[row].result is None:
            return
        c = self.caps[row]
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'IR", f"{c.label or c.path.stem}_IR.wav",
                                              "WAV (*.wav)")
        if path:
            save(path, c.result.ir, c.result.sr, "FLOAT")
            if c.coherent_hz is not None:   # retrouvé au rechargement (mode « Auto »)
                Path(path).with_suffix(".ir.json").write_text(json.dumps({"coherent_hz": c.coherent_hz}))
            self.status(f"IR enregistrée : {path}")

    def busy(self, b):
        self.extract_b.setEnabled(not b)
        self.prog.setVisible(b)
        self.prog.setValue(0)

"""Page « Pistes » : import, assignation d'IR, traitement, écoute."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLineEdit, QMessageBox, QProgressBar, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..core import engine
from ..core.audio import AUDIO_EXT, Audio, load
from ..core.inverse import InverseParams
from ..core.iterative import IterativeParams
from ..core.statistical import RoomParams
from .state import IRLibrary
from .widgets import (Card, Form, PathField, Player, Segmented, WaveView, Worker, button, dspin,
                      ispin, label)

NONE = "—"
FILTER = "Audio (*.wav *.aif *.aiff *.flac *.ogg *.w64 *.caf)"


@dataclass
class Track:
    audio: Audio
    ir: str | None = None
    status: str = "En attente"
    result: engine.ProcessResult | None = None


class TracksPage(QWidget):
    def __init__(self, lib: IRLibrary, status):
        super().__init__()
        self.lib = lib
        self.status = status
        self.tracks: list[Track] = []
        self.worker: Worker | None = None
        self.player = Player()
        self.setAcceptDrops(True)

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(20)

        # ---- colonne principale
        main = QVBoxLayout()
        main.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(label("Pistes", "H1"))
        head.addStretch()
        head.addWidget(button("Charger une IR…", None, self.load_ir))
        head.addWidget(button("Ajouter des pistes…", None, self.add_dialog))
        self.rm_b = button("Retirer", "Ghost", self.remove_selected)
        head.addWidget(self.rm_b)
        main.addLayout(head)
        main.addWidget(label("Glissez vos pistes réverbérées ici. Chaque piste reçoit une IR "
                             "(mesurée dans « Capture » ou chargée depuis un fichier).", "Hint", True))

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Piste", "Format", "Durée", "IR", "État"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for c, w in ((1, 130), (2, 70), (3, 190), (4, 140)):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            self.table.setColumnWidth(c, w)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.itemSelectionChanged.connect(self.on_select)
        split.addWidget(self.table)

        viewer = QWidget()
        vl = QVBoxLayout(viewer)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)
        vh = QHBoxLayout()
        self.view_title = label("Aperçu", "H2")
        vh.addWidget(self.view_title)
        vh.addStretch()
        self.metric_lb = label("", "Muted")
        vh.addWidget(self.metric_lb)
        vl.addLayout(vh)
        self.wave = WaveView()
        vl.addWidget(self.wave, 1)
        ph = QHBoxLayout()
        legend = label("<span style='color:#a78bfa'>●</span> original&nbsp;&nbsp;&nbsp;"
                       "<span style='color:#5eead4'>●</span> sec", "Hint")
        legend.setTextFormat(Qt.RichText)
        ph.addWidget(legend)
        ph.addStretch()
        self.play_wet = button("▶ Original", "Ghost", lambda: self.play(False))
        self.play_dry = button("▶ Sec", "Ghost", lambda: self.play(True))
        self.stop_b = button("■", "Ghost", self.player.stop)
        for b in (self.play_wet, self.play_dry, self.stop_b):
            b.setEnabled(self.player.available)
            ph.addWidget(b)
        if not self.player.available:
            self.play_wet.setToolTip("Installez « sounddevice » pour écouter dans l'application")
        vl.addLayout(ph)
        split.addWidget(viewer)
        split.setSizes([360, 260])
        main.addWidget(split, 1)
        root.addLayout(main, 1)

        # ---- panneau de réglages
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 6, 0)
        sl.setSpacing(14)

        mcard = Card("Méthode")
        self.method = Segmented({"room": "Salle", "inverse": "Inverse", "iterative": "Itérative", "neural": "Neuro"})
        self.method.changed.connect(self.on_method)
        mcard.add(self.method)
        self.method_hint = mcard.add(label("", "Hint", True))
        sl.addWidget(mcard)

        self.room_card = Card("Salle")
        f = Form()
        self.preset = f.row("Situation", QComboBox())
        self.preset.addItems(["Ensemble / orchestre", "Soliste immobile", "Personnalisé"])
        self.late = f.row("Garder les", dspin(60, 0, 300, 5, 0, "ms"),
                          "Durée après le son direct conservée (son direct + premières réflexions).\n"
                          "Seule la réverbération au-delà est retirée. 40–80 ms : naturel ; "
                          "10–20 ms : retire aussi les réflexions d'une petite pièce.")
        self.strength = f.row("Force", dspin(1.0, 0.1, 4, 0.1, 2),
                              "1 = quantité de réverbération prédite par l'IR. Plus = plus agressif.")
        self.floor = f.row("Atténuation max", dspin(-18, -60, -3, 1, 0, "dB"),
                           "Limite la réduction : protège le timbre et évite les artefacts.")
        self.xmode = f.row("Grave exact", QComboBox(),
                           "Sous la fréquence de transition, l'IR est inversée exactement\n"
                           "(corrige les modes de la pièce). Valable seulement si la source\n"
                           "était à l'endroit de la mesure. « Auto » utilise la cohérence mesurée\n"
                           "en combinant plusieurs captures (page Capture).")
        self.xmode.addItems(["Désactivé", "Auto", "Manuel"])
        self.xfreq = f.row("Transition", dspin(600, 40, 8000, 50, 0, "Hz"))
        self.link = f.full(QCheckBox("Canaux liés (paire stéréo)"))
        self.link.setChecked(True)
        self.room_card.add(f)
        sl.addWidget(self.room_card)
        self.preset.currentIndexChanged.connect(self.apply_preset)
        self.xmode.currentIndexChanged.connect(lambda i: self.xfreq.setEnabled(i == 2))
        for w in (self.late, self.strength, self.floor, self.xfreq):
            w.valueChanged.connect(self.to_custom)
        self.xmode.currentIndexChanged.connect(self.to_custom)

        self.mix_card = Card("Modèle de la piste")
        f = Form()
        self.dry = f.row("Part directe", dspin(0.0, 0, 4, 0.05, 3),
                         "Niveau du signal sec mélangé à la réverbération dans la piste "
                         "(0 = réverbération 100 % wet).")
        self.wet = f.row("Gain de l'IR", dspin(1.0, 0, 16, 0.05, 3),
                         "Gain appliqué à l'IR dans la piste (send / niveau de retour).")
        self.target = f.row("Son direct", dspin(2.0, 0, 50, 0.5, 1, "ms"),
                            "IR mesurée (salle, haut-parleur) : 1 à 5 ms. Le résultat garde le son direct\n"
                            "tel que le micro l'entend : la couleur du haut-parleur de mesure n'est pas\n"
                            "inversée et le calage temporel est conservé.\n"
                            "0 = impulsion parfaite : pour une IR numérique exacte (reverb à convolution).")
        self.mix_card.add(f)
        sl.addWidget(self.mix_card)

        self.reg_card = Card("Régularisation")
        f = Form()
        self.reg = f.row("Tikhonov", dspin(-60, -200, 0, 5, 0, "dB"),
                         "Plus bas = plus exact mais plus sensible au bruit.\n"
                         "−100 dB et moins : piste rendue numériquement avec exactement cette IR.\n"
                         "−30 à −50 dB : enregistrement réel, bruit de fond.")
        self.fmin = f.row("Fréq. min", dspin(0, 0, 2000, 5, 0, "Hz"), "Sous cette fréquence, l'IR est jugée peu fiable (0 = aucune limite).")
        self.fmax = f.row("Fréq. max", dspin(0, 0, 96000, 100, 0, "Hz"), "Au-dessus, l'IR est jugée peu fiable (0 = Nyquist).")
        self.reg_card.add(f)
        sl.addWidget(self.reg_card)

        self.it_card = Card("Optimisation")
        f = Form()
        self.iters = f.row("Itérations", ispin(200, 10, 5000, 50))
        self.sparsity = f.row("Parcimonie λ", dspin(0.01, 0, 1, 0.005, 3),
                              "Force de l'a priori « signal sec » : réduit bruit et artefacts.\n"
                              "0 = moindres carrés purs (optimal pour une IR exacte sans bruit).")
        self.it_card.add(f)
        sl.addWidget(self.it_card)

        self.nn_card = Card("Modèle neuronal")
        self.model_field = PathField("Aucun modèle", "file", "Modèle (*.pt)")
        self.nn_card.add(self.model_field)
        self.nn_card.add(label("Le modèle contient sa propre IR : l'IR assignée aux pistes est ignorée.",
                               "Hint", True))
        sl.addWidget(self.nn_card)

        ocard = Card("Sortie")
        f = Form()
        self.out_dir = f.full(PathField("À côté des fichiers source", "dir"))
        self.fmt = QComboBox()
        self.fmt.addItems(["WAV 32 bits flottant", "WAV 24 bits"])
        f.row("Format", self.fmt)
        self.suffix = f.row("Suffixe", QLineEdit("_dry"))
        ocard.add(f)
        sl.addWidget(ocard)

        sl.addStretch()
        for cb in side.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(6)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        col = QWidget()
        col.setFixedWidth(350)
        cl = QVBoxLayout(col)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)
        cl.addWidget(scroll, 1)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setVisible(False)
        cl.addWidget(self.prog)
        row = QHBoxLayout()
        self.cancel_b = button("Annuler", None, self.cancel)
        self.cancel_b.setVisible(False)
        self.go = button("Traiter", "Primary", self.run)
        self.go.setMinimumHeight(38)
        self.go.setToolTip("Traite les pistes sélectionnées, ou toutes si aucune n'est sélectionnée.")
        row.addWidget(self.cancel_b)
        row.addWidget(self.go, 1)
        cl.addLayout(row)
        root.addWidget(col)

        self.lib.changed.connect(self.refresh_ir_combos)
        self._presetting = False
        self.apply_preset(0)
        self.on_method("room")

    # ------------------------------------------------------------ pistes
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        files = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                files += [q for q in pp.rglob("*") if q.suffix.lower() in AUDIO_EXT]
            elif pp.suffix.lower() in AUDIO_EXT:
                files.append(pp)
        self.add_tracks(files)

    def add_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Ajouter des pistes", "", FILTER)
        self.add_tracks(files)

    def add_tracks(self, files):
        errors = []
        for p in files:
            try:
                a = load(p)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{Path(p).name} : {e}")
                continue
            t = Track(a, ir=self.guess_ir(a))
            self.tracks.append(t)
            self.add_row(t)
        if errors:
            QMessageBox.warning(self, "Import", "\n".join(errors))
        if self.tracks and not self.table.selectedItems():
            self.table.selectRow(len(self.tracks) - 1)
        self.status(f"{len(self.tracks)} piste(s)")

    def guess_ir(self, a: Audio) -> str | None:
        names = self.lib.names()
        stem = (a.path.stem if a.path else "").lower()
        for n in names:
            if n.lower() in stem or (stem and stem in n.lower()):
                return n
        return names[0] if len(names) == 1 else None

    def add_row(self, t: Track):
        r = self.table.rowCount()
        self.table.insertRow(r)
        a = t.audio
        self.table.setItem(r, 0, QTableWidgetItem(a.path.name if a.path else "piste"))
        ch = {1: "mono", 2: "stéréo"}.get(a.channels, f"{a.channels} ch")
        self.table.setItem(r, 1, QTableWidgetItem(f"{a.sr / 1000:g} kHz · {ch}"))
        m, s = divmod(a.duration, 60)
        self.table.setItem(r, 2, QTableWidgetItem(f"{int(m)}:{s:04.1f}"))
        cb = QComboBox()
        cb.currentTextChanged.connect(lambda txt, t=t: setattr(t, "ir", None if txt == NONE else txt))
        self.table.setCellWidget(r, 3, cb)
        self.fill_combo(cb, t)
        self.table.setItem(r, 4, QTableWidgetItem(t.status))

    def fill_combo(self, cb: QComboBox, t: Track):
        cb.blockSignals(True)
        cb.clear()
        cb.addItems([NONE] + self.lib.names())
        cb.setCurrentText(t.ir if t.ir in self.lib.items else NONE)
        cb.blockSignals(False)
        t.ir = t.ir if t.ir in self.lib.items else None

    def refresh_ir_combos(self):
        for r, t in enumerate(self.tracks):
            if t.ir is None:
                t.ir = self.guess_ir(t.audio)
            self.fill_combo(self.table.cellWidget(r, 3), t)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            self.tracks.pop(r)
        self.on_select()

    def load_ir(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Charger des IR", "", FILTER)
        for p in files:
            try:
                self.lib.load(p)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "IR", f"{Path(p).name} : {e}")

    def set_status(self, r: int, text: str):
        if r < self.table.rowCount():
            self.table.item(r, 4).setText(text)

    # ------------------------------------------------------------ aperçu
    def current(self) -> tuple[int, Track] | tuple[None, None]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return None, None
        return rows[0], self.tracks[rows[0]]

    def on_select(self):
        r, t = self.current()
        if t is None:
            self.wave.display(None, None, 48000)
            self.view_title.setText("Aperçu")
            self.metric_lb.setText("")
            return
        self.view_title.setText(t.audio.path.name if t.audio.path else "Aperçu")
        dry = t.result.data if t.result else None
        self.wave.display(t.audio.data, dry, t.audio.sr)
        if t.result:
            res = max(t.result.residual_db)
            self.metric_lb.setText(f"{t.result.metric}  <b style='color:#5eead4'>{res:.1f} dB</b>")
            self.metric_lb.setTextFormat(Qt.RichText)
            self.metric_lb.setToolTip(
                "Énergie du résultat par rapport à la piste : la différence est la réverbération retirée." if t.result.metric.startswith("Énergie") else
                "‖sec ∗ IR − original‖ / ‖original‖ : mesure à quel point l'estimation sèche, reconvoluée "
                "par l'IR, retrouve la piste. Très bas (< −40 dB) = IR cohérente avec la piste.")
        else:
            self.metric_lb.setText("")

    def play(self, dry: bool):
        _, t = self.current()
        if t is None:
            return
        if dry and t.result is None:
            self.status("Pas encore de résultat pour cette piste.")
            return
        self.player.play(t.result.data if dry else t.audio.data, t.audio.sr)

    # ------------------------------------------------------------ réglages
    def on_method(self, m: str):
        hints = {
            "room": "Retire la réverbération tardive prédite par l'IR, sans exiger que la source soit "
                    "exactement là où l'IR a été mesurée. Robuste : orchestre, soliste, prise réelle.",
            "inverse": "Déconvolution spectrale sur toute la piste. Instantanée et quasi exacte si la "
                       "piste a été produite avec exactement cette IR.",
            "iterative": "Optimisation L-BFGS : gère les queues coupées et ajoute un a priori de signal "
                         "sec qui limite bruit et artefacts. Plus lent.",
            "neural": "Réseau entraîné pour une IR (page « Modèle »). Le plus robuste quand l'IR "
                      "mesurée ne colle pas parfaitement à la piste.",
        }
        self.method_hint.setText(hints[m])
        self.room_card.setVisible(m == "room")
        self.mix_card.setVisible(m in ("inverse", "iterative"))
        self.reg_card.setVisible(m in ("inverse", "iterative"))
        self.it_card.setVisible(m == "iterative")
        self.nn_card.setVisible(m == "neural")

    def settings(self) -> engine.ProcessSettings:
        return engine.ProcessSettings(
            method=self.method.value(),
            inverse=InverseParams(reg_db=self.reg.value(), fmin=self.fmin.value(),
                                  fmax=self.fmax.value(), dry=self.dry.value(), wet=self.wet.value()),
            iterative=IterativeParams(iterations=self.iters.value(), sparsity=self.sparsity.value()),
            room=RoomParams(late_ms=self.late.value(), strength=self.strength.value(),
                            floor_db=self.floor.value(), link=self.link.isChecked(),
                            crossover_hz=[0.0, -1.0, self.xfreq.value()][self.xmode.currentIndex()]),
            target_ms=self.target.value(),
            model_path=self.model_field.path,
            output_dir=self.out_dir.path,
            suffix=self.suffix.text() or "_dry",
            subtype="FLOAT" if self.fmt.currentIndex() == 0 else "PCM_24",
        )

    PRESETS = {0: (60, 1.0, -18, 0), 1: (15, 1.2, -24, 1)}

    def apply_preset(self, i: int):
        if i not in self.PRESETS:
            return
        late, strength, floor, xm = self.PRESETS[i]
        self._presetting = True
        self.late.setValue(late)
        self.strength.setValue(strength)
        self.floor.setValue(floor)
        self.xmode.setCurrentIndex(xm)
        self.xfreq.setEnabled(xm == 2)
        self._presetting = False

    def to_custom(self, *_):
        if not getattr(self, "_presetting", True) and self.preset.currentIndex() != 2:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(2)
            self.preset.blockSignals(False)

    def set_model(self, path: str):
        self.model_field.set(path)
        self.method.set("neural")

    # ------------------------------------------------------------ traitement
    def run(self):
        if not self.tracks:
            self.status("Ajoutez d'abord des pistes.")
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}) or list(range(len(self.tracks)))
        s = self.settings()
        if s.method != "neural":
            missing = [self.tracks[r].audio.path.name for r in rows if self.tracks[r].ir is None]
            if missing:
                QMessageBox.information(self, "IR manquante",
                                        "Assignez une IR à :\n" + "\n".join(missing))
                return
        elif not s.model_path:
            QMessageBox.information(self, "Modèle", "Choisissez un modèle neuronal (.pt).")
            return
        jobs = [(r, self.tracks[r], self.lib.get(self.tracks[r].ir)) for r in rows]

        def task(w: Worker):
            n = len(jobs)
            for k, (r, t, ir) in enumerate(jobs):
                w.report.emit(("status", r, "Traitement…"))
                res = engine.process(t.audio, ir, s,
                                     progress=lambda f, k=k: w.progress.emit((k + f) / n),
                                     cancelled=w.cancelled)
                path = engine.write(res, engine.output_path(t.audio, s), s.subtype)
                w.report.emit(("result", r, res))
                clip = "  ⚠ crête > 0 dBFS" if res.peak > 1 and s.subtype != "FLOAT" else ""
                tag = "niveau " if res.metric.startswith("Énergie") else ""
                w.report.emit(("status", r, f"✓ {tag}{max(res.residual_db):.0f} dB{clip}"))
                w.message.emit(f"Écrit : {path}")
            return len(jobs)

        self.worker = Worker(task, self)
        self.worker.progress.connect(lambda f: self.prog.setValue(int(f * 1000)))
        self.worker.message.connect(self.status)
        self.worker.report.connect(self.on_report)
        self.worker.done.connect(lambda n: self.finish(f"{n} piste(s) traitée(s)."))
        self.worker.failed.connect(self.on_failed)
        self.busy(True)
        self.worker.start()

    def on_report(self, rep):
        kind, r, val = rep
        if kind == "status":
            self.set_status(r, val)
        elif kind == "result":
            self.tracks[r].result = val
            if self.current()[0] == r:
                self.on_select()

    def on_failed(self, msg):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 4).text() == "Traitement…":
                self.set_status(r, "Erreur" if msg != "Annulé." else "Annulé")
        self.finish(msg)
        if msg != "Annulé.":
            QMessageBox.warning(self, "Traitement", msg)

    def finish(self, msg):
        self.busy(False)
        self.status(msg)

    def busy(self, b: bool):
        self.go.setEnabled(not b)
        self.prog.setVisible(b)
        self.prog.setValue(0)
        self.cancel_b.setVisible(b)

    def cancel(self):
        if self.worker:
            self.worker.cancel()

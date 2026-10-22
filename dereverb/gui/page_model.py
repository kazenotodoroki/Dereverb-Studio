"""Page « Modèle » : entraînement d'un réseau pour une IR."""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QMessageBox, QPlainTextEdit,
                               QProgressBar, QSlider, QVBoxLayout, QWidget)

from ..core.iterative import pick_device
from .state import IRLibrary
from .widgets import Card, Form, LossView, PathField, Worker, button, dspin, ispin, label


class ModelPage(QWidget):
    def __init__(self, lib: IRLibrary, status, on_model_ready):
        super().__init__()
        self.lib = lib
        self.status = status
        self.on_model_ready = on_model_ready
        self.worker: Worker | None = None
        self.trainer = None

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(label("Modèle", "H1"))
        left.addWidget(label("Comme NAM pour un ampli, on entraîne un réseau dédié à UNE réverbération. "
                             "Il part de la méthode inverse et apprend à en corriger les erreurs.",
                             "Hint", True))
        c = Card("Données")
        f = Form()
        self.ir = f.row("IR", QComboBox())
        self.dry = f.full(PathField("Sons secs : générateur de synthèse", "dir"))
        f.full(label("Sans sons secs (prises classiques, archives), le générateur de synthèse est peu "
                     "représentatif : préférez la méthode « Salle ».", "Hint", True))
        self.dry.setToolTip("Dossier de sons secs proches de vos pistes (voix, batterie…). "
                            "Fortement recommandé pour la qualité.")
        c.add(f)
        left.addWidget(c)

        c = Card("Entraînement")
        f = Form()
        self.rob = QSlider(Qt.Horizontal)
        self.rob.setRange(0, 100)
        self.rob.setValue(50)
        f.row("Robustesse", self.rob,
              "0 : la piste a été produite avec exactement cette IR (reverb numérique).\n"
              "100 : salle réelle, la reverb de la piste peut s'écarter de l'IR mesurée.")
        self.rob_lb = f.full(label("", "Hint"))
        self.rob.valueChanged.connect(self.update_rob)
        self.noise = f.row("Bruit des pistes", dspin(-70, -120, -20, 5, 0, "dB"))
        self.steps = f.row("Pas", ispin(3000, 50, 200000, 500))
        self.batch = f.row("Lot", ispin(8, 1, 64))
        self.seg = f.row("Segment", dspin(4, 1, 12, 0.5, 1, "s"))
        self.sr = QComboBox()
        self.sr.addItems(["44100", "48000", "96000"])
        self.sr.setCurrentText("48000")
        f.row("Fréquence", self.sr, "Taux de travail du modèle ; les pistes sont rééchantillonnées.")
        c.add(f)
        dev = pick_device().type.upper()
        c.add(label(f"Calcul : {dev}" + ("  — un GPU accélère fortement l'entraînement" if dev == "CPU" else ""),
                    "Hint", True))
        left.addWidget(c)
        left.addStretch()
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(330)
        root.addWidget(lw)

        right = QVBoxLayout()
        right.setSpacing(12)
        h = QHBoxLayout()
        h.addWidget(label("Progression", "H2"))
        h.addStretch()
        h.addWidget(button("Charger un modèle…", "Ghost", self.load_model))
        self.save_b = button("Enregistrer…", None, self.save_model)
        self.save_b.setEnabled(False)
        h.addWidget(self.save_b)
        self.stop_b = button("Arrêter", None, self.stop)
        self.stop_b.setVisible(False)
        h.addWidget(self.stop_b)
        self.go = button("Entraîner", "Primary", self.train)
        h.addWidget(self.go)
        right.addLayout(h)

        mc = Card()
        mh = QHBoxLayout()
        self.m_step = self._metric(mh, "Pas")
        self.m_val = self._metric(mh, "SNR modèle")
        self.m_base = self._metric(mh, "SNR inverse")
        self.m_eta = self._metric(mh, "Restant")
        mh.addStretch()
        mc.add(mh)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        mc.add(self.prog)
        self.loss = LossView()
        mc.add(self.loss)
        mc.add(label("<span style='color:#5eead4'>●</span> modèle&nbsp;&nbsp;&nbsp;"
                     "<span style='color:#7d8594'>- -</span> méthode inverse seule "
                     "(même données de validation)", "Hint"))
        right.addWidget(mc, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        right.addWidget(self.log, 1)
        root.addLayout(right, 1)

        self.lib.changed.connect(self.refresh)
        self.refresh()
        self.update_rob()

    def _metric(self, lay, title):
        box = QVBoxLayout()
        box.setSpacing(2)
        box.addWidget(label(title, "H2"))
        v = label("—", "Metric")
        box.addWidget(v)
        lay.addLayout(box)
        lay.addSpacing(26)
        return v

    def update_rob(self):
        v = self.rob.value()
        txt = ("IR exacte (reverb numérique)" if v < 15 else
               "Légers écarts" if v < 45 else "Écarts modérés" if v < 75 else "Salle réelle, fortes variations")
        self.rob_lb.setText(txt)

    def refresh(self):
        cur = self.ir.currentText()
        self.ir.clear()
        self.ir.addItems(self.lib.names())
        if cur in self.lib.items:
            self.ir.setCurrentText(cur)

    def write(self, msg: str):
        self.log.appendPlainText(msg)

    # ------------------------------------------------------------ entraînement
    def train(self):
        ir = self.lib.get(self.ir.currentText())
        if ir is None:
            QMessageBox.information(self, "Modèle", "Ajoutez une IR (page Capture ou Pistes) avant d'entraîner.")
            return
        from ..core.neural import NeuralConfig, Trainer
        cfg = NeuralConfig(sr=int(self.sr.currentText()), steps=self.steps.value(), batch=self.batch.value(),
                           seg_seconds=self.seg.value(), robustness=self.rob.value() / 100,
                           noise_db=self.noise.value())
        h = ir.data.mean(axis=1)
        dry = self.dry.path
        self.loss.reset()
        self.log.clear()
        self.ir_name = self.ir.currentText()

        def task(w: Worker):
            tr = Trainer(h, ir.sr, cfg, dry, log=w.message.emit)
            self.trainer = tr
            tr.train(progress=w.report.emit, cancelled=w.cancelled)
            return tr

        self.worker = Worker(task, self)
        self.worker.message.connect(self.write)
        self.worker.report.connect(self.on_report)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.t0 = time.time()
        self.busy(True)
        self.worker.start()

    def on_report(self, r):
        total = self.steps.value()
        self.prog.setValue(int(r.step / total * 1000))
        self.m_step.setText(f"{r.step}")
        if r.step > 5:
            eta = r.elapsed / r.step * (total - r.step)
            self.m_eta.setText(f"{int(eta // 60)} min" if eta > 90 else f"{int(eta)} s")
        if r.val_snr is not None:
            self.loss.add(r.step, r.val_snr, r.base_snr)
            self.m_val.setText(f"{r.val_snr:.1f} dB")
            self.m_base.setText(f"{r.base_snr:.1f} dB")
            if r.step % 500 == 0:
                self.write(f"pas {r.step:>6}  ·  modèle {r.val_snr:6.2f} dB  ·  inverse {r.base_snr:6.2f} dB")

    def on_done(self, tr):
        self.busy(False)
        self.save_b.setEnabled(True)
        vals = [r for r in tr.history if r.val_snr is not None]
        if vals and vals[-1].val_snr <= vals[-1].base_snr + 0.3:
            self.write(f"⚠ Le modèle ({vals[-1].val_snr:.1f} dB) ne dépasse pas la méthode inverse "
                       f"({vals[-1].base_snr:.1f} dB) : entraînez plus longtemps, fournissez des sons secs "
                       "proches de vos pistes, ou utilisez la méthode Itérative.")
        self.write("Terminé. Enregistrez le modèle pour l'utiliser sur vos pistes.")
        self.status("Entraînement terminé.")
        self.save_model()

    def on_failed(self, msg):
        self.busy(False)
        self.write(msg)
        if self.trainer and self.trainer.history:
            self.save_b.setEnabled(True)

    def stop(self):
        if self.worker:
            self.worker.cancel()

    def busy(self, b):
        self.go.setVisible(not b)
        self.stop_b.setVisible(b)
        self.m_eta.setText("—")

    def save_model(self):
        if not self.trainer:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le modèle", f"{self.ir_name}.pt",
                                              "Modèle (*.pt)")
        if path:
            self.trainer.save(path)
            self.write(f"Modèle enregistré : {path}")
            self.on_model_ready(path)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un modèle", "", "Modèle (*.pt)")
        if path:
            self.on_model_ready(path)
            self.status("Modèle sélectionné dans « Pistes ».")

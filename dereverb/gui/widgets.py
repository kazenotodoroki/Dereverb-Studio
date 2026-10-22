"""Composants d'interface réutilisables."""
from __future__ import annotations

import traceback
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (QButtonGroup, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from . import theme

pg.setConfigOptions(antialias=True, background=theme.SURFACE, foreground=theme.MUTED)


# ---------------------------------------------------------------- tâches de fond

class Worker(QThread):
    """Exécute fn(ctx) dans un thread. ctx fournit progress(), message(), cancelled()."""
    progress = Signal(float)
    message = Signal(str)
    report = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, parent: QObject | None = None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except InterruptedError:
            self.failed.emit("Annulé.")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


# ---------------------------------------------------------------- mise en page

def label(text: str, obj: str | None = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


class Card(QFrame):
    def __init__(self, title: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 16, 18, 18)
        self.lay.setSpacing(12)
        if title:
            self.lay.addWidget(label(title, "H2"))

    def add(self, w):
        if isinstance(w, QWidget):
            self.lay.addWidget(w)
        else:
            self.lay.addLayout(w)
        return w


class Form(QGridLayout):
    """Grille libellé / champ, compacte."""
    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(12)
        self.setVerticalSpacing(9)
        self.setColumnStretch(1, 1)
        self._r = 0

    def row(self, text: str, w: QWidget, tip: str | None = None):
        lb = label(text, "Muted")
        if tip:
            lb.setToolTip(tip)
            w.setToolTip(tip)
        self.addWidget(lb, self._r, 0)
        self.addWidget(w, self._r, 1)
        self._r += 1
        return w

    def full(self, w: QWidget):
        self.addWidget(w, self._r, 0, 1, 2)
        self._r += 1
        return w


def dspin(v: float, lo: float, hi: float, step: float = 1.0, dec: int = 1, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(v)
    s.setButtonSymbols(QDoubleSpinBox.NoButtons)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setAlignment(Qt.AlignRight)
    return s


def ispin(v: int, lo: int, hi: int, step: int = 1, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(v)
    s.setButtonSymbols(QSpinBox.NoButtons)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setAlignment(Qt.AlignRight)
    return s


def button(text: str, kind: str | None = None, cb: Callable | None = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    if cb:
        b.clicked.connect(cb)
    return b


class Segmented(QFrame):
    changed = Signal(str)

    def __init__(self, options: dict[str, str]):
        super().__init__()
        self.setObjectName("SegBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.keys = list(options)
        for i, (k, t) in enumerate(options.items()):
            b = QPushButton(t)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b, 1)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(lambda i: self.changed.emit(self.keys[i]))

    def value(self) -> str:
        return self.keys[self.group.checkedId()]

    def set(self, key: str):
        self.group.button(self.keys.index(key)).setChecked(True)
        self.changed.emit(key)


class PathField(QWidget):
    changed = Signal(str)

    def __init__(self, placeholder: str, mode: str = "dir", filt: str = "", save: bool = False):
        super().__init__()
        self.mode, self.filt, self.save = mode, filt, save
        self.path: str | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.lb = label(placeholder, "Hint")
        self.lb.setMinimumWidth(40)
        self.placeholder = placeholder
        lay.addWidget(self.lb, 1)
        lay.addWidget(button("Choisir…", "Ghost", self.pick))
        self.clear_b = button("×", "Ghost", self.clear)
        self.clear_b.setVisible(False)
        lay.addWidget(self.clear_b)

    def pick(self):
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        elif self.save:
            p, _ = QFileDialog.getSaveFileName(self, "Enregistrer", "", self.filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Ouvrir", "", self.filt)
        if p:
            self.set(p)

    def set(self, p: str | None):
        self.path = p
        if p:
            name = p.replace("\\", "/").rstrip("/").split("/")[-1]
            self.lb.setText(name)
            self.lb.setToolTip(p)
            self.lb.setObjectName("")
        else:
            self.lb.setText(self.placeholder)
            self.lb.setObjectName("Hint")
        self.lb.style().unpolish(self.lb)
        self.lb.style().polish(self.lb)
        self.clear_b.setVisible(bool(p))
        self.changed.emit(p or "")

    def clear(self):
        self.set(None)


# ---------------------------------------------------------------- graphiques

def _plot(height: int = 180) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setMinimumHeight(height)
    w.setMenuEnabled(False)
    w.hideButtons()
    pi = w.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.08)
    for ax in ("left", "bottom"):
        a = pi.getAxis(ax)
        a.setPen(pg.mkPen(theme.BORDER))
        a.setTextPen(pg.mkPen(theme.MUTED))
    w.setStyleSheet(f"border: 1px solid {theme.BORDER}; border-radius: 10px;")
    return w


class WaveView(QWidget):
    """Original (violet) et traité (menthe) superposés."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = _plot(170)
        self.plot.getPlotItem().getAxis("left").setWidth(36)
        self.plot.setLabel("bottom", "s")
        self.plot.setMouseEnabled(x=True, y=False)
        self.wet = self.plot.plot(pen=pg.mkPen(theme.WET, width=1))
        self.dry = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=1))
        for c in (self.wet, self.dry):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
        lay.addWidget(self.plot)

    def display(self, wet: np.ndarray | None, dry: np.ndarray | None, sr: int):
        for curve, x in ((self.wet, wet), (self.dry, dry)):
            if x is None:
                curve.clear()
            else:
                m = x.mean(axis=1) if x.ndim == 2 else x
                curve.setData(np.arange(len(m)) / sr, m)
        self.plot.enableAutoRange()


class IRView(QWidget):
    """Enveloppe énergétique (dB) et courbe de décroissance de Schroeder."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = _plot(200)
        self.plot.setLabel("bottom", "s")
        self.plot.setLabel("left", "dB")
        self.plot.setYRange(-100, 3)
        self.plot.setMouseEnabled(x=True, y=False)
        self.env = self.plot.plot(pen=pg.mkPen(theme.WET, width=1))
        self.edc = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        lay.addWidget(self.plot)

    def display(self, ir: np.ndarray | None, sr: int = 48000):
        if ir is None:
            self.env.clear()
            self.edc.clear()
            return
        from ..core.capture import edc_db
        h = ir.mean(axis=1) if ir.ndim == 2 else ir
        e = h ** 2
        w = max(int(0.002 * sr), 1)
        env = np.convolve(e, np.ones(w) / w, "same")
        env_db = 10 * np.log10(env / (env.max() + 1e-30) + 1e-12)
        t = np.arange(len(h)) / sr
        step = max(len(h) // 6000, 1)
        self.env.setData(t[::step], env_db[::step])
        self.edc.setData(t[::step], edc_db(h)[::step])
        self.plot.setXRange(0, t[-1])
        self.plot.setYRange(-100, 3)


class LossView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = _plot(200)
        self.plot.setLabel("left", "SNR validation (dB)")
        self.plot.setLabel("bottom", "pas")
        self.base = self.plot.plot(pen=pg.mkPen(theme.MUTED, width=1, style=Qt.DashLine))
        self.val = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=2), symbol="o", symbolSize=4,
                                  symbolBrush=theme.ACCENT, symbolPen=None)
        lay.addWidget(self.plot)
        self.reset()

    def reset(self):
        self.xs, self.vs, self.bs = [], [], []
        self.base.clear()
        self.val.clear()

    def add(self, step: int, val: float, base: float):
        self.xs.append(step)
        self.vs.append(val)
        self.bs.append(base)
        self.val.setData(self.xs, self.vs)
        self.base.setData(self.xs, self.bs)


# ---------------------------------------------------------------- écoute

class Player:
    """Lecture via sounddevice si disponible."""

    def __init__(self):
        try:
            import sounddevice as sd  # noqa: F401
            self.sd = sd
        except Exception:  # noqa: BLE001
            self.sd = None

    @property
    def available(self) -> bool:
        return self.sd is not None

    def play(self, data: np.ndarray, sr: int):
        if not self.sd:
            return
        self.sd.stop()
        peak = np.abs(data).max()
        self.sd.play(data / peak * 0.9 if peak > 1 else data, sr)

    def stop(self):
        if self.sd:
            self.sd.stop()

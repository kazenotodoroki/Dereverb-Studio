"""Fenêtre principale."""
from __future__ import annotations

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QStackedWidget, QStatusBar, QVBoxLayout, QWidget)

from . import theme
from .page_capture import CapturePage
from .page_model import ModelPage
from .page_tracks import TracksPage
from .state import IRLibrary


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dereverb Studio")
        self.resize(1360, 860)
        self.setMinimumSize(QSize(1100, 700))
        self.lib = IRLibrary()

        root = QWidget()
        root.setObjectName("Root")
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        nav_box = QWidget()
        nav_box.setFixedWidth(196)
        nl = QVBoxLayout(nav_box)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(0)
        brand = QLabel("dereverb<span style='color:#5eead4'>.</span>")
        brand.setObjectName("Brand")
        brand.setTextFormat(Qt.RichText)
        nl.addWidget(brand)
        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        for t in ("Pistes", "Capture", "Modèle"):
            self.nav.addItem(t)
        nl.addWidget(self.nav, 1)
        self.lib_label = QLabel()
        self.lib_label.setObjectName("Hint")
        self.lib_label.setContentsMargins(22, 0, 12, 18)
        self.lib_label.setWordWrap(True)
        nl.addWidget(self.lib_label)
        lay.addWidget(nav_box)

        self.stack = QStackedWidget()
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        msg = lambda m: self.sb.showMessage(m, 8000)  # noqa: E731
        self.tracks = TracksPage(self.lib, msg)
        self.capture = CapturePage(self.lib, msg)
        self.model = ModelPage(self.lib, msg, self.tracks.set_model)
        for p in (self.tracks, self.capture, self.model):
            self.stack.addWidget(p)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.lib.changed.connect(self.update_lib)
        self.update_lib()

    def update_lib(self):
        n = len(self.lib.items)
        self.lib_label.setText("Bibliothèque : aucune IR" if n == 0 else
                               f"Bibliothèque : {n} IR<br><span style='color:#4b5260'>"
                               + "<br>".join(self.lib.names()[:6]) + "</span>")


def run():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Dereverb Studio")
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, theme.BG), (QPalette.Base, theme.BG), (QPalette.Text, theme.TEXT),
                      (QPalette.WindowText, theme.TEXT), (QPalette.Button, theme.SURFACE_2),
                      (QPalette.ButtonText, theme.TEXT), (QPalette.Highlight, theme.ACCENT),
                      (QPalette.HighlightedText, "#04201b"), (QPalette.ToolTipBase, theme.SURFACE_2),
                      (QPalette.ToolTipText, theme.TEXT), (QPalette.PlaceholderText, theme.MUTED)):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    f = QFont()
    f.setFamilies(["Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial"])
    f.setPointSizeF(10)
    app.setFont(f)
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

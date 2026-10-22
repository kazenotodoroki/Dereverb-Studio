"""Thème sombre minimaliste."""
from pathlib import Path

CHEVRON = (Path(__file__).parent / "assets" / "chevron.svg").as_posix()

BG = "#0c0e11"
SURFACE = "#13161b"
SURFACE_2 = "#1a1e24"
BORDER = "#252a32"
TEXT = "#e7e9ec"
MUTED = "#7d8594"
ACCENT = "#5eead4"      # signal sec / actions
WET = "#a78bfa"         # signal réverbéré
WARN = "#fbbf24"
ERROR = "#f87171"

FONT_STACK = '"Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif'

QSS = f"""
* {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {TEXT};
    outline: none;
}}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QWidget {{ background: transparent; }}

/* --- navigation --- */
QListWidget#Nav {{
    background: {BG};
    border: none;
    border-right: 1px solid {BORDER};
    padding: 8px 10px;
}}
QListWidget#Nav::item {{
    padding: 9px 12px;
    border-radius: 7px;
    color: {MUTED};
    margin: 1px 0;
}}
QListWidget#Nav::item:hover {{ background: {SURFACE}; color: {TEXT}; }}
QListWidget#Nav::item:selected {{ background: {SURFACE_2}; color: {TEXT}; }}
QLabel#Brand {{ font-size: 15px; font-weight: 600; letter-spacing: 0.5px; padding: 18px 22px 14px 22px; }}
QLabel#BrandDot {{ color: {ACCENT}; }}

/* --- textes --- */
QLabel#H1 {{ font-size: 20px; font-weight: 600; }}
QLabel#H2 {{ font-size: 11px; font-weight: 600; color: {MUTED}; letter-spacing: 1.2px; text-transform: uppercase; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Metric {{ font-size: 18px; font-weight: 600; color: {ACCENT}; }}
QLabel#Hint {{ color: {MUTED}; font-size: 12px; }}

/* --- panneaux --- */
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

/* --- boutons --- */
QPushButton {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
}}
QPushButton:hover {{ border-color: #3a414d; background: #20252d; }}
QPushButton:pressed {{ background: #161a20; }}
QPushButton:disabled {{ color: #4b5260; border-color: #1e2228; }}
QPushButton#Primary {{
    background: {ACCENT};
    color: #04201b;
    border: none;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #7ff0dd; }}
QPushButton#Primary:disabled {{ background: #1f3b37; color: #55807a; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {MUTED}; padding: 6px 8px; }}
QPushButton#Ghost:hover {{ color: {TEXT}; }}
QPushButton#Seg {{
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    color: {MUTED};
}}
QPushButton#Seg:checked {{ background: {SURFACE_2}; color: {TEXT}; }}
QFrame#SegBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; }}

/* --- champs --- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 8px;
    min-height: 18px;
    selection-background-color: {ACCENT};
    selection-color: #04201b;
}}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: #4b5260; }}
QAbstractSpinBox QLineEdit {{ padding: 0; border: none; background: transparent; min-height: 0; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px; height: 0px; border: none; subcontrol-origin: border;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({CHEVRON}); width: 10px; height: 6px; margin-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px;
    selection-background-color: #262c35; padding: 4px;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid #3a414d; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

/* --- tables & listes --- */
QTableWidget, QListWidget#List, QPlainTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #1f2a2c;
    selection-color: {TEXT};
}}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid #181b20; }}
QListWidget#List::item {{ padding: 7px 8px; border-radius: 6px; }}
QListWidget#List::item:selected {{ background: #1f2a2c; }}
QHeaderView::section {{
    background: {SURFACE};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}
QPlainTextEdit {{ font-family: "JetBrains Mono", "SF Mono", Consolas, monospace; font-size: 12px; color: {MUTED}; padding: 8px; }}

QProgressBar {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 4px;
    height: 6px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2a3039; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2a3039; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: transparent; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; border-radius: 6px; }}
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QMessageBox {{ background: {SURFACE}; }}
QScrollArea {{ border: none; }}
"""

"""État partagé entre les pages : bibliothèque d'IR."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..core.audio import Audio, load


class IRLibrary(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.items: dict[str, Audio] = {}

    def unique(self, name: str) -> str:
        base, k = name, 2
        while name in self.items:
            name = f"{base} ({k})"
            k += 1
        return name

    def add(self, name: str, audio: Audio) -> str:
        name = self.unique(name)
        self.items[name] = audio
        self.changed.emit()
        return name

    def load(self, path: str | Path) -> str:
        a = load(path)
        side = Path(path).with_suffix(".ir.json")
        if side.exists():
            a.meta.update(json.loads(side.read_text()))
        return self.add(Path(path).stem, a)

    def remove(self, name: str):
        self.items.pop(name, None)
        self.changed.emit()

    def names(self) -> list[str]:
        return list(self.items)

    def get(self, name: str | None) -> Audio | None:
        return self.items.get(name) if name else None

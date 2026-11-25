"""Start workspace and keyboard command search for the desktop application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore, QtWidgets

ROLE = QtCore.Qt.ItemDataRole.UserRole


def label(text, kind=None):
    widget = QtWidgets.QLabel(text)
    widget.setWordWrap(True)
    if kind:
        widget.setProperty(kind, True)
    return widget


class WorkspaceHome(QtWidgets.QScrollArea):
    openRecent = QtCore.Signal(str)

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        self.setWidget(host)
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(18)
        layout.addWidget(label("SEISMIC PROCESSING WORKSPACE", "section"))
        layout.addWidget(label("Explore your next survey.", "heading"))
        layout.addWidget(label("Drop seismic files here, build a processing flow and inspect your subsurface models.", "muted"))
        metrics = QtWidgets.QHBoxLayout()
        self.metrics = []
        for text in ("Datasets", "Traces", "Pick sets"):
            card = QtWidgets.QFrame(); card.setProperty("card", True)
            box = QtWidgets.QVBoxLayout(card); box.setContentsMargins(18, 12, 18, 12)
            value = label("0", "metric"); self.metrics.append(value)
            box.addWidget(value); box.addWidget(label(text, "muted")); metrics.addWidget(card)
        layout.addLayout(metrics)
        cards = QtWidgets.QGridLayout(); cards.setSpacing(14)
        self.buttons = {}
        for i, (key, title, description, button_text) in enumerate((
            ("import", "01  Bring in your data", "SEG-Y, SU, SEG-2 and supported SEG-D records.", "Import seismic data"),
            ("project", "02  Continue a project", "Reopen datasets, picks, models and saved processing flows.", "Open project"),
            ("flow", "03  Build a workflow", "Search processes, start from a template and preview a gather.", "Open flow editor"),
            ("synthetic", "04  Explore an example", "Create a synthetic land or marine dataset to practise.", "Create example data"),
        )):
            card = QtWidgets.QFrame(); card.setProperty("card", True)
            box = QtWidgets.QVBoxLayout(card); box.setContentsMargins(18, 16, 18, 16)
            box.addWidget(label(title, "section")); box.addWidget(label(description, "muted")); box.addStretch()
            button = QtWidgets.QPushButton(button_text)
            if i == 0: button.setProperty("primary", True)
            button.clicked.connect(actions[key]); self.buttons[key] = button
            box.addWidget(button); cards.addWidget(card, i // 2, i % 2)
        layout.addLayout(cards)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label("RECENT PROJECTS", "section")); row.addStretch()
        for key, text in (("compute", "CPU / CUDA"), ("remote", "Remote jobs"), ("guide", "User guide")):
            button = QtWidgets.QPushButton(text); button.clicked.connect(actions[key]); row.addWidget(button)
        layout.addLayout(row)
        self.recent = QtWidgets.QListWidget()
        self.recent.setMaximumHeight(160)
        self.recent.itemActivated.connect(lambda item: self.openRecent.emit(item.data(ROLE)))
        self.empty_recent = label("Projects you open or save will appear here. Double-click one to reopen it.", "muted")
        layout.addWidget(self.empty_recent); layout.addWidget(self.recent)
        layout.addWidget(label("Ctrl+K  Find a command     •     Ctrl+O  Import data     •     Ctrl+S  Save project", "muted"))
        layout.addStretch()

    def refresh(self, project, recent):
        for widget, value in zip(self.metrics, (len(project.datasets), sum(d.ntr for d in project.datasets.values()),
                                                len(project.context.picks))):
            widget.setText(f"{value:,}")
        self.recent.clear()
        for path in recent:
            item = QtWidgets.QListWidgetItem(path)
            item.setData(ROLE, path); item.setToolTip(path); self.recent.addItem(item)
        self.recent.setVisible(bool(recent)); self.empty_recent.setVisible(not recent)


@dataclass
class Command:
    title: str
    detail: str
    run: Callable
    enabled: bool = True


class CommandPalette(QtWidgets.QDialog):
    def __init__(self, commands, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find a command")
        self.resize(740, 520)
        self.commands = commands
        self.chosen = None
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(label("Find a command", "heading"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search tools, processing steps or datasets…")
        self.search.setClearButtonEnabled(True)
        self.results = QtWidgets.QListWidget()
        self.results.setWordWrap(True)
        self.hint = label("↑ ↓ to navigate   ·   Enter to open   ·   Esc to close", "muted")
        for widget in (self.search, self.results, self.hint): layout.addWidget(widget)
        layout.setStretch(2, 1)
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self.activate)
        self.results.itemActivated.connect(self.activate)
        self.search.installEventFilter(self)
        self.filter("")
        self.search.setFocus()

    def filter(self, text):
        words = text.casefold().split()
        self.results.clear()
        for i, command in enumerate(self.commands):
            if all(word in f"{command.title} {command.detail}".casefold() for word in words):
                item = QtWidgets.QListWidgetItem(f"{command.title}\n{command.detail}")
                item.setData(ROLE, i)
                if not command.enabled:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.results.addItem(item)
        for row in range(self.results.count()):
            if self.results.item(row).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                self.results.setCurrentRow(row); break
        self.hint.setText("No matching commands. Try a process name or dataset name." if not self.results.count()
                          else "↑ ↓ to navigate   ·   Enter to open   ·   Esc to close")

    def activate(self, *_args):
        item = self.results.currentItem()
        if item is not None and item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            self.chosen = self.commands[item.data(ROLE)]
            self.accept()

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up):
                direction = 1 if event.key() == QtCore.Qt.Key.Key_Down else -1
                row = self.results.currentRow() + direction
                while 0 <= row < self.results.count():
                    if self.results.item(row).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                        self.results.setCurrentRow(row); break
                    row += direction
                return True
        return super().eventFilter(obj, event)

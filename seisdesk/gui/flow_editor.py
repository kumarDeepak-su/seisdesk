"""Flow editor: build a processing sequence from the process palette, edit each step's parameters in a form generated
from its parameter schema, and run the flow (or part of it) on a dataset in the background."""

from __future__ import annotations

import copy
import dataclasses

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.headers import GATHER_KEYS, SEGY_FIELDS
from ..processing.base import CATEGORIES, REGISTRY
from ..workflow.flow import Flow, FlowStep
from ..workflow.templates import templates
from .common import fmt, show_error, start_worker

ROLE = QtCore.Qt.ItemDataRole.UserRole
PROCESS_MIME = "application/x-seisdesk-process"
COMMON_KEYS = list(GATHER_KEYS) + [k for k, _, _, _ in SEGY_FIELDS if k not in GATHER_KEYS]


class ParamForm(QtWidgets.QWidget):
    """Widgets for the parameters of one flow step; edits go straight into ``step.params``."""
    sigEdited = QtCore.Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.step: FlowStep | None = None
        self.header_keys: list[str] = []
        self.input_data = None  # A callable returning the selected dataset, when available.
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel("<i>select a step</i>")
        self.title.setWordWrap(True)
        self.description = QtWidgets.QLabel("")
        self.description.setWordWrap(True)
        self.description.setProperty("muted", True)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        self.form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText("note for this step (optional)")
        self.note.textEdited.connect(self._note_edited)
        self._layout.addWidget(self.title)
        self._layout.addWidget(self.description)
        self._layout.addWidget(self.form_host)
        self._layout.addWidget(self.note)
        self._layout.addStretch(1)

    def set_step(self, step: FlowStep | None) -> None:
        self.step = step
        while self.form.rowCount():
            self.form.removeRow(0)
        if step is None or step.process not in REGISTRY:
            self.title.setText("<i>select a step</i>")
            self.description.setText("")
            self.note.setText("")
            self.note.setEnabled(False)
            return
        cls = REGISTRY[step.process]
        if getattr(self, "local_compute_defaults", True) and step.process in (
                "volume_file_job", "acoustic_model_3d", "acoustic_rtm_3d", "acoustic_fwi_3d"):
            for key, value in self.project.settings.get("compute_defaults", {}).items():
                if key in ("device", "device_index"):
                    step.params.setdefault(key, value)
        self.title.setText(f"<b>{cls.label}</b><br>{cls.category}")
        self.description.setText(cls.description)
        self.note.setEnabled(True)
        self.note.setText(step.note)
        for p in cls.params:
            value = step.params.get(p.name, p.default)
            widget = self._widget(p, value)
            label = (p.label or p.name) + (f" ({p.unit})" if p.unit else "")
            lab = QtWidgets.QLabel(label)
            if p.help:
                lab.setToolTip(p.help)
                widget.setToolTip(p.help)
            self.form.addRow(lab, widget)

    # ------------------------------------------------------------------ widgets per parameter kind
    def _store(self, p, raw, widget=None) -> None:
        if self.step is None:
            return
        try:
            value = p.coerce(raw)
        except (TypeError, ValueError) as exc:
            if widget is not None:
                widget.setStyleSheet("border: 1px solid #d00;")
                widget.setToolTip(str(exc))
            return
        if widget is not None:
            widget.setStyleSheet("")
            widget.setToolTip(p.help)
        self.step.params[p.name] = value
        self.sigEdited.emit()

    def _line(self, p, text: str) -> QtWidgets.QLineEdit:
        w = QtWidgets.QLineEdit(text)
        w.editingFinished.connect(lambda: self._store(p, w.text(), w))
        return w

    def _combo(self, p, items, value, editable=True) -> QtWidgets.QComboBox:
        w = QtWidgets.QComboBox()
        w.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(12)
        w.setEditable(editable)
        w.addItems([str(i) for i in items])
        w.setCurrentText(str(value))
        if editable:
            w.lineEdit().editingFinished.connect(lambda: self._store(p, w.currentText(), w))
            w.activated.connect(lambda _i: self._store(p, w.currentText(), w))
        else:
            w.currentTextChanged.connect(lambda text: self._store(p, text, w))
        return w

    def _with_browse(self, editor, save: bool, filt: str) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(editor, 1)
        b = QtWidgets.QToolButton()
        b.setText("...")

        def browse():
            if save:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "File", "", filt)
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "File", "", filt)
            if path:
                if isinstance(editor, QtWidgets.QComboBox):
                    editor.setCurrentText(path)
                    editor.lineEdit().editingFinished.emit()
                else:
                    editor.setText(path)
                    editor.editingFinished.emit()
        b.clicked.connect(browse)
        lay.addWidget(b)
        return host

    def _widget(self, p, value) -> QtWidgets.QWidget:
        k = p.kind
        if self.step.process in ("joint_tomography", "joint_tomography_3d") and p.name == "branches":
            import json
            from pathlib import Path
            from ..processing.base import resolve_picks
            from .branch_editor import JointBranchDialog
            from .tomography3d_dialog import Branches3DDialog
            host = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(host); layout.setContentsMargins(0, 0, 0, 0)
            button = QtWidgets.QPushButton("Configure arrivals and interfaces…")
            advanced = QtWidgets.QCheckBox("Show definitions text")
            text = QtWidgets.QPlainTextEdit(str(value)); text.setMaximumHeight(160); text.hide()
            summary = QtWidgets.QLabel(); summary.setWordWrap(True)
            def describe():
                try:
                    definitions = json.loads(text.toPlainText())
                    summary.setText(", ".join(f"{name}: {item.get('kind', '?')}" for name, item in definitions.items()))
                except (ValueError, AttributeError):
                    summary.setText("Definitions file or text; open the editor to review.")
            def edit():
                try:
                    raw = text.toPlainText().strip()
                    if not raw.startswith("{"):
                        path = Path(raw)
                        raw = (path if path.is_absolute() else Path(self.project.context.work_dir)/path).read_text(encoding="utf-8")
                    pickset = resolve_picks(self.step.params.get("pickset", "first breaks"), self.project.context)
                    data = self.input_data() if self.input_data else None
                    x_range = None
                    if data is not None and "sx" in data.headers and "gx" in data.headers:
                        import numpy as np
                        positions = np.r_[data.header("sx"), data.header("gx")]
                        dx = float(self.step.params.get("dx", 10))
                        x_range = (np.floor(positions.min()/dx)*dx, np.ceil(positions.max()/dx)*dx)
                    if self.step.process == "joint_tomography_3d":
                        dialog = Branches3DDialog(json.loads(raw), pickset, parent=self)
                    else:
                        dialog = JointBranchDialog(json.loads(raw), pickset, self.project.context.depth_model, self,
                                                    x_range=x_range, zmax=float(self.step.params.get("zmax", 200)))
                    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                        text.setPlainText(dialog.json_text())
                except Exception as exc:
                    show_error(self, "Arrival definitions", exc)
            text.textChanged.connect(lambda: self._store(p, text.toPlainText(), text))
            text.textChanged.connect(describe)
            advanced.toggled.connect(text.setVisible); button.clicked.connect(edit)
            for widget in (button, summary, advanced, text):
                layout.addWidget(widget)
            describe()
            return host
        if k == "text":
            w = QtWidgets.QPlainTextEdit(str(value))
            w.setMaximumHeight(150)
            w.textChanged.connect(lambda: self._store(p, w.toPlainText(), w))
            return w
        if k == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda on: self._store(p, on))
            return w
        if k == "int":
            w = QtWidgets.QSpinBox()
            lo = int(p.minimum) if p.minimum is not None else -2_000_000_000
            hi = int(p.maximum) if p.maximum is not None else 2_000_000_000
            w.setRange(lo, hi)
            w.setValue(int(value))
            w.valueChanged.connect(lambda v: self._store(p, v))
            return w
        if k == "choice":
            return self._combo(p, p.choices, value, editable=False)
        if k == "floats":
            return self._line(p, ", ".join(fmt(v) for v in (value if not isinstance(value, str) else
                                                             p.coerce(value))))
        if k == "keys":
            return self._line(p, ", ".join(value) if not isinstance(value, str) else value)
        if k == "key":
            keys = self.header_keys or COMMON_KEYS
            return self._combo(p, ([""] if value == "" else []) + keys, value)
        if k == "velocity":
            items = ["project", "project model", "1500", "0:1500,1:2000,2:2600,3:3000"]
            combo = self._combo(p, items, value)
            return self._with_browse(combo, False, "Velocity picks (*.txt *.vel);;All files (*)")
        if k == "picks":
            names = list(self.project.context.picks) or [str(value)]
            combo = self._combo(p, names, value)
            return self._with_browse(combo, False, "Pick sets (*.json);;All files (*)")
        if k == "file":
            line = self._line(p, str(value))
            return self._with_browse(line, True, "SEG-Y (*.sgy *.segy);;Seismic Unix (*.su);;All files (*)")
        if k == "infile":
            line = self._line(p, str(value))
            return self._with_browse(line, False, "Supported files (*.csv *.txt *.dat *.npz *.npy *.sgy *.segy *.su *.json);;All files (*)")
        if k == "directory":
            line = self._line(p, str(value))
            host = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(line, 1)
            button = QtWidgets.QToolButton()
            button.setText("...")
            def browse_directory():
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Directory", line.text())
                if path:
                    line.setText(path)
                    line.editingFinished.emit()
            button.clicked.connect(browse_directory)
            layout.addWidget(button)
            return host
        if k == "float":
            return self._line(p, fmt(value))
        return self._line(p, str(value))

    def _note_edited(self, text: str) -> None:
        if self.step is not None:
            self.step.note = text
            self.sigEdited.emit()


class ProcessPalette(QtWidgets.QTreeWidget):
    def startDrag(self, supported_actions):
        item = self.currentItem()
        name = item.data(0, ROLE) if item is not None else None
        if name not in REGISTRY:
            return
        mime = QtCore.QMimeData(); mime.setData(PROCESS_MIME, name.encode("utf-8"))
        drag = QtGui.QDrag(self); drag.setMimeData(mime)
        drag.exec(QtCore.Qt.DropAction.CopyAction)


class StepList(QtWidgets.QListWidget):
    orderChanged = QtCore.Signal()
    processDropped = QtCore.Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropOverwriteMode(False)

    def _process_name(self, event):
        name = bytes(event.mimeData().data(PROCESS_MIME)).decode("utf-8", errors="replace")
        return name if name in REGISTRY else None

    def dragEnterEvent(self, event):
        if self._process_name(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._process_name(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        name = self._process_name(event)
        if name:
            point = event.position().toPoint()
            item = self.itemAt(point)
            row = self.row(item) if item is not None else self.count()
            if item is not None and point.y() > self.visualItemRect(item).center().y():
                row += 1
            self.processDropped.emit(name, row)
            event.acceptProposedAction()
            return
        super().dropEvent(event)
        if event.isAccepted():
            self.orderChanged.emit()


class FlowEditor(QtWidgets.QWidget):
    sigResult = QtCore.Signal(str)
    sigStatus = QtCore.Signal(str)
    sigLog = QtCore.Signal(str)
    sigContextChanged = QtCore.Signal()
    sigFlowChanged = QtCore.Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.flow = Flow([], "flow")
        self.preview_source = None              # callable -> SeismicData of the displayed gather (set by the window)
        self._worker = None
        self._running = False
        self._run_outcome = False
        self._history = [self.flow.to_json()]
        self._history_index = 0
        self._restoring_history = False
        self._build()
        self.sigFlowChanged.connect(self._record_history)
        self._update_run_actions()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        # palette
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search names, categories and descriptions…")
        self.search.setClearButtonEnabled(True)
        self.palette = ProcessPalette()
        self.palette.setDragEnabled(True)
        self.palette.setHeaderHidden(True)
        by_cat = {}
        for cls in REGISTRY.values():
            by_cat.setdefault(cls.category, []).append(cls)
        for cat in list(CATEGORIES) + sorted(set(by_cat) - set(CATEGORIES)):
            if cat not in by_cat:
                continue
            top = QtWidgets.QTreeWidgetItem([cat])
            for cls in sorted(by_cat[cat], key=lambda c: c.label):
                child = QtWidgets.QTreeWidgetItem([cls.label])
                child.setData(0, ROLE, cls.name)
                child.setToolTip(0, cls.description)
                top.addChild(child)
            self.palette.addTopLevelItem(top)
        self.palette.collapseAll()
        add_btn = QtWidgets.QPushButton("+ Add to flow")
        add_btn.setProperty("primary", True)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QtWidgets.QLabel("<b>Processes</b>"))
        ll.addWidget(self.search)
        ll.addWidget(self.palette, 1)
        ll.addWidget(add_btn)

        # steps
        self.name_edit = QtWidgets.QLineEdit("flow")
        self.steps = StepList()
        self.steps.setWordWrap(True)
        self.steps.setToolTip("Drag steps to reorder. Uncheck a step to bypass it. Delete removes the selected step.")
        self.template_combo = QtWidgets.QComboBox()
        self.template_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.template_combo.setMinimumContentsLength(20)
        self.template_combo.addItem("Start from a template...")
        self.template_combo.addItems(list(templates()))
        b_up, b_down, b_dup, b_del, b_clear = (QtWidgets.QPushButton(t) for t in
                                               ("Up", "Down", "Duplicate", "Remove", "Clear"))
        b_load, b_save = QtWidgets.QPushButton("Load flow..."), QtWidgets.QPushButton("Save flow...")
        mid = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Flow</b>"))
        row.addWidget(self.name_edit, 1)
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.redo_btn = QtWidgets.QPushButton("Redo")
        self.undo_btn.setToolTip("Undo a flow edit, including parameters. Ctrl+Z when the step list has focus.")
        self.redo_btn.setToolTip("Redo a flow edit. Ctrl+Y when the step list has focus.")
        row.addWidget(self.undo_btn); row.addWidget(self.redo_btn)
        ml.addLayout(row)
        ml.addWidget(self.template_combo)
        hint = QtWidgets.QLabel("Drag to reorder · Uncheck to bypass")
        hint.setProperty("muted", True); ml.addWidget(hint)
        self.empty_flow = QtWidgets.QLabel("Build your processing sequence.\n\nDouble-click a process on the left,\nor start from a template above.")
        self.empty_flow.setWordWrap(True); self.empty_flow.setProperty("muted", True)
        ml.addWidget(self.empty_flow)
        ml.addWidget(self.steps, 1)
        grid = QtWidgets.QGridLayout()
        for i, b in enumerate((b_up, b_down, b_dup, b_del, b_clear)):
            grid.addWidget(b, 0, i)
        grid.addWidget(b_load, 1, 0, 1, 2)
        grid.addWidget(b_save, 1, 2, 1, 3)
        ml.addLayout(grid)

        # parameters
        self.form = ParamForm(self.project)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("<b>Parameters</b>"))
        rl.addWidget(scroll, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for w in (left, mid, right):
            split.addWidget(w)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 3)
        split.setSizes([290, 360, 430])
        split.setChildrenCollapsible(False)
        self.editor_panels = split

        # run bar
        self.input_combo = QtWidgets.QComboBox()
        self.input_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.input_combo.setMinimumContentsLength(16)
        self.input_combo.setMinimumWidth(200)
        self.form.input_data = lambda: self.project.datasets.get(self.input_combo.currentText())
        self.preview_check = QtWidgets.QCheckBox("displayed gather only (parameter test)")
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("output name (default: input + flow name)")
        self.run_btn = QtWidgets.QPushButton("Run flow")
        self.run_btn.setProperty("primary", True)
        self.run_sel_btn = QtWidgets.QPushButton("Run to selected step")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.run_status = QtWidgets.QLabel("Add a process or choose a template to begin.")
        self.run_status.setProperty("muted", True)
        self.run_status.setWordWrap(True)
        run = QtWidgets.QGridLayout()
        run.addWidget(QtWidgets.QLabel("Input"), 0, 0)
        run.addWidget(self.input_combo, 0, 1)
        run.addWidget(self.preview_check, 0, 2)
        run.addWidget(QtWidgets.QLabel("Output"), 0, 3)
        run.addWidget(self.output_edit, 0, 4)
        run.addWidget(self.run_btn, 1, 0, 1, 2)
        run.addWidget(self.run_sel_btn, 1, 2)
        run.addWidget(self.cancel_btn, 1, 3)
        run.addWidget(self.progress, 1, 4)
        run.addWidget(self.run_status, 2, 0, 1, 5)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(split, 1)
        layout.addLayout(run)

        self.search.textChanged.connect(self._filter)
        self.palette.itemDoubleClicked.connect(lambda item, _c: self._add_item(item))
        add_btn.clicked.connect(lambda: self._add_item(self.palette.currentItem()))
        self.steps.currentRowChanged.connect(self._select_step)
        self.steps.itemChanged.connect(self._item_changed)
        self.steps.orderChanged.connect(self._steps_reordered)
        self.steps.processDropped.connect(self.add_process)
        self.undo_btn.clicked.connect(lambda: self._restore_history(-1))
        self.redo_btn.clicked.connect(lambda: self._restore_history(1))
        b_up.clicked.connect(lambda: self._move(-1))
        b_down.clicked.connect(lambda: self._move(+1))
        b_dup.clicked.connect(self._duplicate)
        b_del.clicked.connect(self._remove)
        b_clear.clicked.connect(self._clear)
        b_load.clicked.connect(self.load_flow)
        b_save.clicked.connect(self.save_flow)
        self.template_combo.activated.connect(self._template)
        self.name_edit.textEdited.connect(self._name_edited)
        self.form.sigEdited.connect(self._refresh_labels)
        self.run_btn.clicked.connect(lambda: self.run(None))
        self.run_sel_btn.clicked.connect(lambda: self.run(self.steps.currentRow() + 1))
        self.cancel_btn.clicked.connect(self.cancel)
        self.input_combo.currentIndexChanged.connect(lambda _i: self._update_run_actions())
        delete = QtGui.QShortcut(QtGui.QKeySequence("Delete"), self.steps)
        delete.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        delete.activated.connect(self._remove)
        for keys, direction in (("Ctrl+Z", -1), ("Ctrl+Y", 1)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self.steps)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda delta=direction: self._restore_history(delta))

    # ------------------------------------------------------------------ palette and steps
    def _filter(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self.palette.topLevelItemCount()):
            top = self.palette.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                child = top.child(j)
                name = child.data(0, ROLE) or ""
                haystack = f"{top.text(0)} {child.text(0)} {name} {child.toolTip(0)}".lower()
                hit = all(word in haystack for word in text.split())
                child.setHidden(not hit)
                shown += hit
            top.setHidden(shown == 0)
            top.setExpanded(bool(text) and shown > 0)

    def _add_item(self, item) -> None:
        if item is None or item.data(0, ROLE) is None:
            return
        name = item.data(0, ROLE)
        self.add_process(name)

    def add_process(self, name, position=None):
        if self._running or name not in REGISTRY:
            return
        defaults = {p.name: p.default for p in REGISTRY[name].params}
        step = FlowStep(name, {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in defaults.items()})
        pos = (self.steps.currentRow() + 1 if self.steps.currentRow() >= 0 else len(self.flow.steps)) if position is None else position
        pos = max(0, min(pos, len(self.flow.steps)))
        self.flow.steps.insert(pos, step)
        self._rebuild(select=pos)

    def _rebuild(self, select: int | None = None) -> None:
        self.steps.blockSignals(True)
        self.steps.clear()
        for i, step in enumerate(self.flow.steps):
            cls = REGISTRY.get(step.process)
            item = QtWidgets.QListWidgetItem(f"{i + 1}. {cls.label if cls else step.process + ' (unknown)'}")
            item.setData(ROLE, i)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if step.enabled else QtCore.Qt.CheckState.Unchecked)
            item.setToolTip(step.note or (cls.description if cls else ""))
            self.steps.addItem(item)
        self.steps.blockSignals(False)
        if select is not None and 0 <= select < len(self.flow.steps):
            self.steps.setCurrentRow(select)
        else:
            self._select_step(self.steps.currentRow())
        self.name_edit.setText(self.flow.name)
        self.empty_flow.setVisible(not self.flow.steps)
        self._update_run_actions()
        self.sigFlowChanged.emit()

    def _steps_reordered(self):
        row = self.steps.currentRow()
        order = [self.steps.item(i).data(ROLE) for i in range(self.steps.count())]
        if sorted(order) != list(range(len(self.flow.steps))):
            self._rebuild(select=row)
            return
        self.flow.steps = [self.flow.steps[i] for i in order]
        self._rebuild(select=row)

    def _name_edited(self, text):
        self.flow.name = text
        self.sigFlowChanged.emit()

    def reset_history(self):
        self._run_outcome = False
        self.progress.setValue(0)
        self._history = [self.flow.to_json()]
        self._history_index = 0
        self._update_run_actions()

    def _record_history(self):
        if self._restoring_history:
            return
        snapshot = self.flow.to_json()
        if snapshot != self._history[self._history_index]:
            self._run_outcome = False
            self.progress.setValue(0)
            self._history = self._history[:self._history_index + 1] + [snapshot]
            self._history = self._history[-100:]
            self._history_index = len(self._history) - 1
        self._update_run_actions()

    def _restore_history(self, direction):
        index = self._history_index + direction
        if self._running or not 0 <= index < len(self._history):
            return
        self._history_index = index
        self._run_outcome = False
        self.progress.setValue(0)
        self._restoring_history = True
        try:
            self.flow = Flow.from_json(self._history[index])
            self._rebuild(select=min(self.steps.currentRow(), len(self.flow.steps) - 1))
        finally:
            self._restoring_history = False
        self._update_run_actions()

    def _refresh_labels(self) -> None:
        row = self.steps.currentRow()
        if 0 <= row < len(self.flow.steps):
            self.steps.item(row).setToolTip(self.flow.steps[row].note)
        self.sigFlowChanged.emit()

    def _select_step(self, row: int) -> None:
        self.form.header_keys = self._input_header_keys()
        self.form.set_step(self.flow.steps[row] if 0 <= row < len(self.flow.steps) else None)
        self._update_run_actions()

    def _item_changed(self, item) -> None:
        row = self.steps.row(item)
        if 0 <= row < len(self.flow.steps):
            self.flow.steps[row].enabled = item.checkState() == QtCore.Qt.CheckState.Checked
            self._update_run_actions()
            self.sigFlowChanged.emit()

    def _move(self, d: int) -> None:
        row = self.steps.currentRow()
        new = row + d
        if 0 <= row < len(self.flow.steps) and 0 <= new < len(self.flow.steps):
            s = self.flow.steps
            s[row], s[new] = s[new], s[row]
            self._rebuild(select=new)

    def _duplicate(self) -> None:
        row = self.steps.currentRow()
        if 0 <= row < len(self.flow.steps):
            self.flow.steps.insert(row + 1, copy.deepcopy(self.flow.steps[row]))
            self._rebuild(select=row + 1)

    def _remove(self) -> None:
        row = self.steps.currentRow()
        if 0 <= row < len(self.flow.steps):
            self.flow.steps.pop(row)
            self._rebuild(select=min(row, len(self.flow.steps) - 1))

    def _clear(self) -> None:
        if self.flow.steps and QtWidgets.QMessageBox.question(self, "Clear flow", "Remove every step?") == \
                QtWidgets.QMessageBox.StandardButton.Yes:
            self.flow.steps.clear()
            self._rebuild()

    def _template(self, index: int) -> None:
        if index <= 0:
            return
        name = self.template_combo.itemText(index)
        self.template_combo.setCurrentIndex(0)
        if self.flow.steps and QtWidgets.QMessageBox.question(
                self, "Template", f"Replace the current flow by '{name}'?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.set_flow(templates()[name])

    def set_flow(self, flow: Flow) -> None:
        self.flow = copy.deepcopy(flow)
        self._rebuild(select=0 if flow.steps else None)

    def load_flow(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load flow", "", "Flows (*.json);;All files (*)")
        if path:
            try:
                self.set_flow(Flow.load(path))
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Load flow", exc)

    def save_flow(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save flow", f"{self.flow.name}.json",
                                                        "Flows (*.json)")
        if path:
            try:
                self.flow.save(path)
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Save flow", exc)

    # ------------------------------------------------------------------ inputs
    def refresh_inputs(self, current: str | None = None) -> None:
        keep = current or self.input_combo.currentText()
        self.input_combo.blockSignals(True)
        self.input_combo.clear()
        self.input_combo.addItems(list(self.project.datasets))
        i = self.input_combo.findText(keep)
        if i >= 0:
            self.input_combo.setCurrentIndex(i)
        self.input_combo.blockSignals(False)
        self._update_run_actions()

    def _update_run_actions(self):
        self.undo_btn.setEnabled(not self._running and self._history_index > 0)
        self.redo_btn.setEnabled(not self._running and self._history_index < len(self._history) - 1)
        available = self.input_combo.currentText() in self.project.datasets
        enabled = any(step.enabled for step in self.flow.steps)
        row = self.steps.currentRow()
        self.run_btn.setEnabled(not self._running and available and enabled)
        self.run_sel_btn.setEnabled(not self._running and available and row >= 0 and
                                   any(step.enabled for step in self.flow.steps[:row + 1]))
        if not self._running and not self._run_outcome:
            self.run_status.setText("Choose an input dataset to run this flow." if not available else
                                    "Add a process or enable a step to begin." if not enabled else
                                    f"Ready · {sum(s.enabled for s in self.flow.steps)} enabled steps · choose Run flow or preview a gather.")

    def _input_header_keys(self) -> list[str]:
        data = self.project.datasets.get(self.input_combo.currentText())
        if data is None:
            return []
        present = list(data.headers)
        return [k for k in COMMON_KEYS if k in present] + sorted(k for k in present if k not in COMMON_KEYS)

    # ------------------------------------------------------------------ running
    def run(self, stop: int | None) -> None:
        if self._worker is not None:
            return
        name = self.input_combo.currentText()
        data = self.project.datasets.get(name)
        if self.preview_check.isChecked() and self.preview_source is not None:
            data = self.preview_source()
            name = f"{name} [test gather]"
        if data is None:
            show_error(self, "Run flow", "Choose an input dataset (load or create one first).")
            return
        if not any(s.enabled for s in self.flow.steps[:stop]):
            show_error(self, "Run flow", "The flow has no enabled steps.")
            return
        try:
            for s in self.flow.steps[:stop]:
                if s.enabled:
                    s.build()
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Run flow", exc)
            return
        flow = copy.deepcopy(self.flow)
        out_name = self.output_edit.text().strip() or f"{name} > {flow.name}"
        project = self.project
        self.sigLog.emit(f"--- running flow '{flow.name}' on '{name}' ---")

        base = dataclasses.replace(project.context, velocity=copy.deepcopy(project.context.velocity),
                                   picks=copy.deepcopy(project.context.picks), results={})

        def job(worker):
            ctx = dataclasses.replace(base, log=worker.message.emit, progress=worker.progress.emit,
                                      cancelled=worker.cancelled)
            result = flow.run(data, ctx, 0, stop)
            return result, ctx, out_name

        self._set_running(True)
        self._worker = start_worker(self, job, self._finished, self._failed, self._message, self._progress)

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.run_status.setText("Cancelling… waiting for the current process to reach a cancellation check.")
            self.sigLog.emit("cancel requested; the flow stops at the next progress check")

    def _set_running(self, on: bool) -> None:
        self._running = on
        for widget in (self.editor_panels, self.input_combo, self.preview_check, self.output_edit):
            widget.setEnabled(not on)
        self._update_run_actions()
        self.cancel_btn.setEnabled(on)
        self.progress.setValue(0)
        if on:
            self._run_outcome = False
            self.run_status.setText("Processing… results will open in the viewer when ready.")

    @QtCore.Slot(object)
    def _finished(self, payload) -> None:
        result, ctx, out_name = payload
        self._worker = None
        self._run_outcome = True
        self._set_running(False)
        self.project.context.velocity = ctx.velocity
        self.project.context.depth_model = ctx.depth_model
        self.project.context.picks.update(ctx.picks)
        self.project.context.results.update(ctx.results)
        key = self.project.add_dataset(result, out_name)
        self.progress.setValue(1000)
        self.run_status.setText(f"Completed · {result.ntr:,} traces × {result.nt:,} samples · {key}")
        self.sigStatus.emit("Processing complete")
        self.sigLog.emit(f"--- finished: dataset '{key}' ({result.ntr} traces x {result.nt} samples) ---")
        self.sigContextChanged.emit()
        self.sigResult.emit(key)

    @QtCore.Slot(str)
    def _failed(self, text: str) -> None:
        self._worker = None
        self._run_outcome = True
        self._set_running(False)
        self.sigLog.emit(text)
        self.run_status.setText("Cancelled. Your input dataset is unchanged." if text.strip() == "Cancelled." else
                               "Processing failed. See the error details and activity log.")
        self.sigContextChanged.emit()
        if text.strip() != "Cancelled.":
            show_error(self, "Flow failed", text)

    @QtCore.Slot(str)
    def _message(self, text: str) -> None:
        self.sigLog.emit(text)
        if self.cancel_btn.isEnabled():
            self.run_status.setText(text.splitlines()[-1] if text else "Processing…")

    @QtCore.Slot(float)
    def _progress(self, fraction: float) -> None:
        self.progress.setValue(int(round(1000 * fraction)))

"""SeisDesk main window: project panel, gather viewers, flow editor, velocity analysis, log and menus."""

from __future__ import annotations

import os
import json
import html

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.seg2 import is_seg2, read_seg2, read_seg2_files
from ..core.segy_io import read_segy, read_su, write_segy, write_su
from ..core.velocity import DepthModel, VelocityPicks
from ..processing.base import REGISTRY
from ..workflow.project import Project
from ..workflow.flow import Flow
from .common import busy, show_error, start_worker
from .dialogs import (HeaderTableDialog, ImportDialog, SEG2ImportDialog, RefractionStaticsDialog, SingleStepDialog, SpectrumDialog,
                      SurveyQCDialog, SyntheticDialog, TextDialog, VelocityFieldDialog)
from .flow_editor import FlowEditor
from .velan import VelocityAnalysis
from .viewer import GatherViewer
from .theme import apply_theme
from .workspace import WorkspaceHome, Command, CommandPalette

ROLE = QtCore.Qt.ItemDataRole.UserRole
SEISMIC_FILTER = ("Seismic files (*.sgy *.segy *.SGY *.SEGY *.su *.SU *.dat *.DAT *.sg2 *.seg2 *.segd *.sgd *.SEGD);;SEG-Y (*.sgy *.segy);;"
                  "Seismic Unix (*.su);;SEG-2 (*.dat *.sg2 *.seg2);;SEG-D (*.segd *.sgd *.SEGD);;All files (*)")
SEG2_FILTER = "SEG-2 shot files (*.dat *.DAT *.sg2 *.SG2 *.seg2 *.SEG2);;All files (*)"


def default_work_dir() -> str:
    docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    path = os.path.join(docs or os.path.expanduser("~"), "SeisDesk")
    os.makedirs(path, exist_ok=True)
    return path


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.ui_settings = settings if settings is not None else QtCore.QSettings("SeisDesk", "Desktop")
        self.theme = apply_theme(self.ui_settings.value("appearance/theme", "light"))
        self.setAcceptDrops(True)
        self.setWindowTitle(f"SeisDesk {__version__}")
        self.resize(1500, 920)
        self.project = Project()
        self.project.context.work_dir = default_work_dir()
        self.project.context.log = self.log
        self.dirty = False
        self.viewers: list[GatherViewer] = []
        self._build()
        self._menus()
        self._build_toolbar()
        self.refresh_project()
        for key, restore in (("window/geometry", self.restoreGeometry), ("window/layout", self.restoreState)):
            value = self.ui_settings.value(key)
            if isinstance(value, QtCore.QByteArray):
                restore(value)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.home = WorkspaceHome({
            "import": self.open_files, "project": self.open_project, "synthetic": self.new_synthetic,
            "flow": lambda: self.tabs.setCurrentWidget(self.flow_editor), "compute": self.compute_devices,
            "remote": self.remote_job, "guide": self.user_guide,
        })
        self.home.openRecent.connect(self.open_recent_project)
        self.tabs.addTab(self.home, "Home")
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.viewer = self._new_viewer()
        self.tabs.addTab(self.viewer, "Viewer")
        self.flow_editor = FlowEditor(self.project)
        self.flow_editor.preview_source = lambda: self.viewer.displayed_data()
        self.tabs.addTab(self.flow_editor, "Flow editor")
        self.velan = VelocityAnalysis(self.project)
        self.tabs.addTab(self.velan, "Velocity analysis")
        for i in range(4):
            self.tabs.tabBar().setTabButton(i, QtWidgets.QTabBar.ButtonPosition.RightSide, None)
        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self._tab_changed)

        # project dock
        self.dataset_list = QtWidgets.QListWidget()
        self.dataset_list.setWordWrap(True)
        self.dataset_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.dataset_list.customContextMenuRequested.connect(self._dataset_menu)
        self.dataset_list.itemActivated.connect(lambda item: self.show_dataset(item.data(ROLE)))
        self.dataset_list.currentItemChanged.connect(lambda cur, _prev: self._dataset_selected(cur))
        self.dataset_search = QtWidgets.QLineEdit()
        self.dataset_search.setPlaceholderText("Filter datasets…")
        self.dataset_search.setClearButtonEnabled(True)
        self.dataset_search.textChanged.connect(self._filter_datasets)
        self.dataset_empty = QtWidgets.QLabel("Import a seismic file to get started.")
        self.dataset_empty.setWordWrap(True)
        self.dataset_empty.setProperty("muted", True)
        self.context_label = QtWidgets.QLabel("")
        self.context_label.setWordWrap(True)
        self.pick_list = QtWidgets.QListWidget()
        self.pick_list.setMaximumHeight(120)
        host = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(host)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.addWidget(QtWidgets.QLabel("<b>Datasets</b>"))
        lay.addWidget(self.dataset_search)
        lay.addWidget(self.dataset_empty)
        lay.addWidget(self.dataset_list, 1)
        self.dataset_list.setToolTip("Double-click to view. Right-click for export, rename and other actions.")
        import_button = QtWidgets.QPushButton("+ Import seismic data")
        import_button.clicked.connect(self.open_files)
        lay.addWidget(import_button)
        lay.addWidget(QtWidgets.QLabel("<b>Pick sets</b>"))
        lay.addWidget(self.pick_list)
        lay.addWidget(QtWidgets.QLabel("<b>Velocities and models</b>"))
        lay.addWidget(self.context_label)
        dock = QtWidgets.QDockWidget("Project", self)
        dock.setObjectName("project_dock")
        dock.setWidget(host)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.project_dock = dock
        self.resizeDocks([dock], [285], QtCore.Qt.Orientation.Horizontal)

        # log dock
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        dock = QtWidgets.QDockWidget("Log", self)
        dock.setObjectName("log_dock")
        dock.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.log_dock = dock
        dock.hide()

        self.flow_editor.sigLog.connect(self.log)
        self.flow_editor.sigResult.connect(self._flow_result)
        self.flow_editor.sigContextChanged.connect(self.refresh_project)
        self.flow_editor.sigFlowChanged.connect(self._flow_edited)
        self.flow_editor.sigStatus.connect(self.status)
        self.velan.sigStatus.connect(self.status)
        self.velan.sigPicksChanged.connect(self._context_changed)
        self.statusBar().showMessage("Open a SEG-Y/SU file (File > Open) or create synthetic data to start.")

    def _new_viewer(self) -> GatherViewer:
        v = GatherViewer(self.project)
        v.sigStatus.connect(self.status)
        v.sigPicksChanged.connect(self._context_changed)
        self.viewers.append(v)
        return v

    def _menus(self) -> None:
        mb = self.menuBar()

        def act(menu, text, fn, shortcut=None, tip=None):
            a = QtGui.QAction(text, self)
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            if tip:
                a.setStatusTip(tip)
            a.triggered.connect(lambda _checked=False: fn())
            menu.addAction(a)
            return a

        def steps(menu, title, names):
            sub_menu = menu.addMenu(title)
            for n in names:
                if n in REGISTRY:
                    act(sub_menu, REGISTRY[n].label + "...", lambda n=n: self.single_step(n))
            return sub_menu

        m = mb.addMenu("&File")
        act(m, "&Open seismic file (SEG-Y / SU / SEG-2)...", self.open_files, "Ctrl+O")
        act(m, "Open SEG-2 shot &files of a line...", self.open_seg2, "Ctrl+Shift+2")
        act(m, "Open GPR profile (DT1 + HD)…", self.open_gpr)
        act(m, "New &synthetic dataset...", self.new_synthetic)
        m.addSeparator()
        act(m, "Export dataset as SEG-Y...", lambda: self.export_dataset("segy"))
        act(m, "Export dataset as Seismic Unix...", lambda: self.export_dataset("su"))
        act(m, "Save display as image (PNG)...", self.save_image, "Ctrl+Shift+S")
        m.addSeparator()
        act(m, "&New project", self.new_project)
        act(m, "Open &project...", self.open_project, "Ctrl+Shift+O")
        act(m, "&Save project", self.save_project, "Ctrl+S")
        act(m, "Save project &as...", lambda: self.save_project(ask=True))
        m.addSeparator()
        act(m, "&Quit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Geometry")
        act(m, "2D geometry: offsets and CMPs...", lambda: self.single_step("geometry_2d"))
        act(m, "Land geometry from shot table (CSV) and topography...", lambda: self.single_step("shot_table_geometry"))
        act(m, "SPS survey geometry (S / R / X files)...", lambda: self.single_step("sps_geometry"))
        act(m, "Streamer geometry from shot and channel numbers...", lambda: self.single_step("streamer_geometry"))
        act(m, "3D binning: inlines and crosslines...", lambda: self.single_step("geometry_3d"))
        act(m, "Header formula (key = a + b*key2 + c*key3)...", lambda: self.single_step("header_formula"))
        steps(m, "More geometry tools", ("utm", "azimuth", "flex_binning", "bin_enlarge"))
        m.addSeparator()
        act(m, "Header table of the dataset", self.header_table)

        m = mb.addMenu("&Processing")
        act(m, "Flow editor", lambda: self.tabs.setCurrentWidget(self.flow_editor), "F5")
        act(m, "Apply one step to the dataset...", self.choose_step)
        act(m, "Seismic Unix programs and file jobs…", self.su_workbench)
        act(m, "3D survey file job…", self.volume_job)
        act(m, "Workstation CPU / CUDA devices…", self.compute_devices)
        act(m, "Remote processing (SSH / Slurm)…", self.remote_job)
        steps(m, "Full 3D acoustic processing", ("acoustic_model_3d", "acoustic_rtm_3d", "acoustic_fwi_3d"))
        m.addSeparator()
        act(m, "Automatic trace editing (dead / noisy traces)...", lambda: self.single_step("auto_edit"))
        act(m, "Despike / noise-burst suppression...", lambda: self.single_step("despike"))
        m.addSeparator()
        act(m, "Elevation statics...", lambda: self.single_step("elevation_statics"))
        act(m, "Refraction statics (delay-time inversion)...", self.refraction_statics)
        act(m, "Refraction statics results", self.show_refraction_statics)
        act(m, "Statics from an x-static table...", lambda: self.single_step("table_statics"))
        act(m, "Refraction convolution section...", lambda: self.single_step("refraction_convolution"))
        m.addSeparator()
        steps(m, "Wavelet and deconvolution", ("decon", "sc_decon", "tv_decon", "gabor_decon", "signature_decon",
                                               "homomorphic_decon", "shaping_filter", "phase_rotation", "zero_phase",
                                               "min_phase"))
        steps(m, "Vibroseis", ("vibro_correlation", "klauder_min_phase", "vibro_convolve"))
        steps(m, "Noise attenuation", ("fx_decon", "median_filter", "taup_filter", "eigen_filter", "harlan",
                                       "deghost", "trace_mix", "structure_smooth", "fx_cc_filter", "fx_alpha_trim",
                                       "fx_weighted_mix", "kfilter", "kfrac", "frac_derivative", "lf_array_forming"))
        steps(m, "Multiples", ("radon_demultiple", "hyperbolic_radon", "taup_decon", "srme",
                               "adaptive_subtraction"))
        steps(m, "Interpolation", ("fx_interpolation", "pocs_fill"))
        steps(m, "Datuming", ("datum_kirchhoff", "datum_zero_offset", "elevation_statics", "floating_datum",
                              "nszone_shift", "streamer_motion"))
        steps(m, "Time migration", ("stolt", "phase_shift", "phase_shift_vti", "kirchhoff_pstm", "stolt_3d",
                                    "phase_shift_3d", "kirchhoff_pstm_3d"))
        steps(m, "Depth migration", ("split_step", "fd_migration", "ffd_migration", "pspi_migration",
                                     "gaussian_beam", "kirchhoff_psdm", "wave_psdm", "rtm", "kirchhoff_psdm_3d"))
        steps(m, "Inversion and tomography", ("fwi", "refraction_tomography", "grm", "mva_tomography"))
        steps(m, "AVO", ("offset_to_angle", "angle_stack", "avo_attributes", "avo_modelling"))
        steps(m, "Converted waves", ("ps_binning", "ps_nmo", "ps_dmo", "vpvs_analysis", "ps_synthetics"))
        steps(m, "Attributes", ("hilbert", "attributes", "spectral_decomp"))
        steps(m, "Multicomponent", ("hrot", "alford", "ltt_rotation", "polarisation", "polar_filter"))
        steps(m, "Modelling", ("fd_acoustic", "fd_elastic", "fd_elastic_vti", "rem_acoustic", "kirchhoff_modelling",
                               "ray_synthetics", "ti_ray_synthetics", "goupillaud", "scatterer_synthetics"))
        steps(m, "Utilities", ("taper", "trace_weight", "dip_divcor", "header_math", "dataset_math",
                               "reciprocal_sum", "frac_autocorr", "impedance_reflectivity", "offset_continuation",
                               "gpr_first_break_removal"))

        m = mb.addMenu("&Velocity")
        act(m, "Velocity analysis on the selected dataset", self.open_velan, "F6")
        act(m, "Show velocity field", self.velocity_field)
        act(m, "Residual moveout semblance scan...", lambda: self.single_step("rmo_scan"))
        act(m, "Anisotropic DMO (migration to zero offset)...", lambda: self.single_step("mzo_dmo"))
        act(m, "Linear moveout (reduced time)...", lambda: self.single_step("lmo"))
        steps(m, "More velocity tools", ("fowler", "dmo_3d", "interval_to_stacking", "ti_velocity_table",
                                         "velocity_semblance_slice", "taup_nmo"))
        m.addSeparator()
        act(m, "Load velocity picks...", self.load_velocity)
        act(m, "Save velocity picks...", self.save_velocity)
        act(m, "Clear all velocity picks", self.clear_velocity)
        m.addSeparator()
        act(m, "Load depth model...", self.load_depth_model)
        act(m, "Save depth model...", self.save_depth_model)
        act(m, "Create 3D velocity model…", self.create_volume)
        act(m, "Inspect 3D velocity model…", self.inspect_volume)

        m = mb.addMenu("Pic&king")
        act(m, "Picking tool (first breaks, refraction, mutes)", self.open_picking, "F7")

        m = mb.addMenu("&Tools")
        act(m, "MASW dispersion analysis...", self.masw)
        act(m, "Layered surface-wave inversion...", self.surface_wave_inversion)
        act(m, "Time-frequency analysis of a trace...", self.time_frequency, "F10")
        act(m, "Well logs and synthetic seismogram...", self.well_logs)
        act(m, "AVO intercept-gradient crossplot...", self.avo_crossplot)
        act(m, "Refraction tomography results...", lambda: self.inversion_results("refraction_tomography"))
        act(m, "Joint refraction / reflection tomography…", lambda: self.single_step("joint_tomography"))
        act(m, "Joint tomography results…", lambda: self.inversion_results("joint_tomography"))
        act(m, "3D arrival tomography…", lambda: self.single_step("joint_tomography_3d"))
        act(m, "3D tomography and arrival QC results…", lambda: self.inversion_results("joint_tomography_3d"))
        act(m, "3D / 5D survey interpolation…", lambda: self.single_step("survey_pocs"))
        act(m, "Survey reconstruction coverage…", lambda: self.inversion_results("survey_pocs"))
        act(m, "GRM candidate XY results...", lambda: self.inversion_results("grm"))
        act(m, "Full-waveform inversion results...", lambda: self.inversion_results("fwi"))
        act(m, "Residual-moveout velocity update results…", lambda: self.velocity_results("mva_tomography"))
        act(m, "Velocity semblance slice results…", lambda: self.velocity_results("velocity_semblance_slice"))
        act(m, "Fowler velocity panels…", lambda: self.velocity_results("fowler"))
        act(m, "Sliding shear rotation angles…", lambda: self.velocity_results("ltt_rotation"))

        m = mb.addMenu("&View")
        self.command_action = act(m, "Find a command…", self.show_commands, "Ctrl+K")
        for title, widget, shortcut in (("Home", self.home, "Ctrl+1"), ("Gather viewer", self.viewer, "Ctrl+2"),
                                        ("Processing flow", self.flow_editor, "Ctrl+3"),
                                        ("Velocity workspace", self.velan, "Ctrl+4")):
            act(m, title, lambda w=widget: self.tabs.setCurrentWidget(w), shortcut)
        m.addSeparator()
        act(m, "Spectra of the displayed traces", self.spectra, "F8")
        act(m, "Trace statistics and similarity…", lambda: self.panel_diagnostic("statistics"))
        act(m, "2D panel spectrum (k1-k2)…", lambda: self.panel_diagnostic("spectrum"))
        act(m, "Survey QC (geometry, fold, amplitudes)", self.survey_qc, "F9")
        act(m, "Open the dataset in a new viewer tab", self.new_viewer_tab)
        act(m, "Dataset summary and history", self.summary)
        m.addSeparator()
        m.addAction(self.project_dock.toggleViewAction())
        m.addAction(self.log_dock.toggleViewAction())
        act(m, "Reset workspace layout", self.reset_workspace)

        m = mb.addMenu("&Help")
        act(m, "User guide", self.user_guide, "F1")
        act(m, "User manual (PDF)", self.open_manual_pdf)
        act(m, "Process and parameter reference", self.process_reference)
        act(m, "About SeisDesk", self.about)

    def _build_toolbar(self):
        bar = self.addToolBar("Workspace")
        bar.setObjectName("workspace_toolbar")
        bar.setMovable(False)
        brand = QtWidgets.QLabel("SeisDesk"); brand.setProperty("brand", True)
        bar.addWidget(brand)
        for title, callback in (("Import", self.open_files), ("Open project", self.open_project),
                                ("Save", self.save_project)):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, fn=callback: fn())
            bar.addWidget(button)
        spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                                          QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        search = QtWidgets.QPushButton("Find a command   Ctrl+K")
        search.clicked.connect(self.show_commands); bar.addWidget(search)
        self.compute_button = QtWidgets.QPushButton("Compute settings")
        self.compute_button.clicked.connect(self.compute_devices); bar.addWidget(self.compute_button)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.setAccessibleName("Appearance")
        self.theme_combo.addItem("Light", "light"); self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(self.theme))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        bar.addWidget(self.theme_combo)
        self.project_status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.project_status)

    def _change_theme(self, _index):
        self.theme = apply_theme(self.theme_combo.currentData())
        self.ui_settings.setValue("appearance/theme", self.theme)

    def reset_workspace(self):
        self.project_dock.setFloating(False); self.log_dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.project_dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.project_dock.show(); self.log_dock.hide()
        self.resizeDocks([self.project_dock], [285], QtCore.Qt.Orientation.Horizontal)
        self.tabs.setCurrentWidget(self.home)

    def _recent_projects(self):
        try:
            paths = json.loads(self.ui_settings.value("projects/recent", "[]"))
            return [p for p in paths if isinstance(p, str)][:8] if isinstance(paths, list) else []
        except (TypeError, ValueError):
            return []

    def _remember_project(self, path):
        if path:
            path = os.path.abspath(path)
            paths = [path] + [p for p in self._recent_projects() if os.path.normcase(p) != os.path.normcase(path)]
            self.ui_settings.setValue("projects/recent", json.dumps(paths[:8]))

    def open_recent_project(self, path):
        if not self._ask_save_changes():
            return
        try:
            with busy():
                project = Project.load(path)
            self._set_project(project)
            self.log(f"opened project {path}")
        except Exception as exc:
            show_error(self, "Open recent project", exc)

    def command_entries(self):
        entries = []
        def collect(menu, prefix=""):
            for action in menu.actions():
                title = action.text().replace("&", "")
                if action.menu():
                    collect(action.menu(), f"{prefix}{title} / ")
                elif not action.isSeparator() and action is not self.command_action:
                    detail = prefix.rstrip(" / ")
                    if not action.shortcut().isEmpty():
                        detail += "  ·  " + action.shortcut().toString()
                    entries.append(Command(title, detail, action.trigger, action.isEnabled()))
        collect(self.menuBar())
        for name in self.project.datasets:
            entries.append(Command(f"Show dataset: {name}", "Project / Datasets", lambda n=name: self.show_dataset(n)))
        for name, cls in REGISTRY.items():
            entries.append(Command(f"Add to flow: {cls.label}", f"Processes / {cls.category} / {name}",
                                   lambda n=name: self._add_flow_process(n), self.flow_editor._worker is None))
        return entries

    def show_commands(self):
        dialog = CommandPalette(self.command_entries(), self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.chosen:
            dialog.chosen.run()

    def _add_flow_process(self, name):
        self.flow_editor.add_process(name)
        self.tabs.setCurrentWidget(self.flow_editor)

    def _filter_datasets(self, *_args):
        text = self.dataset_search.text().strip().casefold()
        count = 0
        for i in range(self.dataset_list.count()):
            item = self.dataset_list.item(i)
            hit = text in item.data(ROLE).casefold()
            item.setHidden(not hit); count += int(hit)
        self.dataset_empty.setText("No matching datasets." if self.project.datasets else "Import a seismic file to get started.")
        self.dataset_empty.setVisible(count == 0)

    def _flow_edited(self):
        self.dirty = True
        self._refresh_workspace_status()

    def _refresh_workspace_status(self):
        if not hasattr(self, "project_status"):
            return
        title = os.path.basename(self.project.path) if self.project.path else "Untitled project"
        self.project_status.setText(f"{title}  ·  {'Unsaved changes' if self.dirty else 'Saved' if self.project.path else 'Not saved'}")
        self.project_status.setToolTip(self.project.path or "Save the project to retain data, picks and processing flows.")
        self.setWindowTitle(f"{'* ' if self.dirty else ''}{title} — SeisDesk {__version__}")
        device = self.project.settings.get("compute_defaults", {}).get("device", "auto")
        self.compute_button.setText(f"Compute: {str(device).upper()}")
        self.compute_button.setToolTip("Configured preference. Open to inspect devices; each job reports its actual execution device.")

    def su_workbench(self):
        from .su_dialog import SUDialog
        SUDialog(self.project, self).exec()
        self.dirty = True

    def volume_job(self):
        from .volume_dialog import VolumeJobDialog
        dialog = VolumeJobDialog(self.project, self)
        dialog.exec()
        self.dirty = True
        if dialog.result_name:
            self._flow_result(dialog.result_name)

    def remote_job(self):
        from .remote_dialog import RemoteJobDialog
        dialog = RemoteJobDialog(self.project, self)
        dialog.exec()
        self.dirty = True
        if dialog.result_name:
            self._flow_result(dialog.result_name)

    def compute_devices(self):
        from .compute_dialog import ComputeDialog
        ComputeDialog(self.project, self).exec()
        self.dirty = True
        self._refresh_workspace_status()

    def create_volume(self):
        from .volume_dialog import VolumeModelDialog
        VolumeModelDialog(self.project, self).exec()
        self.dirty = True

    def inspect_volume(self):
        from ..core.volume import VelocityVolume
        from .volume_dialog import VolumeSlicesDialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "3D velocity model", "", "3D models (*.npz)")
        if path:
            try:
                VolumeSlicesDialog(VelocityVolume.load(path), parent=self).exec()
            except Exception as exc:
                show_error(self, "3D model", exc)

    def velocity_results(self, kind):
        from .velocity_results import FowlerResultsDialog, MVAResultsDialog, VelocitySliceDialog, ShearRotationResultsDialog
        result = self.project.context.results.get(kind)
        if result is None:
            show_error(self, 'Results', f"Run {kind.replace('_', ' ')} first.")
            return
        cls = {'fowler': FowlerResultsDialog, 'mva_tomography': MVAResultsDialog,
               'velocity_semblance_slice': VelocitySliceDialog, 'ltt_rotation': ShearRotationResultsDialog}[kind]
        cls(result, self).exec()

    def panel_diagnostic(self, kind):
        from .stats_dialog import TraceStatisticsDialog
        from .k1k2_dialog import WavenumberSpectrumDialog
        name = self._need_dataset()
        if name:
            cls = TraceStatisticsDialog if kind == "statistics" else WavenumberSpectrumDialog
            try:
                data = self.project.datasets[name] if kind == "statistics" else self.viewer.displayed_data()
                if data is not None:
                    cls(data, self).exec()
            except Exception as exc:
                show_error(self, "Panel diagnostic", exc)

    def avo_crossplot(self) -> None:
        from .avo_dialog import AvoCrossplotDialog
        result = self.project.context.results.get("avo_attributes")
        if result is None:
            show_error(self, "AVO crossplot", "Run AVO attributes first.")
            return
        AvoCrossplotDialog(result["A"], result["B"], self, dt=result["dt"], t0=result["t0"]).exec()

    def inversion_results(self, kind: str) -> None:
        from .fwi_dialog import FwiResultsDialog
        from .tomo_dialog import GRMResultsDialog, TomographyResultsDialog
        from .tomography3d_dialog import Tomography3DResultsDialog
        from .reconstruction_dialog import ReconstructionResultsDialog
        result = self.project.context.results.get(kind)
        if result is None:
            show_error(self, "Results", f"Run {kind.replace('_', ' ')} first.")
            return
        cls = {"fwi": FwiResultsDialog, "grm": GRMResultsDialog,
               "refraction_tomography": TomographyResultsDialog, "joint_tomography": TomographyResultsDialog,
               "joint_tomography_3d": Tomography3DResultsDialog, "survey_pocs": ReconstructionResultsDialog}[kind]
        cls(result, parent=self).exec()

    # ------------------------------------------------------------------ helpers
    @QtCore.Slot(str)
    def log(self, text: str) -> None:
        self.log_view.appendPlainText(str(text))

    @QtCore.Slot(str)
    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 10000)

    def current_dataset(self) -> str | None:
        item = self.dataset_list.currentItem()
        if item is not None:
            return item.data(ROLE)
        return self.viewer.dataset_name

    def refresh_project(self) -> None:
        current = self.current_dataset()
        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        for name, d in self.project.datasets.items():
            item = QtWidgets.QListWidgetItem(f"{name}\n{d.ntr:,} traces · {d.nt:,} samples")
            item.setData(ROLE, name)
            item.setToolTip(d.summary())
            self.dataset_list.addItem(item)
            if name == current:
                self.dataset_list.setCurrentItem(item)
        self.dataset_list.blockSignals(False)
        self.pick_list.clear()
        for name, ps in self.project.context.picks.items():
            self.pick_list.addItem(f"{name}: {ps.count()} picks, {len(ps.phase_order)} phases (by {ps.gather_key})")
        ctx = self.project.context
        vel = f"{len(ctx.velocity)} velocity picks at {ctx.velocity.cdps.size} CMPs" if ctx.velocity is not None \
            and len(ctx.velocity) else "no velocity picks"
        dm = (f"depth model {ctx.depth_model.shape[0]} x {ctx.depth_model.shape[1]} "
              f"(dz {ctx.depth_model.dz:g} m, dx {ctx.depth_model.dx:g} m)") if ctx.depth_model is not None \
            else "no depth model"
        where = self.project.path or "not saved yet"
        self.context_label.setText(f"{vel}<br>{dm}<br>project folder: {html.escape(where)}")
        self._filter_datasets()
        self.home.refresh(self.project, self._recent_projects())
        self._refresh_workspace_status()
        self.flow_editor.refresh_inputs(current)
        for v in self.viewers:
            if v.dataset_name is not None and v.dataset_name not in self.project.datasets:
                v.set_dataset(None)

    def _context_changed(self) -> None:
        self.dirty = True
        self.refresh_project()

    def _dataset_selected(self, item) -> None:
        if item is not None:
            self.flow_editor.refresh_inputs(item.data(ROLE))

    def show_dataset(self, name: str, viewer: GatherViewer | None = None) -> None:
        v = viewer or self.viewer
        v.set_dataset(name)
        self.tabs.setCurrentWidget(v)
        for i in range(self.dataset_list.count()):
            if self.dataset_list.item(i).data(ROLE) == name:
                self.dataset_list.blockSignals(True)
                self.dataset_list.setCurrentRow(i)
                self.dataset_list.blockSignals(False)
                break

    def add_dataset(self, data, name=None, show=True) -> str:
        key = self.project.add_dataset(data, name)
        self.dirty = True
        self.refresh_project()
        self.log(data.summary())
        if show:
            self.show_dataset(key)
        return key

    def _flow_result(self, name: str) -> None:
        self.dirty = True
        self.refresh_project()
        self.show_dataset(name)

    def _tab_changed(self, index: int) -> None:
        w = self.tabs.widget(index)
        if w is self.velan and (self.velan.data is None or self.velan.dataset_name not in self.project.datasets):
            name = self.current_dataset()
            if name:
                self.velan.set_dataset(name)
        if w is self.flow_editor:
            self.flow_editor.refresh_inputs(self.current_dataset())

    def _close_tab(self, index: int) -> None:
        w = self.tabs.widget(index)
        if w in (self.home, self.viewer, self.flow_editor, self.velan):
            return
        if w in self.viewers:
            self.viewers.remove(w)
        self.tabs.removeTab(index)
        w.deleteLater()

    def _need_dataset(self) -> str | None:
        name = self.current_dataset()
        if not name or name not in self.project.datasets:
            show_error(self, "SeisDesk", "Select a dataset in the Project panel first.")
            return None
        return name

    # ------------------------------------------------------------------ files
    def open_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open seismic files", self.project.context.work_dir,
                                                          SEISMIC_FILTER)
        self._open_paths(paths)

    def _open_paths(self, paths):
        from ..core.segd import is_segd
        from .dialogs import SEGDImportDialog
        for path in paths:
            if is_seg2(path):
                dlg = SEG2ImportDialog(parent=self)
                if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    self.read_file({"path": path, "little": False, "dt": 0.0, "name": None,
                                    "seg2": dlg.options()})
                continue
            dlg = (SEGDImportDialog if is_segd(path) or path.lower().endswith((".segd", ".sgd")) else ImportDialog)(path, self)
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                continue
            self.read_file(dlg.options())

    def _drop_paths(self, mime):
        if not mime.hasUrls():
            return []
        urls = mime.urls()
        if not urls or not all(url.isLocalFile() for url in urls):
            return []
        paths = [url.toLocalFile() for url in urls]
        if len(paths) == 1 and os.path.isfile(os.path.join(paths[0], "project.json")):
            return paths
        suffixes = (".sgy", ".segy", ".su", ".dat", ".sg2", ".seg2", ".segd", ".sgd")
        return paths if all(os.path.isfile(p) and p.lower().endswith(suffixes) for p in paths) else []

    def dragEnterEvent(self, event):
        if self._drop_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._drop_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        if len(paths) == 1 and os.path.isdir(paths[0]):
            self.open_recent_project(paths[0])
        else:
            self._open_paths(paths)

    def open_gpr(self):
        from ..core.gpr import read_dt1
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "GPR data with matching HD header", "", "DT1 (*.DT1 *.dt1)")
        if path:
            start_worker(self, lambda _w: read_dt1(path), self._file_read, self._file_failed)

    def open_seg2(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "SEG-2 shot files of one line",
                                                          self.project.context.work_dir, SEG2_FILTER)
        if not paths:
            return
        dialog = SEG2ImportDialog(len(paths), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        paths = sorted(paths)
        self.log(f"reading {len(paths)} SEG-2 files ...")
        self.status(f"reading {len(paths)} SEG-2 files ...")
        start_worker(self, lambda _w: read_seg2_files(paths, **options), self._file_read, self._file_failed)

    def read_file(self, opt: dict) -> None:
        path = opt["path"]
        self.log(f"reading {path} ...")
        self.status(f"reading {os.path.basename(path)} ...")

        def job(_worker):
            timing = dict(target_dt=opt.get("dt") or None, window=opt.get("window", "strict"))
            if is_seg2(path):
                data = read_seg2(path, opt["name"], **opt.get("seg2", {}))
            elif "segd" in opt:
                from ..core.segd import read_segd
                data = read_segd(path, opt["name"], **timing, **opt["segd"])
            elif path.lower().endswith(".su"):
                data = read_su(path, opt["name"], byteorder="<" if opt["little"] else ">", **timing)
            else:
                data = read_segy(path, opt["name"], endian="little" if opt["little"] else "big", **timing)
            return data

        start_worker(self, job, self._file_read, self._file_failed)

    @QtCore.Slot(object)
    def _file_read(self, data) -> None:
        self.add_dataset(data)
        self.status(f"loaded {data.name}: {data.ntr} traces")

    @QtCore.Slot(str)
    def _file_failed(self, text: str) -> None:
        self.log(text)
        show_error(self, "Reading failed", text)

    def new_synthetic(self) -> None:
        dlg = SyntheticDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            with busy():
                data = dlg.make()
            self.add_dataset(data)

    def export_dataset(self, kind: str) -> None:
        name = self._need_dataset()
        if name is None:
            return
        data = self.project.datasets[name]
        ext = ".sgy" if kind == "segy" else ".su"
        default = os.path.join(self.project.context.work_dir, name.replace(">", "-").replace(" ", "_") + ext)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export dataset", default,
                                                        "SEG-Y (*.sgy *.segy)" if kind == "segy" else
                                                        "Seismic Unix (*.su)")
        if not path:
            return
        try:
            with busy():
                (write_segy if kind == "segy" else write_su)(data, path)
            self.log(f"wrote {path}")
            self.status(f"wrote {path}")
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Export", exc)

    # ------------------------------------------------------------------ project
    def _ask_save_changes(self) -> bool:
        """True when it is safe to discard the current project."""
        if self.flow_editor._worker is not None:
            self.status("Cancel the running flow or wait for it to finish before changing projects.")
            return False
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "SeisDesk", "Save the current project first?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard |
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _set_project(self, project: Project) -> None:
        self.project = project
        project.context.log = self.log
        if not project.path:
            project.context.work_dir = default_work_dir()
        for w in [*self.viewers, self.flow_editor, self.velan]:
            w.project = project
        self.flow_editor.form.project = project
        for v in self.viewers:
            v.picking.project = project
            v.set_dataset(None)
            v.picking.refresh_sets()
        self.velan.set_dataset(None)
        self.dataset_search.clear()
        self._remember_project(project.path)
        self.flow_editor.set_flow(next(iter(project.flows.values())) if project.flows else Flow([], "flow"))
        self.flow_editor.reset_history()
        self.dirty = False
        self.refresh_project()
        if project.datasets:
            self.show_dataset(next(iter(project.datasets)))
        else:
            self.tabs.setCurrentWidget(self.home)
        self.dirty = False
        self._refresh_workspace_status()

    def new_project(self) -> None:
        if self._ask_save_changes():
            self._set_project(Project())

    def open_project(self) -> None:
        if not self._ask_save_changes():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder", self.project.context.work_dir)
        if not path:
            return
        if not os.path.exists(os.path.join(path, "project.json")):
            show_error(self, "Open project", "That folder has no project.json (not a SeisDesk project).")
            return
        try:
            with busy():
                project = Project.load(path)
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Open project", exc)
            return
        self._set_project(project)
        self.log(f"opened project {path}")

    def save_project(self, ask: bool = False) -> bool:
        path = self.project.path
        if ask or not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose (or create) an empty project folder",
                                                              self.project.context.work_dir)
            if not path:
                return False
        self.project.flows = {self.flow_editor.flow.name or "flow": self.flow_editor.flow} if self.flow_editor.flow.steps else {}
        try:
            with busy():
                self.project.save(path)
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Save project", exc)
            return False
        self.dirty = False
        self._remember_project(path)
        self.refresh_project()
        self.log(f"project saved in {path}")
        self.status(f"project saved in {path}")
        return True

    def closeEvent(self, event) -> None:                  # noqa: N802 - Qt API
        if not self._ask_save_changes():
            event.ignore()
            return
        for owner in (self, self.flow_editor):            # never destroy a running worker thread
            for thread, worker in getattr(owner, "_seisdesk_threads", []):
                try:
                    if thread.isRunning():
                        worker.cancel()
                        thread.quit()
                        thread.wait(10000)
                except RuntimeError:
                    pass
        self.ui_settings.setValue("window/geometry", self.saveGeometry())
        self.ui_settings.setValue("window/layout", self.saveState())
        event.accept()

    # ------------------------------------------------------------------ dataset actions
    def _dataset_menu(self, pos) -> None:
        item = self.dataset_list.itemAt(pos)
        if item is None:
            return
        name = item.data(ROLE)
        menu = QtWidgets.QMenu(self)
        entries = (("Show", lambda: self.show_dataset(name)),
                   ("Show in a new viewer tab", lambda: self.new_viewer_tab(name)),
                   ("Velocity analysis", lambda: self.open_velan(name)),
                   (None, None),
                   ("Rename...", lambda: self.rename_dataset(name)),
                   ("Delete", lambda: self.delete_dataset(name)),
                   (None, None),
                   ("Export as SEG-Y...", lambda: self._export_named(name, "segy")),
                   ("Export as Seismic Unix...", lambda: self._export_named(name, "su")),
                   (None, None),
                   ("Header table", lambda: HeaderTableDialog(self.project.datasets[name], self).exec()),
                   ("Summary and history", lambda: self.summary(name)))
        for text, fn in entries:
            if text is None:
                menu.addSeparator()
            else:
                menu.addAction(text).triggered.connect(lambda _c=False, f=fn: f())
        menu.exec(self.dataset_list.mapToGlobal(pos))

    def _export_named(self, name: str, kind: str) -> None:
        self.dataset_list.setCurrentRow([self.dataset_list.item(i).data(ROLE)
                                         for i in range(self.dataset_list.count())].index(name))
        self.export_dataset(kind)

    def rename_dataset(self, name: str) -> None:
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename dataset", "New name:", text=name)
        if ok and new.strip() and new.strip() != name:
            key = self.project.rename_dataset(name, new.strip())
            for v in [*self.viewers]:
                if v.dataset_name == name:
                    v.dataset_name = key
                    v.dataset_label.setText(f"<b>{key}</b>")
            if self.velan.dataset_name == name:
                self.velan.dataset_name = key
            self.dirty = True
            self.refresh_project()

    def delete_dataset(self, name: str) -> None:
        if QtWidgets.QMessageBox.question(self, "Delete dataset", f"Delete '{name}' from the project?") == \
                QtWidgets.QMessageBox.StandardButton.Yes:
            self.project.remove_dataset(name)
            if self.velan.dataset_name == name:
                self.velan.set_dataset(None)
            self.dirty = True
            self.refresh_project()

    def new_viewer_tab(self, name: str | None = None) -> None:
        name = name or self._need_dataset()
        if not name:
            return
        v = self._new_viewer()
        self.tabs.addTab(v, name if len(name) < 28 else name[:25] + "...")
        self.show_dataset(name, v)

    def single_step(self, process: str) -> None:
        name = self._need_dataset()
        if name is None:
            return
        if process in ("joint_tomography_3d", "survey_pocs"):
            from .arrival_survey_job import ArrivalSurveyJobDialog
            dlg = ArrivalSurveyJobDialog(self.project, process, name, self)
        else:
            dlg = SingleStepDialog(self.project, process, name, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.result_name:
            for line in getattr(dlg, "logs", []):
                self.log(line)
            self.dirty = True
            self.refresh_project()
            for v in self.viewers:
                if v.dataset_name == dlg.result_name:
                    v.reload()
            self.show_dataset(dlg.result_name)

    def choose_step(self) -> None:
        names = sorted(REGISTRY, key=lambda n: (REGISTRY[n].category, REGISTRY[n].label))
        labels = [f"{REGISTRY[n].category}: {REGISTRY[n].label}" for n in names]
        text, ok = QtWidgets.QInputDialog.getItem(self, "Apply one step", "Process:", labels, 0, False)
        if ok:
            self.single_step(names[labels.index(text)])

    def header_table(self) -> None:
        name = self._need_dataset()
        if name:
            HeaderTableDialog(self.project.datasets[name], self).exec()

    def summary(self, name: str | None = None) -> None:
        name = name or self._need_dataset()
        if not name:
            return
        d = self.project.datasets[name]
        text = d.summary() + "\n\nProcessing history:\n" + "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(d.history))
        if d.meta.get("textual_header"):
            text += "\n\nTextual header:\n" + d.meta["textual_header"]
        TextDialog(f"Dataset: {name}", text, self).exec()

    def survey_qc(self) -> None:
        name = self._need_dataset()
        if name:
            SurveyQCDialog(self.project.datasets[name], self).exec()

    def refraction_statics(self) -> None:
        self.single_step("refraction_statics")
        if "refraction_statics" in self.project.context.results:
            self.show_refraction_statics()

    def show_refraction_statics(self) -> None:
        result = self.project.context.results.get("refraction_statics")
        if not result:
            show_error(self, "Refraction statics", "Run Processing > Refraction statics (delay-time inversion) first.")
            return
        RefractionStaticsDialog(result, self).exec()

    def masw(self) -> None:
        name = self._need_dataset()
        if name:
            from .masw_dialog import MaswDialog
            MaswDialog(self.project.datasets[name], self, project=self.project).exec()
            self.dirty = True

    def surface_wave_inversion(self):
        name = self._need_dataset()
        if name:
            from .surface_inversion_dialog import SurfaceInversionDialog
            SurfaceInversionDialog(self.project, name, parent=self).exec()
            self.dirty = True

    def time_frequency(self) -> None:
        v = self.tabs.currentWidget() if isinstance(self.tabs.currentWidget(), GatherViewer) else self.viewer
        data = v.visible_data()
        if data is None:
            show_error(self, "Time-frequency analysis", "Display a dataset first.")
            return
        from .timefreq_dialog import TimeFrequencyDialog
        TimeFrequencyDialog(data, self).exec()

    def well_logs(self) -> None:
        from .well_dialog import WellLogDialog
        WellLogDialog(self.project, self).exec()

    def spectra(self) -> None:
        v = self.tabs.currentWidget() if isinstance(self.tabs.currentWidget(), GatherViewer) else self.viewer
        data = v.visible_data()
        if data is None:
            show_error(self, "Spectra", "Display a dataset first.")
            return
        SpectrumDialog(data, self).exec()

    # ------------------------------------------------------------------ velocity and picking
    def open_velan(self, name: str | None = None) -> None:
        name = name or self._need_dataset()
        if name:
            self.velan.set_dataset(name)
            self.tabs.setCurrentWidget(self.velan)

    def open_picking(self) -> None:
        self.tabs.setCurrentWidget(self.viewer)
        self.viewer.pick_btn.setChecked(True)

    def velocity_field(self) -> None:
        ctx = self.project.context
        if ctx.velocity is None or len(ctx.velocity) == 0:
            show_error(self, "Velocity field", "There are no velocity picks yet.")
            return
        name = self.current_dataset()
        d = self.project.datasets.get(name) if name else None
        cdps = d.header("cdp") if d is not None and d.has_header("cdp") else ctx.velocity.cdps
        times = d.times if d is not None and d.domain == "time" else np.linspace(0.0, 4.0, 1001)
        VelocityFieldDialog(ctx.velocity, cdps, times, self).exec()

    def load_velocity(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load velocity picks", self.project.context.work_dir,
                                                        "Velocity picks (*.txt *.vel);;All files (*)")
        if path:
            try:
                self.project.context.velocity = VelocityPicks.load(path)
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Load velocity picks", exc)
                return
            self._context_changed()
            self.velan.draw_gathers()

    def save_velocity(self) -> None:
        ctx = self.project.context
        if ctx.velocity is None or len(ctx.velocity) == 0:
            show_error(self, "Save velocity picks", "There are no velocity picks yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save velocity picks",
                                                        os.path.join(ctx.work_dir, "velocity.txt"),
                                                        "Velocity picks (*.txt)")
        if path:
            ctx.velocity.save(path)

    def clear_velocity(self) -> None:
        if QtWidgets.QMessageBox.question(self, "Velocity picks", "Delete every velocity pick?") == \
                QtWidgets.QMessageBox.StandardButton.Yes:
            self.project.context.velocity = VelocityPicks()
            self._context_changed()
            self.velan.draw_gathers()

    def load_depth_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load depth model", self.project.context.work_dir,
                                                        "Depth model (*.npz)")
        if path:
            try:
                self.project.context.depth_model = DepthModel.load(path)
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Load depth model", exc)
                return
            self._context_changed()

    def save_depth_model(self) -> None:
        dm = self.project.context.depth_model
        if dm is None:
            show_error(self, "Save depth model", "There is no depth model yet (Processing: Build depth model).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save depth model",
                                                        os.path.join(self.project.context.work_dir, "depth_model.npz"),
                                                        "Depth model (*.npz)")
        if path:
            dm.save(path)

    # ------------------------------------------------------------------ help
    def user_guide(self) -> None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "docs", "USER_GUIDE.md")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SeisDesk user guide")
        dlg.resize(900, 700)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setSearchPaths([os.path.join(here, "docs")])     # the guide's screenshots in docs/images
        try:
            with open(path, encoding="utf-8") as f:
                browser.setMarkdown(f.read())
        except OSError:
            browser.setPlainText(f"The user guide was not found at {path}.")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(browser)
        dlg.exec()

    def open_manual_pdf(self) -> None:
        import sys
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for path in (os.path.join(base, "SeisDesk_User_Manual.pdf"),
                     os.path.join(base, "dist", "SeisDesk_User_Manual.pdf")):
            if os.path.exists(path):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
                return
        show_error(self, "User manual", "The PDF manual was not found; Help > User guide shows the same text.")

    def process_reference(self) -> None:
        from ..processing.reference import process_reference_html
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Process and parameter reference")
        dlg.resize(900, 700)
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(process_reference_html())
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(browser)
        dlg.exec()

    def save_image(self) -> None:
        """Save the current display (the viewer plot, or the whole velocity-analysis / flow tab) as an image."""
        w = self.tabs.currentWidget()
        target = w.plot if isinstance(w, GatherViewer) else w
        name = (w.dataset_name if isinstance(w, GatherViewer) and w.dataset_name else "display").replace(">", "-")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save display as image", os.path.join(self.project.context.work_dir, f"{name}.png"),
            "PNG image (*.png);;JPEG image (*.jpg)")
        if not path:
            return
        if target.grab().save(path):
            self.status(f"image saved: {path}")
            self.log(f"image saved: {path}")
        else:
            show_error(self, "Save image", f"Could not write {path}")

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, "About SeisDesk",
            f"<b>SeisDesk {__version__}</b><br>Interactive 2D/3D seismic data processing: SEG-Y and SU input/output, "
            "gather display, geometry, editing, mutes, gain, filtering, deconvolution, Radon demultiple, statics, "
            "velocity analysis with VTI eta, NMO, DMO, residual moveout, stacking, time and depth migration, "
            "first-break and refraction analysis.<br><br>Author: Deepak Kumar. Licence: GPL-3.0-or-later.")

"""Editable dispersion picks/layers, cancellable inversion and persisted model QC."""
import csv
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.surface_inversion import invert_dispersion, validate_model, validate_picks
from .common import show_error, start_worker


class SurfaceInversionDialog(QtWidgets.QDialog):
    def __init__(self, project, dataset, picks=None, parent=None):
        super().__init__(parent)
        self.project, self.dataset = project, dataset
        self.result_key = "surface_wave:"+dataset
        self.worker=None; self.cancel_requested=False; self.close_after_cancel=False
        self.result=project.context.results.get(self.result_key)
        self.setWindowTitle("Layered surface-wave inversion — "+dataset); self.resize(1180,800)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel("Rayleigh/Love phase or group velocities • identified modes • isotropic horizontal solid layers. "
                              "Vp/Vs, density and layer count stay fixed. Last row is the half-space (thickness 0). "
                              "Forward engine: disba/CPS. Uncertainty is conditional on these assumptions.")
        note.setWordWrap(True); layout.addWidget(note)
        self.tabs=QtWidgets.QTabWidget(); layout.addWidget(self.tabs,1)
        model_tab=QtWidgets.QWidget(); ml=QtWidgets.QVBoxLayout(model_tab)
        self.layers=QtWidgets.QTableWidget(0,8)
        self.layers.setHorizontalHeaderLabels(["Thickness m","Vp m/s","Vs m/s","Density kg/m³","Vs min","Vs max","h min m","h max m"])
        self.layers.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        ml.addWidget(self.layers)
        row=QtWidgets.QHBoxLayout()
        for label,fn in (("Add layer above half-space",self.add_layer),("Remove selected finite layer",self.remove_layer)):
            b=QtWidgets.QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        ml.addLayout(row); self.tabs.addTab(model_tab,"Layer model and bounds")
        pick_tab=QtWidgets.QWidget(); pl=QtWidgets.QVBoxLayout(pick_tab)
        self.picks=QtWidgets.QTableWidget(0,6)
        self.picks.setHorizontalHeaderLabels(["Frequency Hz","Velocity m/s","Error σ m/s","Mode (0 = fundamental)","rayleigh / love","phase / group"])
        self.picks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        pl.addWidget(self.picks)
        row=QtWidgets.QHBoxLayout()
        for label,fn in (("Import picks CSV…",self.import_picks),("Add pick",lambda:self.add_pick()),("Remove selected pick",self.remove_pick)):
            b=QtWidgets.QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        pl.addLayout(row); self.tabs.addTab(pick_tab,"Dispersion picks")
        result_tab=QtWidgets.QWidget(); rl=QtWidgets.QVBoxLayout(result_tab)
        self.summary=QtWidgets.QLabel("No inversion result yet."); self.summary.setWordWrap(True); rl.addWidget(self.summary)
        plots=QtWidgets.QSplitter()
        self.dispersion=pg.PlotWidget(title="Observed and fitted dispersion"); self.profile=pg.PlotWidget(title="Layered Vs")
        plots.addWidget(self.dispersion); plots.addWidget(self.profile); rl.addWidget(plots,1)
        self.qc=QtWidgets.QTableWidget(0,5); self.qc.setHorizontalHeaderLabels(["Frequency Hz","Mode / wave / kind","Observed m/s","Predicted m/s","Residual / σ"])
        self.qc.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.qc.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers); self.qc.setMaximumHeight(190); rl.addWidget(self.qc)
        b=QtWidgets.QPushButton("Export model, picks and uncertainty…"); b.clicked.connect(self.export_result); rl.addWidget(b)
        self.tabs.addTab(result_tab,"Fit and uncertainty")
        options=QtWidgets.QHBoxLayout()
        self.fit_h=QtWidgets.QCheckBox("Fit finite-layer thicknesses")
        self.starts=QtWidgets.QSpinBox(); self.starts.setRange(1,30); self.starts.setValue(3)
        self.bootstrap=QtWidgets.QSpinBox(); self.bootstrap.setRange(0,500)
        self.bootstrap.setToolTip("Gaussian pick-error refits; use many draws to assess interval stability. 0 disables uncertainty.")
        self.evaluations=QtWidgets.QSpinBox(); self.evaluations.setRange(10,2000); self.evaluations.setValue(150)
        self.damping=QtWidgets.QDoubleSpinBox(); self.damping.setDecimals(3); self.damping.setRange(0,100); self.damping.setValue(.05)
        options.addWidget(self.fit_h)
        for label,widget in (("Starts",self.starts),("Noise refits",self.bootstrap),("Max evaluations",self.evaluations),("Prior weight",self.damping)):
            options.addWidget(QtWidgets.QLabel(label)); options.addWidget(widget)
        self.option_box=QtWidgets.QWidget(); self.option_box.setLayout(options); layout.addWidget(self.option_box)
        self.progress=QtWidgets.QProgressBar(); layout.addWidget(self.progress)
        self.status=QtWidgets.QLabel("Edit layers and picks, then run. First execution compiles the dispersion kernels.")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        buttons=QtWidgets.QHBoxLayout(); self.run_button=QtWidgets.QPushButton("Run inversion"); self.close_button=QtWidgets.QPushButton("Close")
        self.run_button.clicked.connect(self.run); self.close_button.clicked.connect(self.reject)
        buttons.addStretch(); buttons.addWidget(self.run_button); buttons.addWidget(self.close_button); layout.addLayout(buttons)
        setup=project.settings.get(self.result_key,{})
        layer_rows=setup.get("layers",[[10,400,200,1800,80,400,3,30],[0,800,400,2000,200,800,0,0]])
        for values in layer_rows: self.add_values(self.layers,values)
        for pick in (picks if picks is not None else setup.get("picks",[])): self.add_pick(pick)
        self.fit_h.setChecked(setup.get("fit_h",False))
        for name in ("starts","bootstrap","evaluations","damping"):
            if name in setup: getattr(self,name).setValue(setup[name])
        if self.result is not None: self.render_result()

    @staticmethod
    def add_values(table,values,index=None):
        index=table.rowCount() if index is None else index
        table.insertRow(index)
        for col,v in enumerate(values): table.setItem(index,col,QtWidgets.QTableWidgetItem(str(v)))

    def add_layer(self):
        self.add_values(self.layers,[10,600,300,1900,120,600,2,30],max(0,self.layers.rowCount()-1))

    def remove_layer(self):
        row=self.layers.currentRow()
        if 0 <= row < self.layers.rowCount()-1: self.layers.removeRow(row)

    def add_pick(self,pick=None):
        pick=pick or dict(frequency_hz=10,velocity_ms=250,sigma_ms=10,mode=0,wave="rayleigh",kind="phase")
        self.add_values(self.picks,[pick.get(k,default) for k,default in
                                  (("frequency_hz",10),("velocity_ms",250),("sigma_ms",10),("mode",0),("wave","rayleigh"),("kind","phase"))])

    def remove_pick(self):
        if self.picks.currentRow() >= 0: self.picks.removeRow(self.picks.currentRow())

    def values(self):
        layers=np.array([[float(self.layers.item(i,j).text()) for j in range(8)] for i in range(self.layers.rowCount())])
        model=validate_model(layers[:,:4])
        keys=("frequency_hz","velocity_ms","sigma_ms","mode","wave","kind")
        picks=validate_picks([dict(zip(keys,[self.picks.item(i,j).text().strip() for j in range(6)])) for i in range(self.picks.rowCount())])
        return model,picks,layers

    def import_picks(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Dispersion picks","","CSV (*.csv)")
        if not path:return
        try:
            with open(path,encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
            for row in rows:
                if "velocity_ms" not in row and "phase_velocity_ms" in row: row["velocity_ms"]=row["phase_velocity_ms"]
            rows=validate_picks(rows)
            self.picks.setRowCount(0)
            for row in rows:self.add_pick(row)
        except Exception as exc:show_error(self,"Import dispersion picks",exc)

    def run(self):
        if self.worker is not None:return
        try:model,picks,layers=self.values()
        except Exception as exc:show_error(self,"Surface-wave model/picks",exc);return
        options=dict(max_evaluations=self.evaluations.value(),starts=self.starts.value(),bootstrap=self.bootstrap.value(),damping=self.damping.value())
        hb=layers[:-1,6:8] if self.fit_h.isChecked() else None
        self.project.settings[self.result_key]=dict(layers=layers.tolist(),picks=picks,fit_h=self.fit_h.isChecked(),
                                                     starts=self.starts.value(),bootstrap=self.bootstrap.value(),
                                                     evaluations=self.evaluations.value(),damping=self.damping.value())
        self.cancel_requested=False; self.close_after_cancel=False
        self.run_button.setEnabled(False); self.option_box.setEnabled(False); self.tabs.setEnabled(False)
        self.close_button.setText("Cancel run"); self.progress.setValue(0); self.status.setText("Computing dispersion and fitting the model…")
        def job(worker):
            return invert_dispersion(model,picks,layers[:,4:6],hb,**options,
                                       cancelled=worker.cancelled,progress=worker.progress.emit)
        self.worker=start_worker(self,job,self.finished_job,self.failed,on_progress=self.update_progress)

    @QtCore.Slot(float)
    def update_progress(self,fraction):
        self.progress.setValue(round(100*fraction))

    def threads_running(self):
        for thread,_ in getattr(self,"_seisdesk_threads",[]):
            try:
                if thread.isRunning():return True
            except RuntimeError:pass
        return False

    def restore(self):
        self.worker=None; self.run_button.setEnabled(True); self.option_box.setEnabled(True); self.tabs.setEnabled(True)
        self.close_button.setText("Close"); self.close_button.setEnabled(True)

    @QtCore.Slot(object)
    def finished_job(self,result):
        if self.threads_running():QtCore.QTimer.singleShot(10,lambda:self.finished_job(result));return
        if self.cancel_requested:self.failed("Cancelled; previous saved result retained.");return
        self.restore(); self.result=result; self.project.context.results[self.result_key]=result
        self.render_result(); self.tabs.setCurrentIndex(2); self.progress.setValue(100)
        self.status.setText("Result stored in the project. Save the project to keep the picks, bounds, model and uncertainty.")

    @QtCore.Slot(str)
    def failed(self,message):
        if self.threads_running():QtCore.QTimer.singleShot(10,lambda:self.failed(message));return
        self.restore(); self.status.setText(message)
        if self.close_after_cancel:super().reject()

    def render_result(self):
        r=self.result; self.dispersion.clear(); self.profile.clear()
        self.summary.setText(f"RMS {r['initial_rms_ms']:.3f} → {r['rms_ms']:.3f} m/s. "
                             f"Solver: {r['message']}  Noise refits: {r['bootstrap_success']}/{r['bootstrap_requested']} successful. "
                             f"{r['uncertainty']}")
        self.dispersion.setLabel("bottom","Frequency",units="Hz"); self.dispersion.setLabel("left","Velocity",units="m/s")
        groups=sorted({(p['wave'],p['kind'],p['mode']) for p in r['picks']})
        self.dispersion.addLegend()
        for j,group in enumerate(groups):
            ids=sorted([i for i,p in enumerate(r['picks']) if (p['wave'],p['kind'],p['mode'])==group],key=lambda i:r['picks'][i]['frequency_hz'])
            f=np.array([r['picks'][i]['frequency_hz'] for i in ids]); obs=np.array([r['picks'][i]['velocity_ms'] for i in ids])
            color=pg.intColor(j,hues=max(len(groups),2)); pen=pg.mkPen(color,width=2)
            self.dispersion.plot(f,obs,pen=None,symbol='o',symbolBrush=color,symbolSize=6)
            self.dispersion.addItem(pg.ErrorBarItem(x=f,y=obs,height=2*np.array([r['picks'][i]['sigma_ms'] for i in ids]),pen=pen))
            self.dispersion.plot(f,np.asarray(r['predicted_ms'])[ids],pen=pen,name=f"{group[0]} {group[1]} mode {group[2]}")
        def profile(model):
            bottom=np.r_[np.cumsum(model[:-1,0]),sum(model[:-1,0])+max(10.,sum(model[:-1,0])*.5)]
            return np.repeat(model[:,2],2),np.ravel(np.column_stack((np.r_[0,bottom[:-1]],bottom)))
        for model,color in ((r['initial_model'],(160,160,160)),(r['model'],(20,80,190))):
            x,y=profile(np.asarray(model)); self.profile.plot(x,y,pen=pg.mkPen(color,width=2))
        for model in np.asarray(r.get('ensemble',[]))[:50]:
            x,y=profile(model); self.profile.plot(x,y,pen=pg.mkPen(30,160,130,50))
        self.profile.setLabel("bottom","Vs",units="m/s"); self.profile.setLabel("left","Depth",units="m"); self.profile.invertY(True)
        self.qc.setRowCount(0)
        for i,p in enumerate(r['picks']):
            self.add_values(self.qc,[f"{p['frequency_hz']:.3f}",f"{p['mode']} / {p['wave']} / {p['kind']}",
                                     f"{p['velocity_ms']:.3f}",f"{r['predicted_ms'][i]:.3f}",f"{r['normalized_residual'][i]:.2f}"])

    def export_result(self):
        if self.result is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,"Save inversion result","surface-wave.npz","NumPy bundle (*.npz)")
        if not path:return
        try:
            # The project result serializer retains nested pick metadata and NumPy arrays.
            import json
            r=self.result
            arrays={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
            arrays["metadata_json"]=np.array(json.dumps({k:v for k,v in r.items() if k not in arrays}))
            np.savez_compressed(path,**arrays)
            self.status.setText("Saved "+path)
        except Exception as exc:show_error(self,"Export inversion",exc)

    def reject(self):
        if self.worker is not None:
            self.cancel_requested=True; self.worker.cancel(); self.close_button.setEnabled(False)
            self.status.setText("Cancellation requested; waiting for the current dispersion evaluation.")
        else:super().reject()

    def closeEvent(self,event):
        if self.worker is not None:
            self.close_after_cancel=True; self.reject(); event.ignore()
        else:super().closeEvent(event)

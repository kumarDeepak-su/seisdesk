"""Render real v0.4 dialogs from computed synthetic examples on the Windows build runner.

Run with QT_QPA_PLATFORM=offscreen. Any failed computation or screenshot fails the build.
These are illustrative synthetic examples, not field-data benchmarks.
"""
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile

os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
if os.name == 'nt':
    os.environ.setdefault('QT_QPA_FONTDIR', str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.core.velocity import DepthModel
from seisdesk.core import welllog
from seisdesk.core.helmholtz import Helmholtz
from seisdesk.core.tomography import NodeGrid, first_arrivals, invert
from seisdesk.processing import Context, create
from seisdesk.processing.fwi import invert_frequency
from seisdesk.processing.grm import grm_functions
from seisdesk.processing.multicomponent import alford_rotate
from seisdesk.gui.avo_dialog import AvoCrossplotDialog
from seisdesk.gui.fwi_dialog import FwiResultsDialog
from seisdesk.gui.tomo_dialog import TomographyResultsDialog, GRMResultsDialog
from seisdesk.gui.velocity_results import (VelocitySliceDialog, FowlerResultsDialog,
                                           MVAResultsDialog, ShearRotationResultsDialog)
from seisdesk.gui.well_dialog import WellLogDialog
from seisdesk.gui.stats_dialog import TraceStatisticsDialog
from seisdesk.gui.k1k2_dialog import WavenumberSpectrumDialog


def main(destination):
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle('Fusion')
    if os.name == 'nt':
        font_path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'arial.ttf'
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            raise RuntimeError(f'could not load screenshot font: {font_path}')
        app.setFont(QtGui.QFont(families[0], 9))
        if not QtGui.QFontMetrics(app.font()).inFont('A'):
            raise RuntimeError('screenshot font has no Latin glyphs')
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True, background='w', foreground='k')

    def grab(widget, name, close=True):
        widget.show()
        for _ in range(8):
            app.processEvents()
        image = widget.grab()
        if image.isNull() or not image.save(str(out / name)):
            raise RuntimeError(f'could not render {name}')
        print(f'rendered {name}', flush=True)
        if close:
            widget.close()

    with tempfile.TemporaryDirectory() as tmp:
        z = np.arange(100., 601.)
        vp = 2000 + 150*np.sin(z/18); rho = 2200 + 100*np.sin(z/27)
        lines = ['~Version', 'VERS. 2.0', 'WRAP. NO', '~Well', 'NULL. -999.25',
                 '~Curve', 'DEPT.M : depth', 'VP.M/S : velocity', 'RHOB.KG/M3 : density', '~A']
        lines += [f'{a} {b} {c}' for a, b, c in zip(z, vp, rho)]
        las = Path(tmp)/'synthetic.las'; las.write_text('\n'.join(lines))
        syn = welllog.synthetic_seismogram(z, vp, rho, .002)
        times = np.arange(400)*.002
        trace = np.interp(times, syn['times']+.02, syn['synthetic'], left=0, right=0)
        section = SeismicData(np.tile(trace, (21, 1)), .002, {'cdpx': np.arange(21)*10, 'cdpy': 0})
        well = WellLogDialog(SimpleNamespace(datasets={'synthetic line': section}), las_path=str(las))
        well.set_seismic('synthetic line', 100); well.compute()
        well.wavelet_mode.setCurrentText('deterministic'); well.extract_wavelet()
        well.manual_points.setText('0.12:0.12,0.60:0.61'); well.manual_warp()
        well.tabs.setCurrentIndex(3); grab(well, 'v04_welltie.png', False)
        well.tabs.setCurrentIndex(4); grab(well, 'v04_multitrace.png', False)
        survey = Path(tmp)/'deviation.csv'; survey.write_text('0,0,90\n350,15,90\n700,30,90\n')
        well.project_md.setValue(600); well.load_deviation(str(survey))
        well.tabs.setCurrentIndex(5); grab(well, 'v04_deviation.png', False)
        well.compute(); well.avo_depth.setValue(300); well.compute_avo()
        well.tabs.setCurrentIndex(6); grab(well, 'v04_wellavo.png')

    # Invert synthetic first arrivals on a small gradient model.
    grid = NodeGrid(0, 10, 31, 10, 16, radius=3)
    pairs = [(grid.node_at(s, 0), grid.node_at(r, 0))
             for s in [0, 100, 200, 300] for r in range(0, 301, 20) if s != r]
    sources, receivers = np.array(pairs).T
    truth = np.broadcast_to(700+18*grid.z[:, None], (grid.nz, grid.nx)).copy()
    observed = first_arrivals(grid, truth, sources, receivers)[0]
    tomo = invert(grid, truth*.9, sources, receivers, observed, iterations=4,
                  smoothing=.5, damping=.05, vmin=300, vmax=5000)
    tomo.update(x=grid.x, z=grid.z, surface=grid.surface, observed=observed)
    grab(TomographyResultsDialog(tomo), 'v04_tomography.png')
    x = np.arange(0, 1001, 10.); delay = 20*np.sqrt(2000**2-500**2)/(500*2000)
    grm = grm_functions(x, 2*delay+x/2000, x, 2*delay+(1000-x)/2000,
                        2*delay+1000/2000, [0, 20, 40], 500)
    grab(GRMResultsDialog(grm), 'v04_grm.png')

    # Crosswell acoustic demonstration uses the actual solver and optimiser.
    start = np.full((20, 24), 2000.); truth = start.copy(); truth[5:15, 8:16] = 1850
    shots = [(np.array([20., 30.]), np.column_stack([np.full(8, 200.), np.linspace(20, 160, 8)])),
             (np.array([100., 20.]), np.column_stack([np.linspace(20, 210, 8), np.full(8, 170.)]))]
    frequencies = [6., 9.]; observations = []
    for f in frequencies:
        op = Helmholtz(truth, 10, 10, f, nabs=10)
        observations.append([op.sampling(rec) @ op.solve(op.source(src)) for src, rec in shots])
    final, info = invert_frequency(start, 10, 10, frequencies, shots, observations, iterations=5,
                                   vmin=1500, vmax=2500, smoothing=.001, nabs=10, estimate_source=False)
    info['frequencies'] = frequencies
    grab(FwiResultsDialog(dict(x=np.arange(24)*10, z=np.arange(20)*10,
                               starting_model=start, final_model=final, stages=[info])), 'v04_fwi.png')

    # Dipping reflector used for velocity scans/Fowler processing.
    cmp = np.repeat(np.arange(21), 6); x = (cmp-10)*20.
    offsets = np.tile(np.arange(6)*100., 21); dt = .004; nt = 201
    dip = np.deg2rad(15); zero = .35+2*np.sin(dip)*x/2000
    arrival = np.sqrt(zero**2+(offsets*np.cos(dip)/2000)**2)
    w, centre = ricker(25, dt)
    traces = events_to_traces(arrival[:, None], np.ones((len(cmp), 1)), nt, dt, w, centre)
    data = SeismicData(traces, dt, {'cdp': cmp+1, 'cdpx': x, 'offset': offsets, 'sx': x-offsets/2, 'gx': x+offsets/2})
    ctx = Context()
    create('velocity_semblance_slice', time=.35, vmin=1600, vmax=2600, nv=41)(data, ctx)
    grab(VelocitySliceDialog(ctx.results['velocity_semblance_slice']), 'v04_velocity_slice.png')
    create('fowler', vmin=1700, vmax=2300, nv=7, offset_bin=50, stretch=0, write_picks=False)(data, ctx)
    fowler = FowlerResultsDialog(ctx.results['fowler']); fowler.trial.setCurrentIndex(3)
    grab(fowler, 'v04_fowler.png')
    grab(TraceStatisticsDialog(data), 'v04_statistics.png')
    regular = data.subset(np.flatnonzero(data.header('offset') == 0))
    grab(WavenumberSpectrumDialog(regular), 'v04_panel_spectrum.png')

    z = np.arange(0, 1001, 2.); offsets = np.linspace(0, 800, 17)
    peaks = np.sqrt(500**2+(1.05**2-1)*(offsets/2)**2)
    gathers = SeismicData(np.exp(-.5*((z[None]-peaks[:, None])/4)**2), 2,
                          {'cdp': 1, 'cdpx': 500, 'offset': offsets}, domain='depth')
    ctx = Context(depth_model=DepthModel.constant(2100, 51, 51, 20, 20))
    create('mva_tomography', zmin=500, zmax=500, na=121, window=10)(gathers, ctx)
    grab(MVAResultsDialog(ctx.results['mva_tomography']), 'v04_mva.png')

    # Time-varying four-component rotation.
    t = np.arange(500)*.002
    fast = np.sin(2*np.pi*18*t); slow = .4*np.cos(2*np.pi*12*t)
    theta = np.deg2rad(np.where(t < .5, 20, 45))
    four = np.stack(alford_rotate(fast, np.zeros(500), np.zeros(500), slow, -theta))
    ctx = Context()
    create('ltt_rotation')(SeismicData(four, .002, {'sx': 0, 'gx': 100, 'comp': [1, 2, 3, 4]}), ctx)
    grab(ShearRotationResultsDialog(ctx.results['ltt_rotation']), 'v04_shear_rotation.png')

    rng = np.random.default_rng(72); a = rng.normal(0, .03, (100, 100))
    b = -1.2*a+rng.normal(0, .008, a.shape)
    grab(AvoCrossplotDialog(a, b, dt=.004), 'v04_avo_crossplot.png')
    print('All v0.4 diagnostic windows rendered.', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'seisdesk/docs/images')

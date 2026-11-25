"""Compare 3D homogeneous propagation with the analytic free-space Green function.

Run on a compute node, e.g. python tools/validate_3d_physics.py report.json --device cuda.
This checks travel times, waveform amplitude and spherical spreading independently
of the discrete adjoint tests. It does not validate heterogeneous field inversion.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from seisdesk.core.acoustic3d import Acoustic3D
from seisdesk.core.volume import VelocityVolume


def main(args):
    started = time.monotonic()
    velocity, dx, dt, frequency, delay = 2000., 5., .001, 12., .1
    volume = VelocityVolume(np.full((161, 161, 161), velocity, dtype=np.float32), (dx, dx, dx))
    source = np.array([[400., 400., 400.]])
    receivers = source + np.array([[0., 0., 100.], [0., 0., 200.], [0., 120., 160.], [120., 0., 160.]])
    times = np.arange(320)*dt
    def ricker(t):
        phase = np.pi*frequency*(t-delay)
        return 1e8*(1-2*phase**2)*np.exp(-phase**2)
    engine = Acoustic3D(volume, dt, padding=12, device=args.device)
    predicted = engine.forward(source, receivers, ricker(times))
    distances = np.linalg.norm(receivers-source, axis=1)
    # u_tt - c^2 Laplacian(u) = q(t) at one grid node. The equivalent
    # continuum point strength is q(t) times the grid-cell volume.
    analytic = ricker(times[:, None]-distances[None, :]/velocity)*dx**3/(4*np.pi*velocity**2*distances)
    metrics = []
    for i, distance in enumerate(distances):
        p, a = predicted[:, i], analytic[:, i]
        peak_time = float(times[np.argmax(p)])
        metrics.append(dict(distance_m=float(distance), expected_peak_s=float(delay+distance/velocity),
                            measured_peak_s=peak_time, peak_error_s=abs(peak_time-delay-distance/velocity),
                            correlation=float(np.corrcoef(p, a)[0, 1]),
                            relative_l2=float(np.linalg.norm(p-a)/np.linalg.norm(a)),
                            measured_peak=float(p.max()), analytic_peak=float(a.max())))
    spreading_ratio = float(predicted[:, 0].max()/predicted[:, 1].max())
    limits = dict(min_correlation=.99, max_relative_l2=.10, max_peak_error_s=.0021,
                  max_spreading_relative_error=.05)
    passed = (all(m["correlation"] >= limits["min_correlation"] and
                  m["relative_l2"] <= limits["max_relative_l2"] and
                  m["peak_error_s"] <= limits["max_peak_error_s"] for m in metrics)
              and abs(spreading_ratio/2-1) <= limits["max_spreading_relative_error"])
    report = dict(scope="homogeneous 3D free-space Green-function comparison", device=args.device,
                  grid_shape=list(volume.values.shape), spacing_m=dx, dt_s=dt, velocity_m_s=velocity,
                  source_frequency_hz=frequency, points_per_wavelength_at_2_5_f=velocity/(2.5*frequency*dx),
                  receivers=metrics, spreading_ratio_100m_to_200m=spreading_ratio,
                  expected_spreading_ratio=2., thresholds=limits, elapsed_seconds=time.monotonic()-started,
                  status="passed" if passed else "failed", heterogeneous_field_accuracy_validated=False)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit("3D propagation did not meet the predeclared analytic-reference thresholds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    main(parser.parse_args())

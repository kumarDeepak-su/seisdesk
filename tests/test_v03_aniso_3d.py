"""Tests of the version 0.3 anisotropic and 3D migration steps: VTI phase shift, Kirchhoff time migration with eta,
migration to zero offset (anisotropic DMO), residual-moveout scans, 3D phase-shift / Stolt / Kirchhoff migration and
Kirchhoff depth migration from topography."""

import numpy as np

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.core.velocity import VelocityPicks
from seisdesk.processing import REGISTRY, Context, create
from seisdesk.processing import anisotropy, migration_3d
from seisdesk.processing.anisotropy import interval_eta, phase_shift_vti
from seisdesk.processing.migration_time import kirchhoff_time, phase_shift
from seisdesk.processing.moveout import moveout_time
from seisdesk.processing.nmo import nmo_correct

NEW = ("phase_shift_vti", "mzo_dmo", "rmo_scan", "phase_shift_3d", "stolt_3d", "kirchhoff_pstm_3d")


def test_new_steps_registered():
    assert anisotropy.RMOScan.name == "rmo_scan" and migration_3d.Stolt3D.name == "stolt_3d"
    for name in NEW:
        assert name in REGISTRY
    assert "surface" in [p.name for p in REGISTRY["kirchhoff_psdm"].params]
    assert {"use_eta", "eta"} <= {p.name for p in REGISTRY["kirchhoff_pstm"].params}


# ---------------------------------------------------------------------------------------------- VTI time migration
def diffraction_section(nx=96, nt=256, dt=0.004, dx=12.5, v=2000.0, x0=600.0, tau0=0.4, eta=0.0, f0=25.0):
    """Zero-offset section of one point diffractor in a homogeneous VTI medium: t = 2 t_leg(tau0 / 2, |x - x0|)."""
    x = np.arange(nx) * dx
    t = 2.0 * moveout_time(0.5 * tau0, np.abs(x - x0), v, eta)
    w, c = ricker(f0, dt)
    tr = events_to_traces(t[:, None], np.ones((nx, 1)), nt, dt, w, c)
    return SeismicData(tr.astype(np.float32), dt, {"cdp": np.arange(1, nx + 1), "cdpx": x})


def peak_near(traces, x, times, x0, tau0, dx=40.0, dtau=0.03):
    sel_x = np.abs(x - x0) <= dx
    sel_t = np.abs(times - tau0) <= dtau
    block = np.abs(traces[np.ix_(sel_x, sel_t)])
    i, j = np.unravel_index(np.argmax(block), block.shape)
    return float(block.max()), float(x[sel_x][i]), float(times[sel_t][j])


def test_phase_shift_vti_eta_zero_matches_isotropic():
    sec = diffraction_section(nx=48, nt=128, x0=300.0, tau0=0.2)
    vint = np.full(sec.nt, 2000.0)
    a = phase_shift(sec.traces, sec.dt, 12.5, vint, 60.0)
    b = phase_shift_vti(sec.traces, sec.dt, 12.5, vint, np.zeros(sec.nt), 60.0)
    assert np.allclose(a, b, atol=1e-6 * np.abs(a).max())
    iso = create("phase_shift", velocity="2000", fmax=60.0)(sec)
    vti = create("phase_shift_vti", velocity="2000", eta_source="constant", eta=0.0, fmax=60.0)(sec)
    assert np.allclose(iso.traces, vti.traces, atol=1e-6 * np.abs(iso.traces).max())


def test_interval_eta_of_a_constant_medium():
    t = np.arange(200) * 0.004
    v = np.full(t.size, 2200.0)
    assert np.allclose(interval_eta(t, v, np.full(t.size, 0.12), v), 0.12, atol=1e-9)


def test_phase_shift_vti_right_eta_focuses_better():
    eta, x0, tau0 = 0.15, 600.0, 0.4
    sec = diffraction_section(eta=eta, x0=x0, tau0=tau0)
    x, times = sec.header("cdpx"), sec.times
    vint = np.full(sec.nt, 2000.0)
    right = phase_shift_vti(sec.traces, sec.dt, 12.5, vint, np.full(sec.nt, eta), 80.0)
    wrong = phase_shift_vti(sec.traces, sec.dt, 12.5, vint, np.zeros(sec.nt), 80.0)
    p_right, xr, tr = peak_near(right, x, times, x0, tau0)
    p_wrong, _, _ = peak_near(wrong, x, times, x0, tau0)
    assert abs(xr - x0) <= 12.5 and abs(tr - tau0) <= 2.5 * sec.dt
    assert p_right > 1.1 * p_wrong


def test_kirchhoff_pstm_eta_off_is_unchanged_and_eta_focuses():
    sec = diffraction_section(nx=96, nt=200, x0=600.0, tau0=0.4, eta=0.15)
    x, tau = sec.header("cdpx"), sec.times
    v = np.full((tau.size, x.size), 2000.0)
    a, _ = kirchhoff_time(sec.traces, 0.0, sec.dt, x, x, x, tau, v, 800.0)
    b, _ = kirchhoff_time(sec.traces, 0.0, sec.dt, x, x, x, tau, v, 800.0, eta_out=np.zeros_like(v))
    assert np.allclose(a, b, atol=1e-9 * np.abs(a).max())
    base = create("kirchhoff_pstm", velocity="2000", aperture=800.0)(sec)
    same = create("kirchhoff_pstm", velocity="2000", aperture=800.0, use_eta=False, eta=0.0)(sec)
    assert np.array_equal(base.traces, same.traces)
    vti = create("kirchhoff_pstm", velocity="2000", aperture=800.0, eta=0.15)(sec)
    p_iso, _, _ = peak_near(base.traces, base.header("cdpx"), base.times, 600.0, 0.4)
    p_vti, xv, tv = peak_near(vti.traces, vti.header("cdpx"), vti.times, 600.0, 0.4)
    assert abs(xv - 600.0) <= 12.5 and abs(tv - 0.4) <= 2.5 * sec.dt and p_vti > 1.05 * p_iso


# ---------------------------------------------------------------------------------------------- migration to zero offset
def common_offset_line(t0_of_m, v=2000.0, offsets=(100.0, 400.0, 700.0, 1000.0), ncmp=81, dm=12.5, nt=300,
                       dt=0.004, dip_cos=1.0):
    """Common-offset sections over a plane reflector, Levin (1971): t^2 = t0(m)^2 + (offset cos(dip) / v)^2."""
    m = np.arange(ncmp) * dm
    mm = np.tile(m, len(offsets))
    oo = np.repeat(np.asarray(offsets, dtype=float), ncmp)
    t = np.sqrt(t0_of_m(mm) ** 2 + (oo * dip_cos / v) ** 2)
    w, c = ricker(25.0, dt)
    tr = events_to_traces(t[:, None], np.ones((t.size, 1)), nt, dt, w, c)
    heads = {"cdp": np.tile(np.arange(1, ncmp + 1), len(offsets)).astype(float), "cdpx": mm, "sx": mm - oo / 2,
             "gx": mm + oo / 2, "offset": oo}
    return SeismicData(tr.astype(np.float32), dt, heads)


def test_mzo_flat_reflector_equals_nmo():
    d = common_offset_line(lambda m: np.full_like(m, 0.5))
    out = create("mzo_dmo", velocity="2000", use_eta=False, offset_class=300.0, aperture=600.0)(d, Context())
    assert out.ntr == 81 * 4 and np.allclose(np.unique(out.header("offset")), [150.0, 450.0, 750.0, 1050.0])
    times = out.times
    win = (times > 0.4) & (times < 0.6)
    nmo = nmo_correct(d.traces, d.times, d.header("offset"), np.full(d.traces.shape, 2000.0), 0.0, 0.0)
    errors, corr = [], []
    for i in range(28, 53):                                   # interior CMPs, 350-650 m
        for n in range(4):
            a = out.traces[i * 4 + n, win]
            b = nmo[n * 81 + i, win]
            errors.append(abs(times[win][np.argmax(np.abs(a))] - 0.5))
            corr.append(np.corrcoef(a, b)[0, 1])
    assert max(errors) <= 2.5 * d.dt and np.median(corr) > 0.8
    st = create("mzo_dmo", velocity="2000", use_eta=False, offset_class=300.0, aperture=600.0, output="stack")(d)
    assert st.ntr == 81 and abs(st.times[np.argmax(np.abs(st.traces[40]))] - 0.5) <= 2.5 * d.dt


def test_mzo_dipping_event_stacks_at_zero_offset_time():
    v, theta = 2000.0, np.deg2rad(20.0)

    def t0(m):
        return 0.5 + 2.0 * np.sin(theta) * (m - 500.0) / v

    d = common_offset_line(t0, v=v, dip_cos=np.cos(theta))
    # plain hyperbolic NMO leaves the far offset about 30 ms early at this dip
    nmo = nmo_correct(d.traces, d.times, d.header("offset"), np.full(d.traces.shape, v), 0.0, 0.0)
    far = 3 * 81 + 40
    win = np.abs(d.times - 0.5) < 0.06
    assert abs(d.times[win][np.argmax(np.abs(nmo[far, win]))] - 0.5) >= 5 * d.dt
    out = create("mzo_dmo", velocity="2000", use_eta=False, offset_class=300.0, aperture=700.0)(d, Context())
    cx, times = out.header("cdpx"), out.times
    err = []
    for i in np.flatnonzero((cx >= 350.0) & (cx <= 650.0)):
        expected = t0(cx[i])
        w = np.abs(times - expected) < 0.05
        err.append(abs(times[w][np.argmax(np.abs(out.traces[i, w]))] - expected))
    err = np.array(err)
    assert np.median(err) <= 1.5 * d.dt and err.max() <= 3.5 * d.dt


# ---------------------------------------------------------------------------------------------- residual moveout scan
def test_rmo_scan_finds_residual_curvature():
    nx, nt, dt, r_true = 24, 300, 0.004, 0.012
    off = np.linspace(100.0, 2000.0, nx)
    t = 0.6 + r_true * (off / off.max()) ** 2
    w, c = ricker(25.0, dt)
    tr = events_to_traces(t[:, None], np.ones((nx, 1)), nt, dt, w, c)
    d = SeismicData(tr.astype(np.float32), dt, {"cdp": 1.0, "offset": off})
    ctx = Context()
    out = create("rmo_scan", cdps="1", mode="parabolic", rmin=-0.03, rmax=0.03, nr=61)(d, ctx)
    assert out.ntr == 61 + 2 and out.has_header("rmo_r")
    best = ctx.results["rmo_scan"]["best"]["1"]
    assert abs(best[int(round(0.6 / dt))] - r_true) <= 0.002


def test_rmo_scan_eta_after_hyperbolic_nmo():
    nx, nt, dt, v, eta, t0 = 30, 300, 0.004, 2000.0, 0.1, 0.8
    off = np.linspace(50.0, 2000.0, nx)
    w, c = ricker(25.0, dt)
    raw = events_to_traces(moveout_time(t0, off, v, eta)[:, None], np.ones((nx, 1)), nt, dt, w, c)
    times = np.arange(nt) * dt
    flat = nmo_correct(raw.astype(np.float32), times, off, np.full((nx, nt), v), 0.0, 0.0)
    d = SeismicData(flat, dt, {"cdp": 1.0, "offset": off})
    ctx = Context()
    ctx.velocity = VelocityPicks.constant(v)
    create("rmo_scan", cdps="1", mode="eta (VTI)", eta_min=0.0, eta_max=0.25, nr=26, velocity="project")(d, ctx)
    best = ctx.results["rmo_scan"]["best"]["1"]
    assert abs(best[int(round(t0 / dt))] - eta) <= 0.02


# ---------------------------------------------------------------------------------------------- 3D migration
def diffraction_cube(n=16, nt=64, dt=0.004, d=25.0, v=3000.0, tau0=0.12, f0=30.0):
    il, xl = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    il, xl = il.ravel().astype(float), xl.ravel().astype(float)
    x, y = (xl - 1) * d, (il - 1) * d
    xd = yd = (n // 2 - 1) * d
    t = np.sqrt(tau0 ** 2 + 4.0 * ((x - xd) ** 2 + (y - yd) ** 2) / v ** 2)
    w, c = ricker(f0, dt)
    tr = events_to_traces(t[:, None], np.ones((t.size, 1)), nt, dt, w, c)
    heads = {"iline": il, "xline": xl, "cdp": (il - 1) * n + xl, "cdpx": x, "cdpy": y}
    cube = SeismicData(tr.astype(np.float32), dt, heads, name="cube")
    cube.meta["geometry"] = {"type": "3D", "inline_bin": d, "xline_bin": d}
    return cube, xd, yd, tau0


def concentration(d, xd, yd, tau0, size=25.0, nsamp=3):
    e = d.traces.astype(np.float64) ** 2
    near = (np.abs(d.header("cdpx") - xd) <= size + 1e-6) & (np.abs(d.header("cdpy") - yd) <= size + 1e-6)
    it = int(round(tau0 / d.dt))
    return e[near, max(0, it - nsamp): it + nsamp + 1].sum() / e.sum()


def test_3d_migrations_focus_a_point_diffractor():
    cube, xd, yd, tau0 = diffraction_cube()
    ctx = Context()
    ctx.velocity = VelocityPicks.constant(3000.0)
    before = concentration(cube, xd, yd, tau0)
    for name, kw in (("phase_shift_3d", {"velocity": "project"}), ("stolt_3d", {"velocity": 3000.0}),
                     ("kirchhoff_pstm_3d", {"velocity": "project", "aperture": 400.0})):
        out = create(name, **kw)(cube, ctx)
        assert out.ntr == cube.ntr and out.has_header("iline") and out.has_header("cdpy")
        a = np.abs(out.traces)
        i, it = np.unravel_index(np.argmax(a), a.shape)
        assert abs(out.header("cdpx")[i] - xd) <= 25.0 and abs(out.header("cdpy")[i] - yd) <= 25.0, name
        assert abs(out.times[it] - tau0) <= 2.5 * cube.dt, name
        assert concentration(out, xd, yd, tau0) > 2.0 * before, name


# ---------------------------------------------------------------------------------------------- PSDM from topography
def test_kirchhoff_psdm_from_topography_flat_reflector_depth():
    v, er, dt, nt = 2000.0, -200.0, 0.004, 200

    def elev(x):
        return 100.0 + 0.1 * x

    shots, recs = np.arange(0.0, 1001.0, 100.0), np.arange(0.0, 1001.0, 20.0)
    sx, gx = np.repeat(shots, recs.size), np.tile(recs, shots.size)
    t = np.hypot(gx - sx, (elev(sx) - er) + (elev(gx) - er)) / v        # image source below the flat reflector
    w, c = ricker(25.0, dt)
    tr = events_to_traces(t[:, None], np.ones((t.size, 1)), nt, dt, w, c)
    cdpx = np.round(0.5 * (sx + gx) / 10.0) * 10.0
    heads = {"fldr": np.repeat(np.arange(1, shots.size + 1), recs.size), "sx": sx, "gx": gx, "offset": gx - sx,
             "selev": elev(sx), "gelev": elev(gx), "cdp": cdpx / 10.0 + 1.0, "cdpx": cdpx}
    d = SeismicData(tr.astype(np.float32), dt, heads)
    out = create("kirchhoff_psdm", velocity="2000", dz=10.0, zmax=700.0, grid_dx=20.0, aperture=800.0,
                 surface="topography")(d, Context())
    z_ref = out.meta["depth_reference_elevation"]
    assert abs(z_ref - 200.0) < 1e-6
    x, z = out.header("cdpx"), out.times
    inner = (x >= 300.0) & (x <= 700.0)
    depth = z[np.argmax(np.abs(out.traces[inner]), axis=1)]
    assert np.median(np.abs(depth - (z_ref - er))) <= 20.0
    assert np.all(out.traces[inner][:, z < 20.0] == 0.0)              # nothing above the ground (30-70 m down)

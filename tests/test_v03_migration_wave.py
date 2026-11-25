"""Tests of the version 0.3 wave-equation depth migrations: finite difference (45/65 degrees), Fourier finite
difference, PSPI, Gaussian beams (post-stack) and the prestack shot-profile migration."""

import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing import REGISTRY, Context, create
from seisdesk.processing import migration_wave
from seisdesk.processing.migration_depth import split_step

NEW = ("fd_migration", "ffd_migration", "pspi_migration", "gaussian_beam", "wave_psdm")
V, DX, DT, NT, DZ, NX = 2000.0, 10.0, 0.004, 200, 5.0, 64
POSTSTACK = [("fd_migration", {"angle": "45"}), ("fd_migration", {"angle": "65"}), ("ffd_migration", {}),
             ("pspi_migration", {}), ("gaussian_beam", {})]


def zero_offset(times, f0=25.0):
    """A stacked (zero-offset) section with one event at the given two-way times, one per trace."""
    x = np.arange(NX) * DX
    w, c = ricker(f0, DT)
    tr = events_to_traces(np.asarray(times, dtype=float)[:, None], np.ones((NX, 1)), NT, DT, w, c)
    heads = {"cdp": np.arange(1, NX + 1), "cdpx": x, "sx": x, "gx": x, "offset": np.zeros(NX)}
    d = SeismicData(tr.astype(np.float32), DT, heads, name="zero offset")
    d.meta["geometry"] = {"bin_size": DX}
    return d


def diffractor(xd=320.0, zd=300.0):
    x = np.arange(NX) * DX
    return zero_offset(2.0 * np.hypot(x - xd, zd) / V)


def peak(img):
    """(trace, depth sample) of the largest absolute amplitude of a depth section."""
    a = np.abs(img.traces)
    ix, iz = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(ix), int(iz)


def test_new_steps_registered():
    for name in NEW:
        assert name in REGISTRY and REGISTRY[name].category == "Depth migration"


@pytest.mark.parametrize("name,extra", POSTSTACK)
def test_poststack_methods_focus_a_point_diffractor(name, extra):
    out = create(name, velocity=str(V), dz=DZ, zmax=500.0, fmax=50.0, **extra)(diffractor(), Context())
    assert out.domain == "depth" and out.ntr == NX and out.nt == 101
    assert np.all(np.isfinite(out.traces))
    ix, iz = peak(out)
    assert abs(ix - 32) <= 2 and abs(iz - 60) <= 2, (name, ix, iz)


@pytest.mark.parametrize("name,extra", POSTSTACK)
def test_poststack_methods_place_a_dipping_reflector(name, extra):
    dip = np.deg2rad(20.0)
    x = np.arange(NX) * DX
    d = zero_offset(2.0 * (150.0 + x * np.tan(dip)) * np.cos(dip) / V)
    out = create(name, velocity=str(V), dz=DZ, zmax=500.0, fmax=50.0, **extra)(d, Context())
    z_true = 150.0 + 320.0 * np.tan(dip)
    z_img = int(np.argmax(np.abs(out.traces[32]))) * DZ
    assert abs(z_img - z_true) <= 2 * DZ, (name, z_img, z_true)


def test_constant_velocity_extrapolators_agree_with_split_step():
    d = diffractor()
    vel = np.full((101, NX), V)
    ref = split_step(d.traces, DT, DX, vel, DZ, 50.0)
    for method, limit in (("FFD", 0.95), ("PSPI", 0.95), ("FD", 0.6)):
        img = migration_wave.poststack_migration(d.traces, DT, DX, vel, DZ, 50.0, method)
        assert img.shape == ref.shape and np.all(np.isfinite(img))
        assert np.corrcoef(img.ravel(), ref.ravel())[0, 1] > limit, method


def test_lateral_velocity_change_runs_every_extrapolator():
    d = diffractor()
    vel = np.full((101, NX), 1800.0)
    vel[:, NX // 2:] = 2400.0
    ref = split_step(d.traces, DT, DX, vel, DZ, 50.0)
    for method in ("FD", "FFD", "PSPI"):
        img = migration_wave.poststack_migration(d.traces, DT, DX, vel, DZ, 50.0, method, max_refs=4)
        assert np.all(np.isfinite(img))
        # across a sharp jump every method differs from split-step (itself approximate there): a loose agreement only;
        # the constant-velocity tests check the accuracy
        assert np.corrcoef(img.ravel(), ref.ravel())[0, 1] > 0.4, method


def test_fd_step_is_unitary_with_the_expected_phase():
    nxp, dx, dz, u = 512, 10.0, 10.0, 1000.0
    x = np.arange(nxp) * dx
    kx = 2 * np.pi / 200.0
    p = (np.exp(1j * kx * x) * np.exp(-0.5 * ((x - x.mean()) / 600.0) ** 2))[:, None]
    w = np.array([2 * np.pi * 20.0])
    out = migration_wave._fd_step(p, w, np.full(nxp, u), dz, dx, 0.5, 0.25)
    assert abs(np.sum(np.abs(out) ** 2) / np.sum(np.abs(p) ** 2) - 1.0) < 1e-8
    big_x2 = (u * kx / w[0]) ** 2
    expected = -(w[0] / u) * 0.5 * big_x2 / (1.0 - 0.25 * big_x2) * dz          # (kz - w/u) dz, 45 degrees
    phase = np.angle(out[nxp // 2, 0] / p[nxp // 2, 0])
    assert phase < 0 and abs(phase - expected) < 0.2 * abs(expected)


def shots(zr=400.0, f0=20.0):
    """Three split-spread shots over a flat reflector at depth zr in constant velocity V."""
    gx = np.arange(0.0, 1201.0, 10.0)
    sx = np.repeat([300.0, 600.0, 900.0], gx.size)
    gxx = np.tile(gx, 3)
    off = gxx - sx
    w, c = ricker(f0, DT)
    tr = events_to_traces((np.hypot(off, 2 * zr) / V)[:, None], np.ones((off.size, 1)), 300, DT, w, c)
    mid = 0.5 * (sx + gxx)
    heads = {"fldr": np.repeat([1.0, 2.0, 3.0], gx.size), "sx": sx, "gx": gxx, "offset": off,
             "cdp": np.rint(mid / 5.0) + 1, "cdpx": mid}
    return SeismicData(tr.astype(np.float32), DT, heads, name="three shots")


@pytest.mark.parametrize("extrapolator", ["split-step", "PSPI", "FFD", "finite difference"])
def test_prestack_shot_profile_images_a_flat_reflector(extrapolator):
    out = create("wave_psdm", velocity=str(V), extrapolator=extrapolator, dz=10.0, zmax=600.0, fmin=5.0,
                 fmax=40.0, aperture=300.0)(shots(), Context())
    assert out.domain == "depth" and out.ntr == 121 and np.all(np.isfinite(out.traces))
    assert np.allclose(np.diff(out.header("cdpx")), 10.0)
    depths = np.argmax(np.abs(out.traces[30:91]), axis=1) * 10.0
    assert abs(np.median(depths) - 400.0) <= 20.0, extrapolator
    assert abs(int(np.argmax(np.abs(out.traces[60]))) * 10.0 - 400.0) <= 20.0


def test_prestack_needs_coordinates():
    d = shots()
    d = d.copy(headers={k: (np.zeros(d.ntr) if k in ("sx", "gx") else v.copy()) for k, v in d.headers.items()})
    with pytest.raises(ValueError):
        create("wave_psdm", velocity=str(V), zmax=100.0)(d, Context())


def test_poststack_rejects_prestack_input():
    d = shots()
    with pytest.raises(ValueError):
        create("fd_migration", velocity=str(V), zmax=100.0)(d, Context())

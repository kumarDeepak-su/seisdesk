"""Bounded layered isotropic surface-wave inversion using disba's CPS solvers.

SI units at the public boundary: thickness m, Vp/Vs m/s, density kg/m³. Last layer
is a half-space with zero thickness. Fits identified Rayleigh/Love phase/group
modes with known standard errors; Vp/Vs and density remain fixed. Uncertainty is
conditional on this layer topology, mode assignment and those fixed properties.
"""
from __future__ import annotations

import importlib.metadata
import numpy as np
from scipy.optimize import least_squares


def validate_model(model):
    m = np.asarray(model, dtype=float)
    if m.ndim != 2 or m.shape[1] != 4 or not 1 <= len(m) <= 40 or not np.isfinite(m).all():
        raise ValueError("model needs 1–40 rows: thickness_m, vp_ms, vs_ms, density_kgm3")
    if np.any(m[:-1,0] <= 0) or m[-1,0] != 0 or np.any(m[:,2:] <= 0) or np.any(m[:,1] <= np.sqrt(4/3)*m[:,2]):
        raise ValueError("use positive finite layers over a zero-thickness half-space; Vs/rho positive and Vp > sqrt(4/3) Vs")
    return m.copy()


def validate_picks(picks):
    rows = []
    for pick in picks:
        row = dict(frequency_hz=float(pick["frequency_hz"]), velocity_ms=float(pick["velocity_ms"]),
                   sigma_ms=float(pick.get("sigma_ms", 10)), mode=int(pick.get("mode", 0)),
                   wave=str(pick.get("wave", "rayleigh")), kind=str(pick.get("kind", "phase")))
        if (not np.isfinite([row[k] for k in ("frequency_hz", "velocity_ms", "sigma_ms")]).all() or
                min(row["frequency_hz"], row["velocity_ms"], row["sigma_ms"]) <= 0 or not 0 <= row["mode"] <= 8 or
                float(pick.get("mode", 0)) != row["mode"] or row["wave"] not in ("rayleigh", "love") or
                row["kind"] not in ("phase", "group")):
            raise ValueError("invalid dispersion pick: positive frequency/velocity/error, integer mode 0–8, rayleigh/love, phase/group")
        rows.append(row)
    if not rows:
        raise ValueError("no dispersion picks")
    return rows


def predict_dispersion(model, picks, algorithm="dunkin", allow_missing=False):
    import os
    from pathlib import Path
    # Installed program directories need not be writable. Numba's user cache is.
    cache=Path(os.environ.get("LOCALAPPDATA",Path.home()/".cache"))/"SeisDesk"/"numba"
    os.environ.setdefault("NUMBA_CACHE_DIR",str(cache))
    from disba import PhaseDispersion, GroupDispersion
    m = validate_model(model); picks = validate_picks(picks)
    if algorithm != "dunkin":
        raise ValueError("this integration validates the Dunkin solver only")
    # disba expects km, km/s and g/cm³. Accurate root tolerance also matters for derivatives.
    params = (m[:,0]/1000, m[:,1]/1000, m[:,2]/1000, m[:,3]/1000)
    predicted = np.full(len(picks), np.nan)
    for wave, kind, mode in sorted({(p["wave"],p["kind"],p["mode"]) for p in picks}):
        ids = [i for i,p in enumerate(picks) if (p["wave"],p["kind"],p["mode"]) == (wave,kind,mode)]
        periods, inv = np.unique([1/picks[i]["frequency_hz"] for i in ids], return_inverse=True)
        solver = (PhaseDispersion if kind == "phase" else GroupDispersion)(
            *params, algorithm=algorithm, dc=.00001, **({"dt":.005} if kind == "group" else {}))
        from disba import DispersionError
        try:
            curve = solver(periods, wave=wave, mode=mode)
        except DispersionError:
            if not allow_missing:
                raise ValueError(f"no {wave} {kind} mode {mode} for the requested model/frequencies") from None
            continue
        values = np.full(len(periods), np.nan)
        for period, velocity in zip(curve.period, curve.velocity):
            match = np.flatnonzero(np.isclose(periods, period, rtol=1e-10, atol=0))
            if len(match) == 1:
                values[match[0]] = velocity*1000
        predicted[ids] = values[inv]
    if not allow_missing and not np.isfinite(predicted).all():
        raise ValueError("some assigned dispersion modes do not exist at these frequencies; review mode labels and layer bounds")
    return predicted


def invert_dispersion(model, picks, vs_bounds, thickness_bounds=None, *, max_evaluations=150,
                       starts=3, bootstrap=0, seed=42, damping=.05, smoothness=0.,
                       algorithm="dunkin", cancelled=lambda:False, progress=lambda _:None):
    """Multistart bounded log-parameter least squares, optionally with noise refits.

    Bootstrap draws Gaussian pick errors and refits from the optimum. These are
    conditional timing/velocity-error intervals, not a posterior over layer count,
    mode ambiguities, density or Vp/Vs. Cancellation is checked every evaluation.
    """
    m = validate_model(model); picks = validate_picks(picks); n = len(m)
    if not 1 <= starts <= 30 or not 0 <= bootstrap <= 500 or not 10 <= max_evaluations <= 2000:
        raise ValueError("starts 1–30, bootstrap 0–500, maximum evaluations 10–2000 required")
    if not np.isfinite([damping,smoothness]).all() or min(damping,smoothness) < 0:
        raise ValueError("regularisation weights must be finite and nonnegative")
    bounds = np.asarray(vs_bounds, dtype=float)
    if bounds.shape != (n,2):
        raise ValueError("supply lower/upper Vs bounds for every layer")
    variable_h = thickness_bounds is not None
    if variable_h:
        hb = np.asarray(thickness_bounds, dtype=float)
        if hb.shape != (n-1,2):
            raise ValueError("supply thickness bounds for each finite layer")
        bounds = np.vstack((bounds, hb))
    initial = np.r_[m[:,2], m[:-1,0]] if variable_h else m[:,2].copy()
    if (not np.isfinite(bounds).all() or np.any(bounds[:,0] <= 0) or np.any(bounds[:,1] <= bounds[:,0]) or
            np.any(initial < bounds[:,0]) or np.any(initial > bounds[:,1])):
        raise ValueError("positive ordered bounds must enclose the initial model")
    if len(picks) <= len(initial):
        raise ValueError("more picks than free model parameters are required")
    ratio = m[:,1]/m[:,2]
    x0 = np.log(initial); lower,upper = np.log(bounds).T
    observed = np.array([p["velocity_ms"] for p in picks]); sigma = np.array([p["sigma_ms"] for p in picks])
    initial_prediction = predict_dispersion(m,picks,algorithm)
    rng = np.random.default_rng(seed)
    def unpack(x):
        model = m.copy(); model[:,2] = np.exp(x[:n]); model[:,1] = ratio*model[:,2]
        if variable_h:
            model[:-1,0] = np.exp(x[n:])
        return model
    evaluations = 0
    def solve(start, target):
        def residual(x):
            nonlocal evaluations
            if cancelled():
                raise InterruptedError("surface-wave inversion cancelled")
            evaluations += 1
            predicted = predict_dispersion(unpack(x),picks,algorithm,allow_missing=True)
            misfit = np.where(np.isfinite(predicted),(predicted-target)/sigma,1e6)
            return np.r_[misfit, np.sqrt(damping)*(x-x0), np.sqrt(smoothness)*np.diff(x[:n],n=2)]
        return least_squares(residual, start, bounds=(lower,upper), max_nfev=max_evaluations,
                             diff_step=1e-3, ftol=1e-7, xtol=1e-6, gtol=1e-5)
    candidates=[]
    for i in range(starts):
        guess=x0 if i == 0 else rng.uniform(lower,upper)
        fit=solve(guess,observed)
        pred=predict_dispersion(unpack(fit.x),picks,algorithm,allow_missing=True)
        if np.isfinite(pred).all():
            candidates.append((fit,pred))
        progress((i+1)/(starts+bootstrap))
    if not candidates:
        raise ValueError("no fitted model predicts all assigned modes; inspect the picks and bounds")
    fit,prediction=min(candidates,key=lambda pair:float(np.sum(pair[0].fun**2)))
    best=unpack(fit.x)
    ensembles=[]; failed=0
    for i in range(bootstrap):
        res=solve(fit.x,observed+rng.normal(0,sigma))
        pred=predict_dispersion(unpack(res.x),picks,algorithm,allow_missing=True)
        if res.success and np.isfinite(pred).all():
            ensembles.append(unpack(res.x))
        else:
            failed += 1
        progress((starts+i+1)/(starts+bootstrap))
    result=dict(model=best, initial_model=m, picks=picks, predicted_ms=prediction,
                initial_predicted_ms=initial_prediction, residual_ms=prediction-observed,
                normalized_residual=(prediction-observed)/sigma,
                rms_ms=float(np.sqrt(np.mean((prediction-observed)**2))),
                initial_rms_ms=float(np.sqrt(np.mean((initial_prediction-observed)**2))),
                objective=float(np.sum(fit.fun**2)), success=bool(fit.success), message=str(fit.message),
                evaluations=evaluations, starts=starts, valid_starts=len(candidates), vs_bounds=np.asarray(vs_bounds),
                thickness_bounds=np.asarray(thickness_bounds) if variable_h else None,
                bootstrap_requested=bootstrap, bootstrap_failed=failed, bootstrap_success=len(ensembles),
                seed=seed, algorithm=algorithm, engine="disba "+importlib.metadata.version("disba"),
                assumptions="1D horizontal isotropic solid layers; fixed layer count, mode labels, Vp/Vs and density; traction-free surface",
                uncertainty="Gaussian pick-error refits conditional on the stated assumptions; not structural or mode uncertainty")
    if ensembles:
        result.update(ensemble=np.asarray(ensembles), model_p05=np.percentile(ensembles,5,axis=0),
                       model_p95=np.percentile(ensembles,95,axis=0), model_std=np.std(ensembles,axis=0))
    return result

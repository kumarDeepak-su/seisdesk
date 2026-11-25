"""Ready-made processing sequences offered in the flow editor (every step can be edited, switched off or removed)."""

from __future__ import annotations

from .flow import Flow, FlowStep


def _flow(name, steps) -> Flow:
    out = []
    for item in steps:
        process, params = item[0], item[1]
        enabled = item[2] if len(item) > 2 else True
        note = item[3] if len(item) > 3 else ""
        out.append(FlowStep(process, dict(params), enabled, note))
    return Flow(out, name)


def templates() -> dict:
    return {
        "Refraction v0.4: first-arrival tomography": _flow("first-arrival tomography", [
            ("auto_first_breaks", {"pickset": "first breaks"}, False, "optional: review picks in F7 before inversion"),
            ("refraction_tomography", {"pickset": "first breaks"}, False,
             "set geometry, pick phases, grid and starting model; enable after review"),
        ]),
        "Marine v0.4: SRME prediction": _flow("SRME", [
            ("srme", {"output": "multiple model"}, False,
             "requires regular co-located 2D sources/receivers and t0=0; export and inspect the model"),
        ]),
        "Marine v0.4: adaptive subtraction of exported model": _flow("adaptive subtraction", [
            ("adaptive_subtraction", {}, False,
             "select original data as input and the exported SRME model as model file"),
        ]),
        "Converted waves v0.4: CCP, PS NMO, stack": _flow("PS processing", [
            ("ps_binning", {"gamma": 2.0}, True, "use radial PS data with correct polarity; replace assumed Vp/Vs"),
            ("sort", {"keys": ["cdp", "offset"]}),
            ("ps_nmo", {}, False, "set PS moveout velocity and Vp/Vs before enabling"),
            ("stack", {}, False, "enable after PS NMO is checked"),
        ]),
        "Inversion v0.4: acoustic FWI": _flow("acoustic FWI", [
            ("fwi", {}, False, "prepare flat-datum shots and starting model; start with low frequencies"),
        ]),
        "Depth imaging v0.4: acoustic RTM": _flow("acoustic RTM", [
            ("rtm", {}, False, "run on the ORIGINAL reflection shot data using the updated model"),
        ]),
        "Depth imaging v0.4: MVA on migrated image gathers": _flow("MVA update", [
            ("mva_tomography", {}, False, "scan coherent horizons on depth gathers; small velocity changes"),
        ]),
        "Marine 2D: shots to migrated stack": _flow("marine 2D", [
            ("streamer_geometry", {}, False, "only for files without coordinates"),
            ("geometry_2d", {"bin_size": 12.5}),
            ("velocity_mute", {"velocity": 1500.0, "t0": 0.06, "kind": "top"}, False, "direct-wave mute"),
            ("bandpass", {"corners": [4, 8, 60, 80]}),
            ("tpow", {"power": 2.0}),
            ("decon", {"operator_length": 0.16, "gap": 0.004, "prewhitening": 0.1, "tmin": 0.1, "tmax": 3.0}),
            ("sort", {"keys": ["cdp", "offset"]}),
            ("auto_velocity", {"cdp_step": 40, "supergather": 2}, True, "refine in Velocity analysis"),
            ("nmo", {"velocity": "project", "stretch_limit": 0.5}),
            ("radon_demultiple", {"q_cut": 0.05}, False, "on NMO-corrected CMP gathers"),
            ("stack", {}),
            ("phase_shift", {"velocity": "project"}),
        ]),
        "Land 2D: statics, velocities, stack": _flow("land 2D", [
            ("geometry_2d", {"bin_size": 5.0}),
            ("elevation_statics", {"datum": 0.0, "replacement_velocity": 2000.0}, False, "needs elevations"),
            ("velocity_mute", {"velocity": 1800.0, "t0": 0.03, "kind": "top"}, True, "first-arrival mute"),
            ("bandpass", {"corners": [8, 12, 70, 90]}),
            ("agc", {"window": 0.5}),
            ("decon", {"operator_length": 0.12, "gap": 0.002}),
            ("sort", {"keys": ["cdp", "offset"]}),
            ("auto_velocity", {"cdp_step": 20, "supergather": 3}),
            ("nmo", {"velocity": "project"}),
            ("residual_statics", {"tmin": 0.2, "tmax": 1.2}),
            ("stack", {}),
        ]),
        "Land 2D: complete sequence (QC to migrated stack)": _flow("land 2D complete", [
            ("auto_edit", {"low": 0.05, "high": 4.0, "action": "zero"}, True, "dead and noisy traces"),
            ("despike", {"threshold": 3.0}),
            ("velocity_mute", {"velocity": 1800.0, "t0": 0.03, "kind": "top"}, True, "first arrivals"),
            ("cone_mute", {"v_fast": 900.0, "v_slow": 250.0}, False, "ground roll: read the cone edges on a shot"),
            ("taup_filter", {"v_cut": 1000.0}, False, "alternative to the cone mute"),
            ("exp_gain", {"db_per_s": 6.0}),
            ("sc_amplitude", {"tmin": 0.1}),
            ("sc_decon", {"tmin": 0.1, "tmax": 1.0, "prewhitening": 1.0}),
            ("bandpass", {"corners": [8, 12, 70, 90]}),
            ("refraction_statics", {"pickset": "first breaks", "phase": "refractor 1"}, False,
             "needs first-break phases picked in the picking tool"),
            ("sort", {"keys": ["cdp", "offset"]}),
            ("auto_velocity", {"cdp_step": 20, "supergather": 3}),
            ("smooth_velocity", {"cdp_step": 10, "smooth_cdp": 10.0}),
            ("nmo", {"velocity": "project"}),
            ("residual_statics", {"tmin": 0.2, "tmax": 1.2}),
            ("stack", {"normalisation": "diversity"}),
            ("fx_decon", {"gather_key": ""}, True, "random noise on the stack"),
            ("phase_shift", {"velocity": "project"}),
        ]),
        "Refraction: first breaks and layer depths": _flow("refraction", [
            ("auto_first_breaks", {"pickset": "first breaks", "gather_key": "fldr"}),
            ("refraction_analysis", {"pickset": "first breaks"}, True,
             "split the picks into direct and refracted phases in the picking tool first"),
        ]),
        "DMO: NMO, DMO, inverse NMO, NMO, stack": _flow("DMO", [
            ("sort", {"keys": ["cdp", "offset"]}),
            ("nmo", {"velocity": "project"}),
            ("dmo", {"offset_bin": 0.0}),
            ("nmo", {"velocity": "project", "inverse": True}, True, "back to prestack times for a new velocity pass"),
            ("nmo", {"velocity": "project"}),
            ("stack", {}),
        ]),
        "Residual moveout and anisotropic stack": _flow("RMO VTI", [
            ("sort", {"keys": ["cdp", "offset"]}),
            ("nmo", {"velocity": "project", "use_eta": True}),
            ("residual_moveout", {"order": "4"}),
            ("stack", {}),
        ]),
        "Prestack time migration (Kirchhoff)": _flow("PSTM", [
            ("geometry_2d", {"bin_size": 12.5}, False),
            ("kirchhoff_pstm", {"velocity": "project", "aperture": 2000.0}),
        ]),
        "Post-stack depth migration": _flow("depth migration", [
            ("build_depth_model", {"source": "picks (Dix)", "dz": 10.0, "zmax": 4000.0}),
            ("split_step", {"velocity": "project model", "dz": 10.0, "zmax": 4000.0}),
        ]),
        "Prestack depth migration (Kirchhoff)": _flow("PSDM", [
            ("build_depth_model", {"source": "picks (Dix)", "dz": 10.0, "zmax": 3000.0}),
            ("kirchhoff_psdm", {"velocity": "project model", "dz": 10.0, "zmax": 3000.0}),
        ]),
        "Time to depth conversion": _flow("depth conversion", [
            ("time_to_depth", {"using": "velocity picks", "velocity": "project", "dz": 5.0}),
        ]),
    }

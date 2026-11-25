"""Saved-flow entry point for modal dispersion inversion."""
import csv
import os
import numpy as np
from .base import Process, Param, register


@register
class SurfaceWaveInversion(Process):
    name="surface_wave_inversion"
    label="Layered surface-wave inversion (disba/CPS)"
    category="Inversion"
    description=("Fit identified Rayleigh/Love phase/group modes in a 1D layered isotropic model. "
                 "Vp/Vs, density and layer count fixed; bounded Vs/optional thickness updates and conditional pick-error refits.")
    params=(Param("model_file","infile","","Layer model CSV",
                  help="Columns: thickness_m,vp_ms,vs_ms,density_kgm3,vs_min,vs_max,h_min,h_max. Last thickness=0 for half-space."),
            Param("picks_file","infile","","Dispersion picks CSV",
                  help="Columns: frequency_hz,velocity_ms,sigma_ms,mode,wave,kind. Mode 0=fundamental; rayleigh/love; phase/group."),
            Param("fit_thickness","bool",False,"Fit finite-layer thicknesses"),
            Param("starts","int",3,"Starting models",minimum=1,maximum=30),
            Param("bootstrap","int",0,"Gaussian pick-error refits",minimum=0,maximum=500),
            Param("seed","int",42,"Random seed",minimum=0),
            Param("max_evaluations","int",150,"Maximum evaluations per fit",minimum=10,maximum=2000),
            Param("damping","float",.05,"Prior weight",minimum=0),
            Param("algorithm","choice","dunkin","Rayleigh solver",choices=("dunkin",)))

    def run(self,data,context):
        from ..core.surface_inversion import invert_dispersion
        with open(os.path.join(context.work_dir,self["model_file"]),encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        keys=("thickness_m","vp_ms","vs_ms","density_kgm3","vs_min","vs_max","h_min","h_max")
        model=np.array([[float(row.get(k,0)) for k in keys] for row in rows])
        if model.ndim != 2 or model.shape[1] != 8:raise ValueError("invalid or empty layered model CSV")
        with open(os.path.join(context.work_dir,self["picks_file"]),encoding="utf-8-sig",newline="") as f:
            picks=list(csv.DictReader(f))
        result=invert_dispersion(model[:,:4],picks,model[:,4:6],model[:-1,6:8] if self["fit_thickness"] else None,
                                  **{k:self[k] for k in ("starts","bootstrap","seed","max_evaluations","damping","algorithm")},
                                  cancelled=context.cancelled,progress=context.report)
        context.results["surface_wave:"+data.name]=result
        context.log(f"Surface-wave RMS {result['initial_rms_ms']:.3f} to {result['rms_ms']:.3f} m/s; {result['message']}")
        out=data.copy(); out.add_history("layered dispersion inversion; model and QC stored in project results")
        return out

"""Small layered dispersion recovery example shared by validation and GUI previews."""
import numpy as np
from seisdesk.core.surface_inversion import predict_dispersion,invert_dispersion


def example(bootstrap=20):
    truth=np.array([[10,400,200,1800],[0,800,400,2000]],float)
    picks=[dict(frequency_hz=float(f),velocity_ms=250.,sigma_ms=2.,mode=0,wave=wave,kind="phase")
           for wave in ("rayleigh","love") for f in np.linspace(8,50,16)]
    exact=predict_dispersion(truth,picks)
    noise=np.random.default_rng(17).normal(0,2,len(picks))
    for p,v in zip(picks,exact+noise):p["velocity_ms"]=float(v)
    initial=truth.copy(); initial[:,1:3]*=1.15; initial[0,0]=13
    result=invert_dispersion(initial,picks,[[100,350],[250,650]],[[5,20]],starts=3,bootstrap=bootstrap,
                              max_evaluations=120,damping=.01)
    result['synthetic_truth']=truth
    return result

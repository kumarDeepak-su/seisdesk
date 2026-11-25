import json
from pathlib import Path
import sys
import numpy as np
from v10_example import example


def main(destination):
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    result=example(bootstrap=30)
    assert result["rms_ms"] < 3
    np.testing.assert_allclose(result["model"][:,2],result["synthetic_truth"][:,2],rtol=.04)
    assert abs(result["model"][0,0]-10)<.6
    assert result["bootstrap_success"] >= 25
    report={k:v for k,v in result.items() if not isinstance(v,np.ndarray) and k != "picks"}
    for k in ("model","synthetic_truth","model_p05","model_p95"):report[k]=result[k].tolist()
    (root/"surface-report.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for wave,color in (("rayleigh","tab:blue"),("love","tab:orange")):
        ids=[i for i,p in enumerate(result["picks"]) if p["wave"]==wave]
        f=[result["picks"][i]["frequency_hz"] for i in ids]
        v=[result["picks"][i]["velocity_ms"] for i in ids]
        ax.errorbar(f,v,yerr=2,fmt="o",ms=3,color=color,label=wave+" observations ±σ")
        ax.plot(f,result["predicted_ms"][ids],color=color)
    ax.set(xlabel="Frequency (Hz)",ylabel="Phase velocity (m/s)",title="Synthetic modal dispersion fit"); ax.legend(fontsize=8)
    def step(model):
        h=model[0,0]; return [model[0,2],model[0,2],model[1,2],model[1,2]],[0,h,h,25]
    for model in result["ensemble"]:bx.plot(*step(model),color="tab:green",alpha=.15)
    for key,label,color in (("initial_model","Initial","gray"),("synthetic_truth","Truth","k"),("model","Fit","tab:blue")):
        bx.plot(*step(result[key]),label=label,color=color,lw=2)
    bx.set(xlabel="Vs (m/s)",ylabel="Depth (m)",title="Layer recovery and 30 pick-error refits"); bx.invert_yaxis(); bx.legend()
    fig.savefig(root/"v10_layered_surface_waves.png",dpi=150); plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":main(sys.argv[1])

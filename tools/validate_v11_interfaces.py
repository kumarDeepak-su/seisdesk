"""Independent plane-reflection checks and persisted 3D boundary diagnostics."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v11_example import example
from seisdesk.core.tomography3d import NodeGrid3D, arrivals3d
from seisdesk.core.interfaces3d import save_interfaces3d
from seisdesk.workflow.results import save_results, load_results


def render(result, truth, report, root):
    fig, axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row, name in enumerate(truth):
        limits = (min(float(truth[name].min()),float(result["sampled_interfaces"][name].min())),
                  max(float(truth[name].max()),float(result["sampled_interfaces"][name].max())))
        if limits[0] == limits[1]:limits=(limits[0]-1,limits[1]+1)
        for col, (ax, values, title) in enumerate(zip(axes[row],
                (truth[name],result["sampled_interfaces"][name],result["interface_std"][name]),
                ("Analytical plane depth", "Fitted depth; crosses = contacts", "Conditional depth standard deviation"))):
            lo,hi=limits if col<2 else (0,max(1.,float(np.nanmax(values))))
            image=ax.imshow(values,origin="lower",extent=(-5,85,-5,85),cmap="viridis",vmin=lo,vmax=hi)
            if col==1:
                iy,ix=np.where(result["interface_hits"][name]>0)
                ax.scatter(np.asarray(result["axes"][2])[ix],np.asarray(result["axes"][1])[iy],marker="x",s=30,c="black")
            fig.colorbar(image,ax=ax,label="m"); ax.set_title(f"{name}: {title}"); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    fig.suptitle("Independent plane reflections with measurement noise • fixed known 2000 m/s velocity\n"
                 f"18 arrivals; RMS {report['initial_rms_ms']:.3f} → {report['final_rms_ms']:.3f} ms; 20 noise refits; 5 m depth grid\n"
                 "Surfaces away from contacts are extrapolations; zero spread can reflect grid resolution")
    fig.savefig(root/"v11_interface_recovery.png",dpi=140,bbox_inches="tight",pad_inches=.15); plt.close(fig)


def main(destination):
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    result, truth = example(bootstrap=20)
    save_interfaces3d(result, root/"interfaces.npz")
    save_results({"joint_tomography_3d":result}, root/"project-results.npz")
    report = dict(arrivals=len(result["observed"]), initial_rms_ms=result["rms"][0]*1000,
                  final_rms_ms=result["rms"][-1]*1000, bootstrap=result["uncertainty_ensemble"]["replicates"],
                  uncertainty=result["uncertainty_ensemble"]["assumption"], interfaces={})
    for name in truth:
        observed = result["interface_hits"][name]>0
        report["interfaces"][name] = dict(controls=result["interface_controls"][name].tolist(),
            contact_nodes=int(observed.sum()), nodes=observed.size,
            mean_absolute_depth_error_m=float(np.mean(np.abs(result["sampled_interfaces"][name]-truth[name]))),
            conditional_std_max_m=float(np.nanmax(result["interface_std"][name])))
    render(result,truth,report,root)
    sources=np.array([[0,20,20],[0,20,80],[0,80,20],[0,80,80]],float)
    receivers=sources[::-1].copy()
    normal=np.array([1.,-.05,-.10]); d0=40.
    mirrored=sources-2*((sources@normal-d0)/(normal@normal))[:,None]*normal
    analytical=np.linalg.norm(mirrored-receivers,axis=1)/2000
    refinement=[]
    for step in (10,5):
        grid=NodeGrid3D((120//step+1,100//step+1,100//step+1),(step,step,step),radius=3)
        predicted=arrivals3d(grid,np.full(grid.shape,2000.),grid.nodes_for(sources)[0],grid.nodes_for(receivers)[0],
                            ["R"]*4,{"R":dict(kind="reflection",depth=d0,slope_x=.10,slope_y=.05)},rays=False)[0]
        refinement.append(dict(spacing_m=step,predicted_s=predicted.tolist(),reference_s=analytical.tolist(),
                               maximum_relative_error=float(np.max(np.abs(predicted-analytical)/analytical))))
    report["independent_dipping_plane_refinement"]=refinement
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        import os
        if os.name != "nt" and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("render scientific figures on Windows or a compute node")
        root=Path(sys.argv[1]); result=load_results(root/"project-results.npz")["joint_tomography_3d"]
        yy,xx=np.meshgrid(result["axes"][1],result["axes"][2],indexing="ij")
        render(result,{"dipping":25+.125*xx+.125*yy,"deep":np.full(xx.shape,55.)},
               json.loads((root/"report.json").read_text()),root)
    else:
        main(sys.argv[1])

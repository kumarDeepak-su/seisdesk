"""Real SEG-D against the independent reader, plus timing/amplitude diagnostic."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from seisdesk.core.segd import read_segd
from seisdesk.core.sampling import common_samples


def main(root):
    root=Path(root); destination=root/"v10-benchmark"; destination.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/"field-data/segd/reference"))
    from SEGD_reader.segd_reader import SEG_D_Reader
    path=root/"field-data/segd/smartsolo.segd"
    reader=SEG_D_Reader(str(path)); reader.open_file()
    reference=reader.get_data_plus_key_headers(); reader.close_file()
    expected=[]; intervals=[]
    for key,group in reference.items():
        if key == "file_header":continue
        desc=group["description"]
        mp=desc["descaleMultiplier"]*.25
        mp += np.sign(mp or 1)*desc["descaleMultiplierExtendedResolution"]
        for line,points in group["traceData"].items():
            for point,records in points.items():
                for number,trace in records.items():
                    expected.append((int(number),np.asarray(trace["trace_data"],dtype=np.float32)*2**mp))
    data=read_segd(path)
    assert data.ntr == len(expected)
    errors=[]
    for i,(number,trace) in enumerate(expected):
        assert data.header("tracf")[i] == number
        assert len(trace) == data.nt
        np.testing.assert_array_equal(data.traces[i],trace)
        errors.append(float(np.max(np.abs(data.traces[i]-trace))))
    report=dict(reference="SEGDreader/SEGD_reader",reference_commit="0dc57b6a1eecb266a05bc265031490bc5d9cbe68",
                source="https://github.com/SEGDreader/SEGD_reader/blob/0dc57b6a1eecb266a05bc265031490bc5d9cbe68/examples/data/453005563.0001.2024.06.30.19.35.34.000.Z.segd",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),traces=data.ntr,samples=data.nt,dt_s=data.dt,
                max_absolute_sample_difference=max(errors),sample_units=data.meta["sample_units"],
                timing_note="Header sample intervals and first-sample timing checked; geographic coordinates and proprietary continuous timestamps are not inferred.")
    assert data.dt == reference["file_header"]["dominant_sampling_interval_microsec"]*1e-6
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
    sample= data.traces[0]; stride=max(1,len(sample)//6000)
    axes[0].plot(data.times[::stride],sample[::stride],lw=.6)
    axes[0].set(title="Public SmartSolo SEG-D — first selected trace",xlabel="Time (s)",ylabel="Amplitude (mV)")
    dt=[.001,.002]; starts=[-.01325,.00275]
    signal=lambda t: np.sin(2*np.pi*23*t)+.3*np.cos(2*np.pi*61*t)
    rows=[signal(s+np.arange(2001)*d) for s,d in zip(starts,dt)]
    out,newdt,t0,_=common_samples(rows,dt,starts,.004,"intersection")
    time=t0+np.arange(out.shape[1])*newdt; keep=(time>.3)&(time<.5)
    axes[1].plot(time[keep],signal(time[keep]),"k-",label="Analytical bandlimited event")
    for i in range(2):axes[1].plot(time[keep],out[i,keep],"o",ms=3,label=f"Resampled input dt {dt[i]*1000:g} ms")
    axes[1].set(xlabel="Time (s)",ylabel="Amplitude",title="Mixed-rate, fractional-delay alignment onto 4 ms samples")
    axes[1].legend(fontsize=8)
    fig.savefig(destination/"v10_field_import.png",dpi=150); plt.close(fig)
    (destination/"field-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":main(sys.argv[1])

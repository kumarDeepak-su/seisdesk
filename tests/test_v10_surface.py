import numpy as np
import pytest
from scipy.optimize import brentq
from seisdesk.core.surface_inversion import predict_dispersion, invert_dispersion


def example():
    truth=np.array([[10,400,200,1800],[0,800,400,2000]],float)
    picks=[dict(frequency_hz=f,velocity_ms=250,sigma_ms=2,mode=0,wave=wave,kind="phase")
           for wave in ("rayleigh","love") for f in np.linspace(8,50,12)]
    for pick,v in zip(picks,predict_dispersion(truth,picks)):pick['velocity_ms']=float(v)
    initial=truth.copy(); initial[:,1:3] *= 1.15
    return truth,initial,picks


def test_rayleigh_halfspace_matches_secular_polynomial():
    vs,vp=300.,np.sqrt(3)*300
    q2=next(float(r.real) for r in np.roots([1,-8,24-16*(vs/vp)**2,-16*(1-(vs/vp)**2)]) if abs(r.imag)<1e-8 and 0<r.real<1)
    picks=[dict(frequency_hz=f,velocity_ms=vs,sigma_ms=2) for f in (3,10,80)]
    predicted=predict_dispersion([[0,vp,vs,1900]],picks)
    np.testing.assert_allclose(predicted,np.sqrt(q2)*vs,atol=.01,rtol=0)


def test_love_modes_match_independent_layer_equation():
    model=np.array([[10,400,200,1800],[0,800,400,2000]],float)
    picks=[]; expected=[]
    for mode in (0,1):
        for f in (30.,40.,50.):
            w=2*np.pi*f
            def equation(c):
                q=w*np.sqrt(1/200**2-1/c**2); p=w*np.sqrt(1/c**2-1/400**2)
                return q*10-np.arctan(2000*400**2*p/(1800*200**2*q))-mode*np.pi
            if equation(400-1e-6)<=0:continue
            expected.append(brentq(equation,200+1e-6,400-1e-6))
            picks.append(dict(frequency_hz=f,velocity_ms=250,mode=mode,wave="love"))
    np.testing.assert_allclose(predict_dispersion(model,picks),expected,atol=.02,rtol=0)


def test_group_velocity_matches_independent_phase_derivative():
    truth,_,picks=example()
    picks=picks[:12]
    c=predict_dispersion(truth,picks)
    low=predict_dispersion(truth,[dict(p,frequency_hz=p['frequency_hz']*.999) for p in picks])
    high=predict_dispersion(truth,[dict(p,frequency_hz=p['frequency_hz']*1.001) for p in picks])
    expected=c/(1-(high-low)/(.002*c))
    group=predict_dispersion(truth,[dict(p,kind='group') for p in picks])
    np.testing.assert_allclose(group,expected,atol=.2,rtol=0)
    with pytest.raises(ValueError,match="Dunkin"):
        predict_dispersion(truth,picks,"fast-delta")


def test_layer_recovery_bootstrap_and_cancellation():
    truth,initial,picks=example()
    result=invert_dispersion(initial,picks,[[100,350],[250,650]],starts=1,bootstrap=3,max_evaluations=80,damping=0)
    assert result['rms_ms'] < .1 and result['rms_ms'] < result['initial_rms_ms']/100
    np.testing.assert_allclose(result['model'][:,2],truth[:,2],rtol=.01)
    assert result['bootstrap_success'] == 3
    assert np.all(result['model_p95'][:,2] >= result['model_p05'][:,2])
    with pytest.raises(InterruptedError):
        invert_dispersion(initial,picks,[[100,350],[250,650]],starts=1,cancelled=lambda:True)


def test_thickness_recovery_and_missing_mode_rejection():
    truth,initial,picks=example(); initial[0,0]=13
    result=invert_dispersion(initial,picks,[[100,350],[250,650]],[[5,20]],starts=1,max_evaluations=120,damping=0)
    assert result['rms_ms'] < .15
    assert result['model'][0,0] == pytest.approx(10,rel=.02)
    with pytest.raises(ValueError,match="modes|mode"):
        predict_dispersion(truth,[dict(frequency_hz=.1,velocity_ms=250,mode=6,wave="love")])


def test_saved_flow_csv_model_and_results_round_trip(tmp_path):
    import csv
    from seisdesk.processing import create,Context
    from seisdesk.core.dataset import SeismicData
    from seisdesk.workflow.project import Project
    truth,initial,picks=example()
    with (tmp_path/'model.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['thickness_m','vp_ms','vs_ms','density_kgm3','vs_min','vs_max','h_min','h_max'])
        for row,lo,hi in zip(initial,[100,250],[350,650]):writer.writerow([*row,lo,hi,0,0])
    with (tmp_path/'picks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(picks[0]));writer.writeheader();writer.writerows(picks)
    data=SeismicData(np.ones((2,20)),.001,name='surface')
    context=Context(work_dir=str(tmp_path))
    output=create('surface_wave_inversion',model_file='model.csv',picks_file='picks.csv',starts=1)(data,context)
    np.testing.assert_array_equal(output.traces,data.traces)
    assert context.results['surface_wave:surface']['rms_ms'] < .1
    project=Project();project.context=context;project.add_dataset(output);project.save(str(tmp_path/'project'))
    back=Project.load(str(tmp_path/'project'))
    np.testing.assert_array_equal(back.context.results['surface_wave:surface']['model'],context.results['surface_wave:surface']['model'])

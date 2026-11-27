import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location("motion_length",Path(__file__).resolve().parents[1]/"scripts/experiments/analyze_motion_length.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_blocked_linear_scale_recovers_smooth_trend():
    t=np.arange(100,dtype=float)
    f=np.column_stack((t.astype(int),t.astype(int)+10))
    y=2+.003*t
    pred,extra,fold,records=m.blocked_predictions(f,np.zeros(100,dtype=int),t,y)
    assert np.isfinite(pred[1]).all()
    np.testing.assert_allclose(pred[1],y,atol=1e-8)
    assert np.mean(abs(pred[0]-y))>.01
    for r in records:
        train=(f[:,1]<r["test_start"])|(f[:,0]>r["test_end"])
        assert train.sum()==r["train"]
    assert extra.any()


def test_lag_excludes_shared_endpoints():
    f=np.column_stack((np.arange(40),np.arange(40)+10))
    r=m.lag_correlations(f,np.zeros(40,dtype=int),np.arange(40,dtype=float),[10,11])
    assert r[0]["nonoverlap_count"]==0
    assert r[1]["nonoverlap_count"]==29

import numpy as np
import pytest
from spade_calib.calibration.fixed_edge_refinement import (
    Frame,edge_distance_field,residual_frame,transform_from_delta,delta_from_transform,
    robust_values,fit,extrinsic_error,
)


def test_exact_integer_edge_and_invalid_projection_cost():
    K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    field=edge_distance_field(np.array([[50.,50.]]),(101,101))
    f=Frame(0,np.array([[0.,0.,5.],[0.,0.,5.]]),field)
    X=np.eye(4)
    r,v,_=residual_frame(f,X,K)
    assert np.allclose(r,0) and v.all()
    X[0,3]=10
    r,v,_=residual_frame(f,X,K)
    assert len(r)==2 and not v.any() and np.all(r>50)
    X=np.eye(4);X[2,3]=-6
    r,v,_=residual_frame(f,X,K)
    assert not v.any() and np.all(r>50)


def test_chart_round_trip():
    X0=np.eye(4);X0[:3,3]=[1.,2.,3.]
    d=np.array([.2,-.5,.7,.1,-.2,.3])
    X=transform_from_delta(X0,d)
    assert np.allclose(delta_from_transform(X0,X),d)
    assert extrinsic_error(X,X)["translation_m"]==0


def test_translation_fit_with_fixed_support():
    K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    # Distinct isolated image features, x shift 0.25m = 5 pixels at z5.
    xyz=np.array([[-1.,-1.,5.],[1.,1.,5.],[0.,1.,5.]])
    field=edge_distance_field(np.array([[35,30],[75,70],[55,70]]),(101,101))
    f=Frame(0,xyz,field)
    X,info=fit([f],np.eye(4),K,"xz",np.zeros(6),max_nfev=100)
    assert abs(X[0,3]-.25)<.01
    assert abs(X[2,3])<.02
    assert np.allclose(X[:3,:3],np.eye(3)) and X[1,3]==0


def test_robust_cost_matches_formula():
    r=np.array([0.,1.,10.,100.]);s=5.
    assert np.allclose(robust_values(r,s),2*s*s*(np.sqrt(1+(r/s)**2)-1))
    with pytest.raises(ValueError):edge_distance_field(np.empty((0,2)),(10,10))

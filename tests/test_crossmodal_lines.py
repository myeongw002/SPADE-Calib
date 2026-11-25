import numpy as np
import pytest
from spade_calib.calibration.crossmodal_lines import (
    fit_wall,finite_intersection,fit_lines,project_line,information,perturb)


def test_supported_plane_intersection():
    rng=np.random.default_rng(17)
    z=rng.uniform(0,3,200);a=rng.uniform(0,2,200)
    p=np.column_stack((np.full(200,8.),a,z))
    q=np.column_stack((8+a,np.zeros(200),z))
    n,d,m,_=fit_wall(p)
    n2,d2,m2,_=fit_wall(q)
    endpoints,r=finite_intersection(n,d,p[m],n2,d2,q[m2])
    np.testing.assert_allclose(endpoints[:,0],8,atol=1e-9)
    np.testing.assert_allclose(endpoints[:,1],0,atol=1e-9)
    assert r["length_m"]>.75
    with pytest.raises(ValueError):
        finite_intersection(n,d,p,n2,d2,q+[5,0,0])


def test_x_only_exact_recovery_and_rank():
    K=np.array([[700.,0,620],[0,700,180],[0,0,1]])
    X=np.eye(4)
    records=[]
    for i,(x,z) in enumerate([(-2,8),(3,12),(1,20)]):
        p=np.array([[x,-1,z],[x,1,z]],float)
        uv,_=project_line(p,X,K)
        records.append(dict(id=str(i),frame=i,xyz_lidar=p,uv_image=uv))
    initial=perturb(X,[0,0,0,.25,0,0])
    Y,r=fit_lines(records,initial,K,mode="x_only")
    assert r["success"]
    np.testing.assert_allclose(Y,X,atol=1e-7)
    assert information(Y,records,K)["effective_rank"]<6


def test_linked_roi_orientation_and_original_point_identity():
    from spade_calib.calibration.crossmodal_lines import lidar_bev_xy,polygon_roi_mask
    xyz=np.array([[5.,2.,0.],[5.,-2.,1.],[9.,-2.,1.],[6.,-2.,2.],[2.,0.,0.]])
    bev=lidar_bev_xy(xyz)
    np.testing.assert_array_equal(bev,[[-2,5],[2,5],[2,9],[2,6],[0,2]])
    ids=np.array([3,0,1])
    uv=np.array([[100,100],[200,200],[105,105]],float)
    poly_bev=[[1,4],[3,4],[3,7],[1,7]]
    poly_img=[[95,95],[102,95],[102,102],[95,102]]
    np.testing.assert_array_equal(polygon_roi_mask(xyz,ids,uv,{"bev":poly_bev}),
                                  [False,True,False,True,False])
    np.testing.assert_array_equal(polygon_roi_mask(xyz,ids,uv,{"image":poly_img}),
                                  [False,False,False,True,False])
    np.testing.assert_array_equal(polygon_roi_mask(xyz,ids,uv,{"image":poly_img,"bev":poly_bev}),
                                  [False,False,False,True,False])

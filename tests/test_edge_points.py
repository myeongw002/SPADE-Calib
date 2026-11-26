import numpy as np
from spade_calib.calibration.edge_points import (
    dense_edges, plane_normals, sparse_projection, sparse_edges,
)

K=np.array([[100.,0.,50.],[0.,100.,40.],[0.,0.,1.]])

def test_sparse_original_ids_and_unknown():
    # Pixel coordinates 10, 15, 20 on same row; near at 10/15,
    # far at 20, an occluded far return at pixel 15.
    uv=np.array([[10.2,30.2],[15.2,30.2],[20.2,30.2],[15.2,30.2]])
    z=np.array([5.,5.,10.,10.])
    xyz=(np.c_[uv,np.ones(4)]@np.linalg.inv(K).T)*z[:,None]
    q,D,I=sparse_projection(xyz,np.eye(4),K,(80,101))
    assert I[30,15]==1 and np.isnan(D[30,16]) and I[30,16]==-1
    e=sparse_edges(q,xyz,K,radius=6.)
    assert e["point_indices"].tolist()==[1]
    assert e["background_indices"].tolist()==[2]
    assert e["support_indices"].tolist()==[0]
    assert np.allclose(e["normal_uv"],[[1,0]])
    assert np.allclose(e["bracket_width_px"],5)
    assert np.allclose(e["xyz_lidar"],xyz[[1]])
    q,_,_=sparse_projection(xyz[:2],np.eye(4),K,(80,101))
    assert len(sparse_edges(q,xyz[:2],K,radius=6.)["uv"])==0

def test_dense_jump_and_smooth():
    y,x=np.mgrid[:80,:101]
    jump=(x<50).astype(np.float32)+.001*y
    a=dense_edges(jump,K)["jump"]
    assert len(a["uv"])>20
    assert np.max(abs(a["uv"][:,0]-49.5))<=2
    assert np.median(a["normal_uv"][:,0])>.9
    b=dense_edges(3*jump+7,K)["jump"]
    # Near-tied NMS may choose either pixel after float rounding.
    assert abs(np.median(a["uv"][:,0])-np.median(b["uv"][:,0]))<=1
    smooth=dense_edges(x+.1*y,K)
    assert len(smooth["jump"]["uv"])==0
    assert len(smooth["crease"]["uv"])==0

def test_crease_and_plane_normal():
    y,x=np.mgrid[:80,:101]
    e=dense_edges(abs(x-50)+.001*y,K)["crease"]
    assert len(e["uv"])>20
    assert np.max(abs(e["uv"][:,0]-50))<=2
    assert np.allclose(plane_normals(np.array([[50.,40.]]),np.array([[1.,0.]]),K),[[1,0,0]])

def test_empty_cloud():
    p=np.empty((0,3))
    q,D,I=sparse_projection(p,np.eye(4),K,(80,101))
    e=sparse_edges(q,p,K)
    assert np.isnan(D).all() and (I==-1).all()
    assert e["uv"].shape==(0,2) and e["plane_normal"].shape==(0,3)

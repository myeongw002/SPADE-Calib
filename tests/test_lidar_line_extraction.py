import numpy as np
from dataclasses import replace
from spade_calib.calibration.lidar_line_extraction import Settings,fit_plane,intersection,extract_lines,line_relation


def corner():
    u,z=np.meshgrid(np.linspace(.01,.98,20),np.linspace(.01,1.98,40))
    a=np.column_stack((u.ravel(),np.zeros(u.size),z.ravel()))
    b=np.column_stack((np.zeros(u.size),u.ravel(),z.ravel()))
    p=np.vstack((a,b))+[4,2,0]
    scans=np.tile(np.arange(5),len(p)//5)
    return p,scans,len(a)


def test_intersection_and_direction():
    p,s,n=corner();cfg=Settings()
    a=fit_plane(p,np.arange(n),cfg);b=fit_plane(p,np.arange(n,len(p)),cfg)
    l,reason=intersection(p,s,a,b,cfg)
    assert reason=="ok"
    np.testing.assert_allclose(l["xyz"][:,:2],[[4,2],[4,2]],atol=1e-10)
    assert np.linalg.norm(np.diff(l["xyz"],axis=0))>1.5
    # Rotate exact corner to horizontal: no vertical restriction.
    q=p[:,[2,1,0]]
    a=fit_plane(q,np.arange(n),cfg);b=fit_plane(q,np.arange(n,len(p)),cfg)
    l,reason=intersection(q,s,a,b,cfg)
    assert reason=="ok"
    assert abs(np.diff(l["xyz"],axis=0)[0,0])>1.5


def test_reject_extrapolated_corner_and_collinear():
    p,s,n=corner();cfg=Settings()
    p[:n,0]+=1.5
    a=fit_plane(p,np.arange(n),cfg);b=fit_plane(p,np.arange(n,len(p)),cfg)
    assert intersection(p,s,a,b,cfg)[0] is None
    q=np.column_stack((np.arange(40),np.zeros(40),np.zeros(40)))
    assert fit_plane(q,np.arange(40),cfg) is None


def test_automatic_corner():
    p,s,_=corner()
    raw,merged,stats=extract_lines(p,s,replace(Settings(),cell_m=2.,trials=128))
    assert raw and merged and stats["merged_lines"]<=stats["raw_lines"]
    ref=np.array([[4,2,.1],[4,2,1.9]])
    assert any(line_relation(ref,l["xyz"])["mean_distance_m"]<.10 and
               line_relation(ref,l["xyz"])["angle_deg"]<5 for l in merged)
    assert extract_lines(p,np.zeros(len(p),int),Settings())[0]==[]

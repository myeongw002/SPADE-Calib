import numpy as np
from dataclasses import replace
from spade_calib.calibration.camera_guided_lines import GuideSettings,active_cells,extract_guided,associate
from spade_calib.calibration.lidar_line_extraction import Settings


def test_finite_corridor_and_depth_cells():
    K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    p=np.array([[0,0,5],[0,0,10],[0,2,5],[2,0,5],[0,0,-5.]])
    cfg=Settings(cell_m=1.)
    guide=replace(GuideSettings(),min_image_length_px=10, min_cell_votes=1,
                  roi_width_px=2,endpoint_margin_px=0,min_projection_length_px=5)
    cells,ids,stats=active_cells(p,[[50,45,50,55]],np.eye(4),K,(100,100),cfg,guide)
    assert set(cells)=={(0,0,5),(0,0,10)}
    assert len(ids)==5 and stats["occupied_cells"]==5


def test_guided_corner_and_original_geometry():
    u,z=np.meshgrid(np.linspace(.01,.8,20),np.linspace(-.9,.9,40))
    a=np.column_stack((u.ravel(),z.ravel(),np.zeros(u.size)))
    b=np.column_stack((np.zeros(u.size),z.ravel(),u.ravel()))
    p=np.vstack((a,b))+[0,0,5]
    scans=np.arange(len(p))%5
    K=np.array([[200.,0,100],[0,200,100],[0,0,1.]])
    s=[[100,60,100,140]]
    lines,stats=extract_guided(p,scans,s,np.eye(4),K,(200,200),replace(Settings(),trials=128))
    assert stats["active_cells"]>0 and lines
    rows=associate(lines,s,np.eye(4),K,(200,200),require_guide=True)
    assert any(r["status"]=="proposed" for r in rows)
    for line in lines:
        assert all(np.all(q["ids"]<len(p)) for q in line["planes"])


def test_competing_image_lines_and_guide_mismatch():
    K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    line=dict(xyz=np.array([[0,-1,5],[0,1,5]]),guide_votes={1:10})
    guide=replace(GuideSettings(),min_image_length_px=20)
    s=[[50,30,50,70],[80,30,80,70]]
    r=associate([line],s,np.eye(4),K,(100,100),guide,require_guide=True)
    assert r[0]["status"]=="guide_mismatch"
    s=[[49,30,49,70],[51,30,51,70]]
    r=associate([line],s,np.eye(4),K,(100,100),guide,require_guide=True)
    assert r[0]["status"]=="ambiguous"

import numpy as np
from spade_calib.calibration.multiview_edge_evidence import (
    load_beams,recover_rings,ring_edges,recurrence,projected_candidates,
)

def test_xml_and_offset_recovery(tmp_path):
    angles=np.linspace(-25,2,64)
    xml="<db>"+"".join(f"<px><id_>{i}</id_><vertCorrection_>{a}</vertCorrection_><vertOffsetCorrection_>20</vertOffsetCorrection_><horizOffsetCorrection_>3</horizOffsetCorrection_></px>" for i,a in enumerate(angles))+"</db>"
    p=tmp_path/"db.xml";p.write_text(xml)
    b=load_beams(p)
    k=np.array([0,20,63]);a=np.deg2rad(angles[k]);d=np.array([5.,10.,20.])
    x=d*np.cos(a)-.2*np.sin(a);z=d*np.sin(a)+.2*np.cos(a)
    pts=np.column_stack((x,np.full(3,.03),z))
    r=recover_rings(pts,b)
    assert np.array_equal(r["ring"],k)
    assert max(r["error_deg"])<1e-10
    r=recover_rings(pts,b,allowed_ranks=[20])
    assert r["ring"].tolist()==[-1,20,-1]

def test_same_ring_cyclic_and_gap():
    az=np.deg2rad([359.6,0.,.4,0.8])
    r=np.array([5.,5.,10.,10.]);p=np.c_[r*np.cos(az),r*np.sin(az),np.zeros(4)]
    e=ring_edges(p,np.zeros(4,int))
    assert e.tolist()==[[1,2,0]]
    assert len(ring_edges(p,np.array([0,0,1,1])))==0
    assert len(ring_edges(p,np.zeros(4,int),az_gap_deg=.2))==0

def test_distinct_scan_support():
    p=np.array([[10.,0,0],[10.01,0,0],[10.1,0,0],[20.,0,0]])
    s=np.array([0,0,1,2]);o=np.array([[0.,0,0],[0,1,0],[0,2,0]])
    r=recurrence(p,s,o,radius=.25)
    assert r["scan_count"].tolist()==[2,2,2,1]
    assert r["support_indices"][2,0]==1
    assert r["view_angle_deg"][0]>0
    assert recurrence(p[:0],s[:0],o)["support_indices"].shape==(0,3)

def test_visibility():
    p=np.array([[2.,2.,2.],[6.,4.,2.]])
    depth=np.full((10,10),np.nan);depth[1,1]=1.
    q=projected_candidates(p,np.eye(4),np.eye(3),(10,10),depth)
    assert q["visible"].tolist()==[False,True]
    assert q["reference_known"].tolist()==[True,False]

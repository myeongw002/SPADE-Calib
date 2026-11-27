import numpy as np
from spade_calib.camera.segment_mvs import read_text_model,subset_text_model,anchor_metric

def test_subset_keeps_indices_and_world_gauge(tmp_path):
    p=tmp_path/"global";p.mkdir()
    (p/"cameras.txt").write_text("1 PINHOLE 100 100 50 50 50 50\n")
    (p/"images.txt").write_text("1 1 0 0 0 0 0 0 1 a.png\n1 2 7 3 4 8\n2 1 0 0 0 1 0 0 1 b.png\n5 6 7\n3 1 0 0 0 2 0 0 1 c.png\n7 8 8\n")
    (p/"points3D.txt").write_text("7 1 2 3 255 0 0 0.1 1 0 2 0\n8 4 5 6 0 255 0 0.2 1 1 3 0\n")
    out=tmp_path/"segment"
    report=subset_text_model(read_text_model(p),["a.png","b.png"],out)
    assert report["points"]==1 and report["observations"]==2
    _,imgs,pts=read_text_model(out)
    assert imgs["a.png"][1]==["1","2","7","3","4","-1"]
    assert imgs["b.png"][0][5]=="1"
    assert pts[7][1:4]==["1","2","3"]
    assert set(pts)=={7}

def test_metric_transform_world_to_camera():
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    t=np.array([1.,2.,3.]);p=np.array([[2.,1.,4.]])
    assert np.allclose(anchor_metric(p,R,t,2),[[0,8,14]])

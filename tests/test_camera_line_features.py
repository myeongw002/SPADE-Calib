"""Camera-only E1 unit and cached-ELSED integration checks."""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from spade_calib.camera.line_features import (
    line_geometry, depth_profiles, select_arms, geometry_metrics,
)

K = np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1.]])


def test_normal_and_equal_budget():
    s = np.array([[[50, 10], [50, 70]], [[20, 10], [20, 70]],
                  [[80, 10], [80, 70]], [[10, 40], [90, 40]]], float)
    length, n = line_geometry(s, K)
    assert np.allclose(abs(n[0]), [1,0,0])
    assert np.allclose(abs(n[-1]), [0,1,0])
    assert n[1,0]*n[1,2] * n[2,0]*n[2,2] < 0
    profiles = np.array([[.4,.2,.4,1], [.5,.3,.5,1], [.2,.8,.8,1], [np.nan]*3+[0]])
    arms, pool = select_arms(length,n,2,17,profiles)
    assert len(arms["all"]) == 4
    for name, ids in arms.items():
        if name != "all":
            assert len(ids) == 2 and set(ids) <= set(pool)
    assert geometry_metrics(s[arms["normal"]],n[arms["normal"]],(80,100))["normal_eigenvalues"][0] > 0


def test_depth_affine_and_types():
    y,x = np.mgrid[:80,:101]
    s = np.array([[[50,10],[50,70]], [[1,10],[1,70]]],float)
    linear = (x + .1*y).astype(np.float32)
    jump = linear + 60*(x>=50)
    crease = np.abs(x-50)+.1*y
    a = depth_profiles(linear,s)
    b = depth_profiles(jump,s)
    c = depth_profiles(crease,s)
    assert max(a[0,:2]) < 1e-4
    assert b[0,0] > .5 and b[0,1] < 1e-4
    assert c[0,1] > .3 and c[0,0] < 1e-4
    assert np.allclose(depth_profiles(jump*3+7,s),b,atol=1e-5,equal_nan=True)
    assert np.isnan(a[1,:3]).all()
    with pytest.raises(ValueError):
        depth_profiles(np.ones((80,101)),s)


def test_empty():
    s = np.empty((0,2,2))
    length,n = line_geometry(s,K)
    arms,pool = select_arms(length,n,10,17,np.empty((0,4)))
    assert all(len(a)==0 for a in arms.values())
    assert geometry_metrics(s,n,(80,100))["normal_eigenvalues"] is None


def test_camera_only_cli(tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/experiments/run_camera_line_features.py"
    spec = importlib.util.spec_from_file_location("e1_runner",path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    images = tmp_path/"images"
    images.mkdir()
    cache = tmp_path/"cache"/"frame_000069"
    cache.mkdir(parents=True)
    img = np.zeros((80,101,3),np.uint8)
    assert cv2.imwrite(str(images/"000069.png"),img)
    assert cv2.imwrite(str(cache/"original.png"),img)
    s = np.array([[20,10,20,70],[50,10,50,70],[80,10,80,70],[10,40,90,40]])
    np.savez(cache/"elsed_segments.npz",segments=s,scores=np.ones(len(s)))
    cal = tmp_path/"calib.txt"
    cal.write_text("P2: 100 0 50 0 0 100 40 0 0 0 1 0\nTr: intentionally not parsed\n")
    d = tmp_path/"depth"
    d.mkdir()
    y,x = np.mgrid[:80,:101]
    np.save(d/"000069.npy",x+.1*y+60*(x>=50))
    out = tmp_path/"e1"
    assert runner.main(["--image-dir",str(images),"--calib",str(cal),"--elsed-dir",
                        str(cache.parent),"--frames","69","--budget","2",
                        "--depth-dir",str(d),"--out",str(out)]) == 0
    summary = json.loads((out/"summary.json").read_text())
    assert summary["aggregate"]["all"]["count"] == 4
    assert summary["aggregate"]["fusion"]["count"] == 2
    assert summary["depth_mode"] == "external_unverified"
    with (out/"review.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["geometry_label"] == "" for r in rows)
    assert (out/"frame_000069"/"comparison.png").is_file()

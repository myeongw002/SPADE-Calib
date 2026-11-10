"""Deterministic synthetic tests; no user recording is distributed in CI."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.io.colmap import RegisteredImage
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.selection.trajectory import (
    SelectionSettings, match_trajectories, build_motion_pairs,
    estimate_rotation_and_sfm_scale, select_scenes, write_scene_selection,
    from_recording_index,
)


def synthetic_data(motion='rich'):
    n=301
    ns=np.int64(1_800_000_000_000_000_000)+np.arange(n,dtype=np.int64)*100_000_000
    s=np.arange(n)*.1
    if motion=='rich':
        rot=Rotation.from_euler('xyz',np.stack((.12*np.sin(s*.2),
                                             .18*np.sin(s*.32),
                                             .2*np.sin(s*.25)),axis=1))
    elif motion=='yaw':
        rot=Rotation.from_euler('z',(.20*np.sin(s*.25))[:,None])
    else:
        rot=Rotation.from_euler('z',np.zeros((n,1)))
    pos=np.stack((.4*s,.1*np.sin(.15*s),.09*np.cos(.27*s)),axis=1)
    R_X=Rotation.from_euler('xyz',[.05,.08,-.12]).as_matrix()
    t_X=np.array([.15,-.05,.1])
    sfm_scale=8.
    camera_R=rot.as_matrix()@R_X
    camera_p=(pos+np.einsum('nij,j->ni',rot.as_matrix(),t_X))/sfm_scale + np.array([2.,-3.,1.])
    records=[]
    for i,(R,p) in enumerate(zip(camera_R,camera_p)):
        wxyz=Rotation.from_matrix(R.T).as_quat()[[3,0,1,2]]
        tvec=-R.T@p
        records.append(RegisteredImage(i+1,f'{int(ns[i])}.jpg',1,tuple(wxyz),tuple(tvec)))
    trajectory=GlimTrajectory.from_array(np.column_stack((ns.astype(float)*1e-9,pos,rot.as_quat())))
    return trajectory,records


def test_handeye_convention_and_metric_scale():
    glim,images=synthetic_data()
    samples=match_trajectories(glim,images)
    p=build_motion_pairs(samples,int(samples.timestamps_ns[0]),int(samples.timestamps_ns[-1]),SelectionSettings())
    R,s,details=estimate_rotation_and_sfm_scale(p)
    assert details['rotation_residual_median_deg']<.01
    assert abs(s-8)<.05
    np.testing.assert_allclose(R,Rotation.from_euler('xyz',[.05,.08,-.12]).as_matrix(),atol=.003)


def test_yaw_only_has_unobservable_rotation_axis():
    glim,images=synthetic_data('yaw')
    samples=match_trajectories(glim,images)
    result=select_scenes(samples,SelectionSettings(window_s=12,window_stride_s=6,
                                                   min_pairs=5,max_scenes=2))
    assert result['full_recording_diagnostics']['rotation_excitation_deg'][0]<.01
    assert result['selected_count']>0
    assert result['provisional_handeye']['scale_is_positive']


def test_scene_selection_is_disjoint_and_deterministic():
    glim,images=synthetic_data()
    samples=match_trajectories(glim,images)
    cfg=SelectionSettings(window_s=10,window_stride_s=5,min_pairs=5,max_scenes=2)
    r=select_scenes(samples,cfg)
    t=select_scenes(samples,cfg)
    assert [x['id'] for x in r['selected']]==[x['id'] for x in t['selected']]
    assert len(r['candidates'])>2 and len(r['selected'])==2
    a,b=r['selected']
    assert a['stop_ns']<=b['start_ns'] or b['stop_ns']<=a['start_ns']
    assert r['selected_union_score']>0


def test_sparse_lidar_coverage_cannot_pass_scene_selection(tmp_path):
    glim,images=synthetic_data()
    samples=match_trajectories(glim,images)
    cfg=SelectionSettings(window_s=10,window_stride_s=5,min_pairs=5,max_scenes=2)
    # Two isolated raw scans cannot cover a ten-second Scene.
    stamps=samples.timestamps_ns[[30,32]]
    report=select_scenes(samples,cfg,lidar_scan_stamps_ns=stamps)
    assert report['selected_count']==0
    assert all(not c['lidar_coverage_valid'] for c in report['candidates'])
    out=tmp_path/'report.json'
    write_scene_selection(report,out)
    assert json.loads(out.read_text())['asset_coverage_checked']


def test_full_lidar_coverage_passes():
    glim,images=synthetic_data()
    samples=match_trajectories(glim,images)
    cfg=SelectionSettings(window_s=10,window_stride_s=5,min_pairs=5,max_scenes=2)
    report=select_scenes(samples,cfg,lidar_scan_stamps_ns=samples.timestamps_ns[::5])
    assert report['selected_count']==2
    assert all(s['lidar_coverage_valid'] for s in report['selected'])


def test_camera_sfm_clock_mismatch_and_duplicate_timestamps():
    glim,images=synthetic_data()
    with pytest.raises(ValueError,match='overlapping'):
        match_trajectories(glim,images,camera_to_lidar_offset_ns=100_000_000_000)
    with pytest.raises(ValueError,match='Duplicate'):
        match_trajectories(glim,[images[0],images[0],*images[1:]])
    with pytest.raises(ValueError,match='timestamp'):
        match_trajectories(glim,[RegisteredImage(1,'foo.jpg',1,(1,0,0,0),(0,0,0)),*images])


def test_recording_index_requires_sfm_and_lidar_availability():
    glim,images=synthetic_data()
    class Timed:
        def __init__(self,n): self.relative_path=n
    class A:
        def __init__(self,img,yes):
            self.image=Timed(img.name)
            self.sfm=img
            self.lidar=object() if yes else None
            self.glim_covered=True
    index=SimpleNamespace(associations=[A(x,i%2==0) for i,x in enumerate(images)],
                          glim_trajectory=glim,
                          config=SimpleNamespace(camera_to_lidar_offset_ns=0))
    s=from_recording_index(index)
    assert len(s.timestamps_ns)<len(images)
    assert len(s.timestamps_ns)>100


def test_settings_validation():
    with pytest.raises(ValueError):SelectionSettings(window_s=1.,pair_baseline_s=2.)
    with pytest.raises(ValueError):SelectionSettings(max_lidar_gap_s=0.)
    with pytest.raises(ValueError):SelectionSettings(min_lidar_scans_per_window=1)
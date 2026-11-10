"""Synthetic tests of recording indexing and no-implied-synchronization policy."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import yaml

from spade_calib.cli import main
from spade_calib.io.recording import (index_recording, load_recording_config,
                                      write_recording_index)


BASE = 1791439637016960512


def _cloud(path, *, with_time=True, ascii_mode=False):
    fields = "x y z intensity ring time" if with_time else "x y z intensity ring"
    sizes = "4 4 4 4 2 4" if with_time else "4 4 4 4 2"
    types = "F F F F U F" if with_time else "F F F F U"
    values = [(2.,0.,0.,3.,1,0.001),(3.,0.,0.,4.,2,0.002)]
    header = (f"# .PCD v0.7\nVERSION .7\nFIELDS {fields}\nSIZE {sizes}\n"
              f"TYPE {types}\nCOUNT {'1 ' * (6 if with_time else 5)}\n"
              f"WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA {'ascii' if ascii_mode else 'binary'}\n").encode()
    if ascii_mode:
        data = ("\n".join(" ".join(map(str, row if with_time else row[:-1]))
                          for row in values) + "\n").encode()
    else:
        dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<f4'),('ring','<u2')]
                      + ([('time','<f4')] if with_time else []))
        arr = np.zeros(2,dtype=dt)
        for i,row in enumerate(values):
            for j,key in enumerate(dt.names): arr[key][i]=row[j]
        data=arr.tobytes()
    path.write_bytes(header+data)


def make_config(tmp_path, *, mapped=False):
    root = tmp_path / 'raw'
    images=root/'images'; scans=root/'points'; sfm=root/'sfm'
    for d in (images,scans,sfm): d.mkdir(parents=True)
    f1 = f"{BASE}"
    f2 = f"{BASE + 100_000_000}"
    (images/(f1+'.jpg')).write_bytes(b'jpeg')
    (images/(f2+'.jpg')).write_bytes(b'jpeg')
    for f in (f1,f2): _cloud(scans/(f+'.pcd'))
    (sfm/'camera_poses.csv').write_text(
        'image_id,image_name,camera_id,qw,qx,qy,qz,tx,ty,tz\n'
        f'1,{f1}.jpg,1,1,0,0,0,0,0,0\n')
    # GLIM covers both frames.
    t0=(BASE-500_000_000)*1e-9
    np.savetxt(root/'traj_lidar.txt', np.array([[t0,0,0,0,0,0,0,1],
                                             [t0+2.,1,0,0,0,0,0,1]]),fmt='%.9f')
    cfg={
        'schema_version':2, 'dataset':{'name':'full_take','root':'raw'},
        'inputs':{'images':'images','lidar_scans':'points','sfm':'sfm','glim':'traj_lidar.txt'},
        'matching': {'max_pair_delta_ms':20,'camera_to_lidar_offset_ns':0},
        'pcd': {'time_field':'time','time_unit':'s','range_m':[0.5,100]},
    }
    conf=tmp_path/'recording.yaml'
    conf.write_text(yaml.safe_dump(cfg))
    return conf, root


def test_full_dataset_index_is_sorted_and_lazy(tmp_path):
    cfg,_=make_config(tmp_path)
    recording=index_recording(load_recording_config(cfg))
    assert len(recording.images)==len(recording.lidar_scans)==2
    assert [r.image.timestamp_ns for r in recording.associations]==[BASE,BASE+100_000_000]
    assert [r.sfm is not None for r in recording.associations]==[True,False]
    assert [r.lidar is not None for r in recording.associations]==[True,True]
    assert all(x.glim_covered for x in recording.associations)
    cloud=recording.load_lidar_scan(1)
    assert cloud.scan_id==1
    np.testing.assert_array_equal(cloud.source_return_index,[0,1])
    np.testing.assert_allclose(cloud.time_offset_s,[0.001,0.002],atol=1e-8)
    assert len(recording.lidar_window(BASE-1,BASE+1))==1
    selected=recording.load_window_scans(recording.lidar_window(BASE-1,BASE+100_000_001))
    assert [x.scan_id for x in selected]==[0,1]
    manifest=write_recording_index(recording,tmp_path/'outputs'/'index.json')
    assert manifest['summary']['images_with_sfm_pose']==1
    assert manifest['summary']['lidar_scans_within_glim_range']==2
    assert len(manifest['metadata_fingerprint_sha256'])==64
    assert 'sfm_qvec_wxyz' in manifest['camera_associations'][0]
    assert json.loads((tmp_path/'outputs'/'index.json').read_text())==manifest


def test_cli_index_success_and_no_legacy_breakage(tmp_path, capsys):
    cfg,_=make_config(tmp_path)
    out=tmp_path/'index.json'
    assert main(['index','--config',str(cfg),'--output',str(out)])==0
    assert json.loads(out.read_text())['summary']['images']==2
    assert 'full_take' in capsys.readouterr().out
    assert main(['index','--config',str(tmp_path/'absent.yaml'),'--output',str(out)])==2


def test_explicit_clock_offset_and_tolerance(tmp_path):
    cfg,root=make_config(tmp_path)
    doc=yaml.safe_load(cfg.read_text())
    doc['matching']['camera_to_lidar_offset_ns']=100_000_000
    doc['matching']['max_pair_delta_ms']=10
    cfg.write_text(yaml.safe_dump(doc))
    rec=index_recording(load_recording_config(cfg))
    assert [a.lidar.timestamp_ns if a.lidar else None for a in rec.associations]==[BASE+100_000_000,None]
    assert [a.delta_ns for a in rec.associations]==[0,None]


def test_short_filenames_require_timestamp_csv(tmp_path):
    cfg,root=make_config(tmp_path)
    (root/'images'/f'{BASE}.jpg').rename(root/'images'/'a.jpg')
    with pytest.raises(ValueError,match='supply CSV'):
        index_recording(load_recording_config(cfg))
    with (root/'cam_times.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['file','timestamp_ns']);w.writerow(['a.jpg',BASE]);
        w.writerow([f'{BASE+100_000_000}.jpg',BASE+100_000_000])
    doc=yaml.safe_load(cfg.read_text());doc['inputs']['image_timestamps']='cam_times.csv'
    cfg.write_text(yaml.safe_dump(doc))
    assert index_recording(load_recording_config(cfg)).associations[0].image.relative_path=='a.jpg'


def test_duplicate_timestamp_is_rejected(tmp_path):
    cfg,root=make_config(tmp_path)
    (root/'images'/f'{BASE}.jpg').rename(root/'images'/'dup.jpg')
    with (root/'cam_times.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['file','timestamp_ns']);w.writerow(['dup.jpg',BASE+100_000_000]);
        w.writerow([f'{BASE+100_000_000}.jpg',BASE+100_000_000])
    doc=yaml.safe_load(cfg.read_text());doc['inputs']['image_timestamps']='cam_times.csv'
    cfg.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='Duplicate timestamps'):
        index_recording(load_recording_config(cfg))


def test_missing_per_return_timestamp_rejected(tmp_path):
    cfg,root=make_config(tmp_path)
    _cloud(root/'points'/f'{BASE}.pcd',with_time=False)
    rec=index_recording(load_recording_config(cfg))
    with pytest.raises(ValueError,match="requires scalar field 'time'"):
        rec.load_lidar_scan(0)


def test_config_rejects_bad_range_and_invalid_offset(tmp_path):
    cfg,_=make_config(tmp_path)
    doc=yaml.safe_load(cfg.read_text());doc['pcd']['range_m']=[10,5]
    cfg.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='range'):
        load_recording_config(cfg)
    doc['pcd']['range_m']=[.5,100];doc['matching']['camera_to_lidar_offset_ns']=1.5
    cfg.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='int64'):
        load_recording_config(cfg)


def test_unregistered_sfm_remains_missing_and_manifest_does_not_overwrite_source(tmp_path):
    cfg,_=make_config(tmp_path)
    rec=index_recording(load_recording_config(cfg))
    assert rec.associations[1].sfm is None
    with pytest.raises(ValueError,match='Refusing to overwrite'):
        write_recording_index(rec,rec.images[0].path)


def test_ascii_pcd_and_source_return_indices(tmp_path):
    cfg,root=make_config(tmp_path)
    _cloud(root/'points'/f'{BASE}.pcd',ascii_mode=True)
    rec=index_recording(load_recording_config(cfg))
    scan=rec.load_lidar_scan(0)
    assert scan.xyz.shape==(2,3)
    np.testing.assert_array_equal(scan.ring,[1,2])
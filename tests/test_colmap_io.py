"""COLMAP image pose metadata readers; no pycolmap binary installed."""
from __future__ import annotations

import struct

import pytest

from spade_calib.io.colmap import load_registered_images, resolve_colmap_images


def test_colmap_images_bin_minimal(tmp_path):
    model=tmp_path/'sparse'/'0';model.mkdir(parents=True)
    f=model/'images.bin'
    packed=struct.pack('<Q',1)
    packed+=struct.pack('<i4d3di',11,1,0,0,0,1,2,3,2)
    packed+=b'folder/image.jpg\0'
    packed+=struct.pack('<Q',2)+b'\0'*48
    f.write_bytes(packed)
    poses=load_registered_images(model)
    assert poses['folder/image.jpg'].image_id==11
    assert poses['folder/image.jpg'].tvec==(1.,2.,3.)
    assert resolve_colmap_images(tmp_path)==f


def test_colmap_images_txt_and_csv(tmp_path):
    (tmp_path/'images.txt').write_text('# header\n'
        '2 1 0 0 0 1.2 0 0 1 image 002.jpg\n\n',encoding='utf-8')
    pose=load_registered_images(tmp_path)['image 002.jpg']
    assert pose.tvec==(1.2,0,0)
    (tmp_path/'camera_poses.csv').write_text('image_id,image_name,camera_id,qw,qx,qy,qz,tx,ty,tz\n'
         '1,a.jpg,1,1,0,0,0,0,0,0\n')
    assert 'a.jpg' in load_registered_images(tmp_path/'camera_poses.csv')


def test_multiple_sfm_models_must_be_selected(tmp_path):
    for n in ['0','1']:
        d=tmp_path/'sparse'/n;d.mkdir(parents=True)
        (d/'images.txt').write_text('1 1 0 0 0 0 0 0 1 a.jpg\n\n')
    with pytest.raises(ValueError,match='Multiple SfM reconstructions'):
        resolve_colmap_images(tmp_path)


def test_truncated_binary_rejected(tmp_path):
    p=tmp_path/'images.bin';p.write_bytes(struct.pack('<Q',1))
    with pytest.raises(ValueError,match='Truncated'):
        load_registered_images(p)
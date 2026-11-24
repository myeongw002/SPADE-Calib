import struct
import numpy as np
from scipy.spatial.transform import Rotation

from spade_calib.calibration.segment_geometry import (
    read_sparse_tracks, metric_local_points, SparseTrack, associate_patches,
    refine_shared_extrinsic,
)
from spade_calib.correspondence.sfm_lidar_planes import LidarPatch


def test_binary_text_tracks_and_local_metric(tmp_path):
    text = tmp_path / "points3D.txt"
    text.write_text("1 3 0 4 0 0 0 0.5 10 0 11 2\n")
    binary = tmp_path / "points3D.bin"
    binary.write_bytes(struct.pack("<Q",1)+struct.pack("<QdddBBBdQ",1,3.,0.,4.,0,0,0,.5,2)
                       +struct.pack("<iiii",10,0,11,2))
    centers = {10: np.array([1.,0.,0.]), 11: np.array([2.,0.,0.])}
    for path in (text,binary):
        tracks = list(read_sparse_tracks(path))
        assert tracks[0].image_ids == (10,11)
        p, ids = metric_local_points(tracks, centers, np.eye(3), centers[10], 2.)
        np.testing.assert_allclose(p, [[4.,0.,8.]])
        np.testing.assert_array_equal(ids, [1])
        p, _ = metric_local_points(tracks, {10: centers[10]}, np.eye(3), centers[10], 2.)
        assert p.shape == (0,3)


def patch(pid, xyz, normal):
    return LidarPatch(pid, normal, -float(xyz.mean(0)@normal), xyz, 0., np.array([0]))


def test_three_planes_shared_transform_and_rank():
    rng = np.random.default_rng(42)
    X = np.eye(4)
    X[:3,:3] = Rotation.from_rotvec([.02,-.03,.04]).as_matrix()
    X[:3,3] = [.11,-.07,.08]
    camera, lidar = [], []
    for k in range(3):
        n = np.eye(3)[k]
        pl = rng.uniform(-2,2,(180,3))
        pl[:,k] = 5.
        pc = pl @ X[:3,:3].T + X[:3,3]
        lidar.append(patch(k,pl,n))
        camera.append(patch(k,pc,X[:3,:3]@n))
    pairs = associate_patches(camera,lidar,np.eye(4),0)
    assert len(pairs)==3
    fit = refine_shared_extrinsic(pairs,np.eye(4))
    assert fit["success"]
    np.testing.assert_allclose(fit["T_camera_lidar"],X,atol=1e-6)
    assert refine_shared_extrinsic(pairs[:1],np.eye(4))["reason"]=="rank_deficient"


def test_far_parallel_planes_not_matched():
    rng = np.random.default_rng(1)
    p = rng.uniform(-1,1,(100,3)); p[:,2] = 3.
    q = p + [20.,0.,0.]
    assert not associate_patches([patch(0,p,np.array([0.,0.,1.]))],
                                 [patch(0,q,np.array([0.,0.,1.]))],np.eye(4),0)

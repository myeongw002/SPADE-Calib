import numpy as np
from spade_calib.evaluation.segment_projection import project_xyz, visible_pixels


def test_rectified_projection_uses_camera_transform_once():
    K = np.array([[100.,0,50],[0,100,40],[0,0,1]])
    X = np.eye(4); X[0,3]=.5
    p = np.array([[0.,0.,5.],[0,0,-1],[100,0,5],[0,0,100],[np.nan,0,5]])
    ids,uv,z = project_xyz(p,X,K,100,80)
    np.testing.assert_array_equal(ids,[0])
    np.testing.assert_allclose(uv,[[60.,40.]])
    np.testing.assert_allclose(z,[5.])


def test_zbuffer_prefers_nearest_depth():
    uv=np.array([[10.1,20.1],[10.9,20.9],[11.,20.]])
    ids=visible_pixels(uv,np.array([5.,2.,3.]),100)
    np.testing.assert_array_equal(ids,[1,2])

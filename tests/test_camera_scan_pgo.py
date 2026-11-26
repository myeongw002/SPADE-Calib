import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from spade_calib.calibration.camera_scan_pgo import pose,inverse,relative,edge_error,Graph,solve,Scan,register,RegistrationSettings


def test_conventions_and_gauge():
    R=Rotation.from_rotvec([[0,0,0],[.1,.2,.3],[.2,-.1,.4]]).as_matrix()
    C=pose(R,np.array([[0.,0,0],[1.,0,0],[1.,1,0]]))
    X=pose(R[1],np.array([.2,.3,.1]));Y=pose(R[2],np.array([-.1,.4,.5]))
    ij=np.array([[0,1],[1,2],[0,2]])
    A=relative(C@X,ij)
    np.testing.assert_allclose(A,inverse(X)@relative(C,ij)@X,atol=1e-12)
    np.testing.assert_allclose(relative(C@X@inverse(Y)@Y,ij),A,atol=1e-12)
    g=Graph(C,np.zeros(3,int),np.array([0]),ij,ij,A,np.tile(np.eye(6),(3,1,1)))
    out,cp=solve(g,X,pose_mode="translation",free_X=True)
    assert out["success"] and out["cost"]<1e-18
    np.testing.assert_allclose(cp,C,atol=1e-10)


def test_register_exact_three_planes():
    rng=np.random.default_rng(13)
    p=rng.uniform(-3,3,(900,3));n=np.repeat(np.eye(3),300,axis=0)
    for k in range(3):p[k*300:(k+1)*300,k]=3.
    T=pose(Rotation.from_rotvec([.008,-.006,.004]).as_matrix(),np.array([.04,-.02,.03]))
    q=(p-T[:3,3])@T[:3,:3];m=n@T[:3,:3]
    a=Scan(p,n,cKDTree(p));b=Scan(q,m,cKDTree(q))
    r=register(a,b,np.eye(4),RegistrationSettings())
    assert r["status"]=="ok" and r["rank"]==6
    np.testing.assert_allclose(r["T"],T,atol=1e-5)

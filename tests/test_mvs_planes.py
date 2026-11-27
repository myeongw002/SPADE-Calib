import numpy as np
from spade_calib.camera.mvs_planes import Settings,inliers,normal_rank,experiment

def test_sign_invariant_normal_gate():
    p=np.array([[0,1,4],[0,1,4],[0,1,4]])
    n=np.array([[0,1,0],[0,-1,0],[1,0,0]])
    cfg=Settings()
    assert inliers(p,n,np.array([0.,-1,0]),1.,cfg,False).all()
    assert inliers(p,n,np.array([0.,-1,0]),1.,cfg,True).tolist()==[True,True,False]

def test_rank():
    assert normal_rank([{"normal":[0,1,0]},{"normal":[1,0,0]}])["rank"]==2
    assert normal_rank([{"normal":v} for v in np.eye(3)])["rank"]==3

def test_ground_and_wall():
    x,z=np.meshgrid(np.linspace(-2,2,18),np.linspace(4,10,24))
    ground=np.column_stack([x.ravel(),np.full(x.size,1.5),z.ravel()])
    y,z=np.meshgrid(np.linspace(-2,1,16),np.linspace(4,10,24))
    wall=np.column_stack([np.full(y.size,4.),y.ravel(),z.ravel()])
    p=np.concatenate([ground,wall]);n=np.repeat([[0.,-1,0],[1.,0,0]],[len(ground),len(wall)],axis=0)
    g,arms,ids,_=experiment(p,n,Settings(iterations=128,min_points=40,max_planes=2))
    assert g["G1"] is not None
    assert len(ids)==len(wall)
    assert len(arms["P1"])==1
    assert abs(arms["P1"][0][1]["normal"][0])>.999

#!/usr/bin/env python3
"""MVS3 pilot: fixed-GT raw64 reference; MVS2 multi-segment normal audit.
Dependencies: numpy, scipy. No ICP, pose fitting, or scale fitting.
Candidate geometry is not physical correspondence certification.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def mat(x):
    m=np.eye(4); m[:3]=np.asarray(x).reshape(3,4); return m

def gt_inputs(calib_path, poses_path):
    c={k.strip():np.fromstring(v,sep=' ') for k,v in (l.split(':',1) for l in Path(calib_path).read_text().splitlines() if ':' in l)}
    poses=np.loadtxt(poses_path,ndmin=2)
    for k in ('P0','P2','Tr'):
        if k not in c or c[k].size!=12 or not np.isfinite(c[k]).all():
            raise ValueError(f'Invalid or missing KITTI calibration entry: {k}')
    if poses.shape[1]!=12 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite KITTI N x 12 GT poses')
    P0=c['P0'].reshape(3,4); P=c['P2'].reshape(3,4)
    if not np.allclose(P0[:,3],0,atol=1e-8):
        raise ValueError('Expected KITTI rectified camera-0 projection with zero translation')
    C=np.eye(4); C[:3,3]=np.linalg.solve(P[:,:3],P[:,3]); X=C@mat(c['Tr'])
    return C,X,np.stack([mat(p) for p in poses])

def transform(p,T): return p@T[:3,:3].T+T[:3,3]
def voxel(p,h=.10):
    if not len(p): return p.reshape(-1,3)
    _,i=np.unique(np.floor(p/h).astype(np.int64),axis=0,return_index=True); return p[np.sort(i)]

def cloud(scan_dir, frames, anchor, C, X, poses):
    out=[]
    for f in frames:
        raw=np.fromfile(scan_dir/f'{f:06d}.bin',dtype='<f4')
        if not len(raw) or raw.size%4: raise ValueError(f'Invalid raw scan {f}')
        p=raw.reshape(-1,4)[:,:3].astype(float); r=np.linalg.norm(p,axis=1)
        p=p[np.isfinite(p).all(1)&(r>=1)&(r<=80)]
        T=C@np.linalg.inv(poses[anchor])@poses[f]@np.linalg.inv(C)@X
        p=transform(p,T); out.append(p[np.linalg.norm(p,axis=1)<=67])
    return voxel(np.concatenate(out)) if out else np.empty((0,3))

def load_planes(folder):
    s=read(folder/'summary.json'); a=np.load(folder/'sampled.npz'); p=a['xyz_m']; result=[]
    if s['ground']['G1']['status']=='ok': result.append(('G1',s['ground']['G1'],p[np.load(folder/'G1_indices.npy')]))
    if 'P1' in s['arms']:
        ids=np.load(folder/'P1'/'indices.npz')
        for row in s['arms']['P1']['planes']:
            key=f"plane_{row['patch_id']:02d}"; result.append(('P1/'+key,row,p[ids[key]]))
    return s,result

def metrics(n):
    n=np.asarray(n,float).reshape(-1,3)
    if not len(n): return {'count':0,'singular_values':[0.,0.,0.],'rank':0,'weak_direction':None,'condition':None}
    n=n/np.linalg.norm(n,axis=1,keepdims=True); _,s,v=np.linalg.svd(n/np.sqrt(len(n)),full_matrices=True); s=np.pad(s,(0,3-len(s)))
    rank=int((s>max(1e-10,.05*s[0])).sum())
    return {'count':len(n),'singular_values':s.tolist(),'rank':rank,'weak_direction':v[-1].tolist(),'condition':float(s[0]/s[-1]) if rank==3 else None}

def fit(p, nref):
    c=p.mean(0); _,v=np.linalg.eigh((p-c).T@(p-c)/len(p)); n=v[:,0]
    if n@nref<0: n=-n
    return n,float(-n@c)

def stat(x):
    if not len(x): return None
    return {'count':len(x),'signed_median_m':float(np.median(x)),'abs_p50_p90_m':np.percentile(np.abs(x),[50,90]).tolist(),'rmse_m':float(np.sqrt(np.mean(x*x)))}

def audit(p, train, test, n, d, seed=20261127, slab=1.5, footprint=.35, tol=.12, min_points=40):
    """Use finite tangent support, then fit reference candidates without forcing MVS offset.
    Search direction +/-20 deg and slab +/-1.5 m are association priors, not truth.
    Holdout includes all points in fixed tangent footprint/slab (clutter possible).
    """
    n=np.asarray(n,float); n/=np.linalg.norm(n); c=p.mean(0)
    _,_,v=np.linalg.svd(n[None,:],full_matrices=True); B=v[1:].T; tree=cKDTree((p-c)@B)
    def roi(q):
        q=q[np.abs(q@n+d)<=slab]; return q[tree.query((q-c)@B,workers=1)[0]<=footprint]
    q=roi(train); h=roi(test); result={'mvs_centroid_m':c.tolist(),'mvs_normal':n.tolist(),'mvs_offset_m':d,'roi_train':len(q),'roi_holdout':len(h),'raw_holdout_vs_mvs':stat(h@n+d),'candidates':[],'status':'insufficient_support'}
    if len(q)<min_points: return result,q,h
    rng=np.random.default_rng(seed); remaining=np.arange(len(q)); cos=np.cos(np.deg2rad(20))
    for _ in range(3):
        if len(remaining)<min_points: break
        r=q[remaining]; best=None
        for _ in range(512):
            tri=r[rng.choice(len(r),3,replace=False)]; m=np.cross(tri[1]-tri[0],tri[2]-tri[0]); mag=np.linalg.norm(m)
            if mag<1e-9: continue
            m/=mag
            if abs(m@n)<cos: continue
            b=-float(m@tri[0]); ids=np.flatnonzero(abs(r@m+b)<=tol)
            if best is None or len(ids)>len(best): best=ids
        if best is None or len(best)<min_points: break
        for _ in range(5):
            m,b=fit(r[best],n); keep=best[np.abs(r[best]@m+b)<=tol]
            if len(keep)==len(best): break
            best=keep
            if len(best)<min_points: break
        if len(best)<min_points: break
        m,b=fit(r[best],n); angle=float(np.degrees(np.arccos(np.clip(m@n,-1,1))))
        if angle>20: break
        support=r[best]; uv=(support-c)@B; ex=np.percentile(uv,95,axis=0)-np.percentile(uv,5,axis=0)
        # Overlap is fraction of MVS samples within 0.35m tangential distance of fitted LiDAR support.
        overlap=float(np.mean(cKDTree(uv).query((p-c)@B)[0]<=footprint))
        # Held-out points assigned by reference plane only; report retained fraction and unconditional ROI stats above.
        hr=h@m+b; hv=h[np.abs(hr)<=tol]
        result['candidates'].append({'train_support':len(best),'train_roi_fraction':len(best)/len(q),'lidar_normal':m.tolist(),'lidar_offset_m':b,'normal_angle_deg':angle,'mvs_centroid_signed_to_lidar_m':float(c@m+b),'offset_difference_lidar_minus_mvs_m':float(b-d),'mvs_points_vs_lidar_plane':stat(p@m+b),'lidar_train_fit':stat(support@m+b),'tangent_extent_m':ex.tolist(),'mvs_tangent_coverage':overlap,'holdout_support':len(hv),'holdout_fraction':len(hv)/len(h) if len(h) else None,'lidar_holdout_vs_mvs_conditional':stat(hv@n+d)})
        remaining=np.delete(remaining,best)
    result['status']='review_required' if result['candidates'] else 'no_candidate'
    return result,q,h

def save_ply(path, groups):
    rows=[]
    for p,col in groups:
        rows.append(np.column_stack([p,np.tile(col,(len(p),1))]))
    a=np.concatenate(rows); header=f'ply\nformat ascii 1.0\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with path.open('w') as f: f.write(header); np.savetxt(f,a,fmt=['%.5f']*3+['%d']*3)

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('diversity'); a.add_argument('--planes',type=Path,nargs='+',required=True); a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('validate'); a.add_argument('--planes',type=Path,required=True); a.add_argument('--report',type=Path,required=True,help='MVS1 segment report.json'); a.add_argument('--calib',type=Path,required=True,help='KITTI sequence calib.txt'); a.add_argument('--gt-poses',type=Path,required=True,help='KITTI camera-0 GT poses txt'); a.add_argument('--lidar-dir',type=Path,required=True); a.add_argument('--sequence',default='07'); a.add_argument('--stride',type=int,default=5); a.add_argument('--extra-z-npz',type=Path,help='Optional MVS2-D candidate.npz for this segment only'); a.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='diversity':
        alln=[]; non=[]; rows=[]
        for folder in args.planes:
            _,ps=load_planes(folder); ns=[r['normal'] for _,r,_ in ps]; alln+=ns; non.extend(r['normal'] for k,r,_ in ps if k!='G1')
            rows.append({'folder':str(folder),'all':metrics(ns),'nonground':metrics([r['normal'] for k,r,_ in ps if k!='G1']),'planes':[{'id':k,'normal':r['normal'],'centroid_m':r['centroid_m'],'rmse_m':r['rmse_m']} for k,r,_ in ps]})
        result={'segments':rows,'pooled':metrics(alln),'pooled_nonground':metrics(non),'warning':'Same local sensor-axis parameterization. Equal plane weights; correct matches and fixed R/scale assumed; duplicates and unverified surfaces may inflate rank.'}
    else:
        if args.stride<1: ap.error('stride must be >=1')
        s,ps=load_planes(args.planes); rep=read(args.report); anchor=int(rep['anchor_frame']); frames=sorted({int(Path(x).stem) for x in rep['selected_images']});
        if anchor not in frames: raise ValueError('Anchor not in MVS selected images')
        segment_dir=Path(s["input_ply"]).parts[-3]
        if args.report.parent.name.startswith("segment_") and args.report.parent.name!=segment_dir: raise ValueError("MVS1 report and MVS2 input segment mismatch")
        if args.extra_z_npz:
            z=np.load(args.extra_z_npz); sample=np.load(args.planes/'sampled.npz')['xyz_m']; ids=z['sample_indices']
            if not np.allclose(sample[ids],z['xyz_m'],rtol=0,atol=1e-8): raise ValueError('Extra candidate does not match current MVS2 sample')
            ps.append(('Z_diagnostic',{'normal':z['normal'].tolist(),'offset_m':float(z['offset_m'])},z['xyz_m']))
        C,X,poses=gt_inputs(args.calib,args.gt_poses)
        if max(frames)>=len(poses): raise ValueError('Frame exceeds GT trajectory')
        chosen=sorted(set(frames[::args.stride]+[anchor,frames[-1]])); other=[f for f in chosen if f!=anchor]; train_ids=other[::2]; test_ids=other[1::2]
        if not train_ids or not test_ids: raise ValueError('Need nonempty disjoint reference frame groups')
        raw=cloud(args.lidar_dir,[anchor],anchor,C,X,poses); tr=cloud(args.lidar_dir,train_ids,anchor,C,X,poses); te=cloud(args.lidar_dir,test_ids,anchor,C,X,poses)
        rows=[]
        for k,r,p in ps:
            acc,q,h=audit(p,tr,te,r['normal'],r['offset_m']); single,qr,_=audit(p,raw,np.empty((0,3)),r['normal'],r['offset_m']); rows.append({'plane':k,'gt_accumulated':acc,'anchor_raw64':single}); save_ply(args.out/(k.replace('/','_')+'_overlay.ply'),[(p,[255,170,0]),(q,[0,180,255]),(h,[0,230,70]),(qr,[180,0,255])])
            print(k,acc['status'],len(acc['candidates']),flush=True)
        result={'sequence':args.sequence,'anchor':anchor,'calib_path':str(args.calib.resolve()),'gt_poses_path':str(args.gt_poses.resolve()),'gt_X_camera2_lidar':X.tolist(),'train_frames':train_ids,'holdout_frames':test_ids,'reference_counts':{'anchor':len(raw),'train':len(tr),'holdout':len(te)},'planes':rows,'warning':'GT evaluation only. Raw64 reference uses GT poses, rigid scans, no deskew; dynamic/clutter and camera pose/scale errors remain. Plane candidates require physical review; conditional holdout statistics are selection-biased. No extrinsic fitting.'}
    write(args.out/'summary.json',result); print(args.out/'summary.json')
if __name__=='__main__': main()

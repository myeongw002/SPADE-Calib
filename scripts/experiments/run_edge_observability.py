#!/usr/bin/env python3
"""E0 pilot: frozen L0/C0 candidates, B4-c initialization; optional raw DA2.
No calibration fit. No GT selection. Read existing outputs/ directories by default; ZIPs remain optional.
"""
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

class Archive:
    def __init__(self,path,marker,root_name):
        self.path=Path(path); self.z=None; self.prefix=''
        if self.path.is_dir():
            self.require(marker)
        elif self.path.is_file():
            if not zipfile.is_zipfile(self.path):
                raise ValueError(f'{self.path}: expected an output directory or ZIP archive')
            self.z=zipfile.ZipFile(self.path)
            # Accept a flat archive or an archive containing the known run root.
            # Nested per-frame summary.json files must not be mistaken for the root.
            hits=[n for n in self.z.namelist() if n==marker or
                  n==f'{root_name}/{marker}' or n.endswith(f'/{root_name}/{marker}')]
            if len(hits)!=1:
                self.z.close()
                raise ValueError(f'{self.path}: expected exactly one {root_name}/{marker} or root {marker}; found {hits}')
            self.prefix=hits[0][:-len(marker)]
        else:
            raise FileNotFoundError(f'Missing input: {self.path} (expected existing output directory with {marker}, or explicit ZIP)')
    def require(self,name):
        if self.z:
            if self.prefix+name not in self.z.namelist():
                raise FileNotFoundError(f'Missing archive member: {self.path}!/{self.prefix+name}')
        elif not (self.path/name).is_file():
            raise FileNotFoundError(f'Missing input file: {self.path/name}')
    def read(self,name):
        self.require(name)
        return self.z.read(self.prefix+name) if self.z else (self.path/name).read_bytes()
    def json(self,name): return json.loads(self.read(name))
    def npz(self,name): return dict(np.load(io.BytesIO(self.read(name)),allow_pickle=False))
    def image(self,name):
        a=cv2.imdecode(np.frombuffer(self.read(name),np.uint8),cv2.IMREAD_COLOR)
        if a is None: raise ValueError(name)
        return a

def sha(b):return hashlib.sha256(b).hexdigest()
def line_normals(s,K):
    p=np.concatenate([s,np.ones((*s.shape[:2],1))],axis=2); l=np.cross(p[:,0],p[:,1]); l/=np.linalg.norm(l[:,:2],axis=1)[:,None]
    n=l@K; n/=np.linalg.norm(n,axis=1)[:,None]
    return l,n

def schur(H):
    # Parameters [rot xyz, t xyz], desired [tx,tz], nuisance [rot xyz,ty].
    a=[3,5];b=[0,1,2,4]; S=H[np.ix_(a,a)]-H[np.ix_(a,b)]@np.linalg.pinv(H[np.ix_(b,b)],rcond=1e-10)@H[np.ix_(b,a)]
    return (S+S.T)/2

def metrics(H):
    S=schur(H);return dict(xz_information=S.tolist(),xz_eigenvalues=np.linalg.eigvalsh(S).tolist(),full_eigenvalues=np.linalg.eigvalsh(H).tolist())

def greedy(M,k,weights=None,H=None):
    # E-optimal set selection (image-only proxy, no 3D/covariance claim).
    H=np.zeros((2,2)) if H is None else H.copy(); w=np.ones(len(M)) if weights is None else weights
    out=[];remaining=np.arange(len(M))
    for _ in range(min(k,len(M))):
        T=H[None]+M[remaining]*w[remaining,None,None]
        eig=np.linalg.eigvalsh(T); j=np.lexsort((remaining,-eig[:,1],-eig[:,0]))[0]
        i=int(remaining[j]);out.append(i);H=T[j];remaining=np.delete(remaining,j)
    return np.array(out,int)

def project(q,K,shape):
    q=q.copy(); h,w=shape
    if np.all(q[:,2]<.5):return None
    for i in range(2):
        if q[i,2]<.5:q[i]+=(.5-q[i,2])/(q[1-i,2]-q[i,2])*(q[1-i]-q[i])
    uv=(q@K.T)[:,:2]/q[:,2,None];a,b=uv;d=b-a;lo,hi=0.,1.
    for p,r in ((-d[0],a[0]),(d[0],w-1-a[0]),(-d[1],a[1]),(d[1],h-1-a[1])):
        if abs(p)<1e-12:
            if r<0:return None
        elif p<0:lo=max(lo,r/p)
        else:hi=min(hi,r/p)
        if lo>hi:return None
    if np.linalg.norm(d)*(hi-lo)<15:return None
    # image clipping corresponds to perspective-correct 3D interpolants.
    def at(v):
        t=v*q[0,2]/((1-v)*q[1,2]+v*q[0,2]);return q[0]+t*(q[1]-q[0])
    return np.stack([a+lo*d,a+hi*d]),np.stack([at(lo),at(hi)])

def match(p,s):
    v=p[1]-p[0];L=np.linalg.norm(v);v/=L
    dv=s[:,1]-s[:,0];length=np.linalg.norm(dv,axis=1);u=dv/length[:,None];normal=np.column_stack([-u[:,1],u[:,0]])
    dist=np.mean(abs(np.einsum('nki,ni->nk',p[None]-s[:,0,None],normal)),axis=1)
    ang=np.rad2deg(np.arccos(np.clip(abs(u@v),0,1)));iv=(s-p[0])@v
    overlap=np.maximum(0,np.minimum(L,iv.max(1))-np.maximum(0,iv.min(1)))/L
    cost=(dist/20)**2+(ang/10)**2+(1-overlap)**2
    ids=np.flatnonzero((dist<=20)&(ang<=10)&(overlap>=.3));ids=ids[np.lexsort((ids,cost[ids]))]
    if not len(ids):return 'no_candidate',None,None
    i=ids[0]; detail=dict(distance_px=float(dist[i]),angle_deg=float(ang[i]),overlap=float(overlap[i]),cost=float(cost[i]),gap=float(cost[ids[1]]-cost[i]) if len(ids)>1 else None)
    if len(ids)>1 and cost[ids[1]]-cost[i]<.1:return 'ambiguous',None,detail
    return 'proposed',int(i),detail

def depth_scores(D,s):
    """Affine-invariant relative inverse-depth cues, not metric normals.
    Fit opposite-side profiles separately (offsets 3..9px); jump is
    extrapolated intercept difference; crease is slope difference.
    Normalize by robust local range with global noise floor. Keep RGB location.
    """
    D=np.asarray(D,np.float32);h,w=D.shape;out=[]
    scale=float(np.percentile(D,95)-np.percentile(D,5))
    if not np.isfinite(D).all() or scale<=1e-8:raise ValueError('invalid depth')
    offsets=np.arange(3,10,dtype=float)
    A=np.column_stack([np.ones(len(offsets)),offsets]);P=np.linalg.pinv(A)
    for p in s:
        v=p[1]-p[0];v/=np.linalg.norm(v);n=np.array([-v[1],v[0]])
        centers=p[0]+np.linspace(.1,.9,17)[:,None]*(p[1]-p[0]);pos=centers[:,None]+offsets[None,:,None]*n;neg=centers[:,None]-offsets[None,:,None]*n
        ok=np.all((pos[...,0]>=0)&(pos[...,0]<=w-1)&(pos[...,1]>=0)&(pos[...,1]<=h-1)&(neg[...,0]>=0)&(neg[...,0]<=w-1)&(neg[...,1]>=0)&(neg[...,1]<=h-1),axis=1)
        if ok.sum()<5:out.append([0,0,0,0]);continue
        def sample(x):return cv2.remap(D,x[...,0].astype('float32'),x[...,1].astype('float32'),cv2.INTER_LINEAR)
        a,b=sample(pos[ok]),sample(neg[ok]);ca=a@P.T;cb=b@P.T
        rng=np.maximum(np.ptp(np.column_stack([a,b]),axis=1),.03*scale)
        jump=(ca[:,0]-cb[:,0])/rng;crease=6*(ca[:,1]+cb[:,1])/rng
        # sign coherence avoids rewarding random oscillations along the line.
        j=abs(np.median(jump));c=abs(np.median(crease));q=min(1.,max(j,c))*float(ok.mean())
        out.append([j,c,q,float(ok.mean())])
    return np.array(out)

def jac(q,l,K,X):
    # pixel point-to-line residual, same physical scale units as B4; rotation left, t additive.
    r=q-X[:3,3];z=q[:,2];v=l@K
    g=v[None]/z[:,None]-(q@v/z**2)[:,None]*np.array([0,0,1.])
    return np.column_stack([np.cross(r,g),g])*np.r_[np.full(3,np.deg2rad(1)),np.full(3,.1)]/2.

def prior(ground,motion,X):
    m=motion; s=m['reference_scales'][m['segment_ids']];R0=X[:3,:3];t0=X[:3,3];N=len(s)
    assert np.allclose(ground['baseline']['T_camera_lidar'],m['X_he']) and np.allclose(ground['baseline']['scales'],m['reference_scales'])
    rows=[g for g in ground['ground_diagnostics'] if g['accepted']];nl=np.array([g['lidar']['normal'] for g in rows]);dc=np.array([g['camera']['offset_m'] for g in rows]);dl=np.array([g['lidar']['offset_m'] for g in rows]);units=np.r_[np.full(3,np.deg2rad(1)),np.full(3,.1)]
    def f(x):
        R=Rotation.from_rotvec(x[:3]*units[:3]).as_matrix()@R0;t=t0+x[3:]*units[3:]
        h=np.column_stack([Rotation.from_matrix((m['Rb']@R).transpose(0,2,1)@(R@m['Ra'])).as_rotvec()/np.deg2rad(.5),((m['Rb']-np.eye(3))@t+s[:,None]*m['tb']-m['ta']@R.T)/.05]);g=(dc-dl+(nl@R.T)@t)/.15
        return np.r_[h.ravel()/np.sqrt(N),g*np.sqrt(.1/len(g))]
    # Validate exact robust objective against B4 archive before using raw information.
    r=f(np.zeros(6));h=r[:6*N]*np.sqrt(N);g=r[6*N:]/np.sqrt(.1/len(rows));cost=np.sum(np.hypot(1,h)-1)/N+.1*np.mean(np.hypot(1,g)-1)
    assert np.isclose(cost,ground['experiments']['B4c_w0.1']['final']['total_cost'],rtol=1e-7,atol=1e-9),(cost,ground['experiments']['B4c_w0.1']['final']['total_cost'])
    J=np.column_stack([(f(e)-f(-e))/2e-5 for e in np.eye(6)*1e-5]);return J.T@J,float(cost)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'m0':'outputs/lines07_M0_ghost','c0':'outputs/lines07_C0',
              'ground':'outputs/ground07_B4c','motion':'outputs/trajectory_oracle07'}
    for name,path in defaults.items():
        p.add_argument('--'+name,type=Path,default=Path(path),help=f'Existing output directory or ZIP (default: {path})')
    p.add_argument('--out',type=Path,required=True,help='Fresh output directory')
    p.add_argument('--budget',type=int,default=10);p.add_argument('--depth-dir',type=Path);p.add_argument('--da2-repo',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--encoder',default='vitl');p.add_argument('--device',default='auto');p.add_argument('--image-dir',type=Path);a=p.parse_args()
    if a.budget<1 or (a.depth_dir and a.checkpoint) or (a.checkpoint and (not a.da2_repo or not a.image_dir)):p.error('invalid depth/budget arguments')
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError(a.out)
    m0=Archive(a.m0,'manifest.json','lines07_M0_ghost')
    c0=Archive(a.c0,'summary.json','lines07_C0')
    ground=Archive(a.ground,'summary.json','ground07_B4c')
    motion=Archive(a.motion,'motion_pairs.npz','trajectory_oracle07')
    manifest=m0.json('manifest.json');ann=m0.json('annotations.json');cfg=c0.json('summary.json');g=ground.json('summary.json');m=motion.npz('motion_pairs.npz')
    assert sha(m0.read('manifest.json'))==cfg['manifest_sha256']==ann['manifest_sha256'] and sha(m0.read('annotations.json'))==cfg['annotations_sha256']
    K=np.array(manifest['K']);X=np.array(g['experiments']['B4c_w0.1']['final']['T_camera_lidar']); H,cost=prior(g,m,X)
    # Check every required frame input before creating any result files.
    for f in sorted(map(int,cfg['frames'])):
        for name in ('elsed_segments.npz','comparison.png','global_matches.json'):
            c0.require(f'frame_{f:06d}/{name}')
        if a.image_dir and not (a.image_dir/f'{f:06d}.png').is_file():
            raise FileNotFoundError(a.image_dir/f'{f:06d}.png')
        if a.depth_dir and not (a.depth_dir/f'{f:06d}.npy').is_file():
            raise FileNotFoundError(a.depth_dir/f'{f:06d}.npy')
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[];reviews=[];frames={};sums=defaultdict(list);post_sums=defaultdict(list);manual=[];Dmode=bool(a.depth_dir or a.checkpoint)
    for f in sorted(map(int,cfg['frames'])):
        path=f'frame_{f:06d}/';arr=c0.npz(path+'elsed_segments.npz');s0=arr['segments'].reshape(-1,2,2);length0=np.linalg.norm(s0[:,1]-s0[:,0],axis=1);ids=np.flatnonzero(length0>=40);s=s0[ids];length=length0[ids];l,n=line_normals(s,K);M=n[:,[0,2],None]*n[:,None,[0,2]]
        # C0 comparison contains 6 panels; first 25 rows header, then untouched original.
        panel=c0.image(path+'comparison.png');h=panel.shape[0]//6-25;w=panel.shape[1];img=panel[25:25+h].copy()
        if h<2 or panel.shape[0]!=6*(h+25):raise ValueError('Unexpected C0 panel layout')
        if a.image_dir:
            original=cv2.imread(str(a.image_dir/f'{f:06d}.png'))
            if original is None or not np.array_equal(original,img):raise ValueError(f'Image mismatch {f}')
        if Dmode:
            if a.checkpoint:
                from spade_calib.camera.depth_inference import infer_da2
                D,meta=infer_da2(a.image_dir/f'{f:06d}.png',a.checkpoint,encoder=a.encoder,device=a.device,da2_repo=a.da2_repo)
                np.save(a.out/f'{f:06d}_depth.npy',D);(a.out/f'{f:06d}_depth.json').write_text(json.dumps(meta,indent=2))
            else:D=np.load(a.depth_dir/f'{f:06d}.npy',allow_pickle=False)
            if D.shape!=(h,w):raise ValueError(f'Depth shape mismatch {f}')
            ds=depth_scores(D,s)
        else:ds=np.zeros((len(s),4))
        choices={'length':np.argsort(-length,kind='stable')[:a.budget],'random':np.random.default_rng(17+f).permutation(len(s))[:a.budget],'normal':greedy(M,a.budget)}
        if Dmode:choices.update(depth=np.argsort(-ds[:,2],kind='stable')[:a.budget],fusion=greedy(M,a.budget,.25+.75*ds[:,2]))
        recs=c0.json(path+'global_matches.json'); proposals=defaultdict(list);counts=Counter()
        for rec in recs:
            xyz=np.array(rec['xyz_lidar']);q=xyz@X[:3,:3].T+X[:3,3];pr=project(q,K,(h,w))
            if pr is None:counts['not_projectable']+=1;continue
            uv,qclip=pr;status,i,detail=match(uv,s);counts[status]+=1
            if i is None:continue
            # One image line -> one selected LiDAR candidate, other fragments remain unverified.
            proposals[i].append((detail['cost'],int(rec['id']),xyz,uv,qclip,detail))
        best={i:sorted(v,key=lambda x:(x[0],x[1]))[0] for i,v in proposals.items()}
        # Reject competing 3D candidates instead of silently picking a physical match.
        unique={i:v for i,v in best.items() if len(proposals[i])==1}
        fi={'eligible':len(s),'status':dict(counts),'distinct_proposed':len(best),'unique_proposed':len(unique),'arms':{},'postmatch':{}}
        # Compare equal budgets on the SAME unique proposal pool. All physical
        # matches remain unverified. Add each local block to the fixed B4 prior.
        pool=np.array(sorted(unique),int);blocks=[]
        for i in pool:
            _,lid,xyz,uv,qclip,detail=unique[i];J=jac(np.linspace(qclip[0],qclip[1],5),l[i],K,X);blocks.append(J.T@J/len(J));reviews.append((f,int(ids[i]),lid,s[i],uv,xyz))
            rows.append(dict(frame=f,arm='all_proposals',image_id=int(ids[i]),lidar_id=lid,nx=n[i,0],ny=n[i,1],nz=n[i,2],**detail))
        if len(pool):
            B=np.array(blocks);k=min(2,len(pool));chosen=[];remaining=list(range(len(pool)));G=H.copy()
            for _ in range(k):
                # Per-frame ranking, provisional lambda=.1; fixed normalization
                # within each frame's comparison, not calibrated covariance.
                j=max(remaining,key=lambda j:(np.linalg.eigvalsh(schur(G+.1*B[j]/k))[0],-j));chosen.append(j);remaining.remove(j);G+=.1*B[j]/k
            post={'length':np.argsort(-length[pool],kind='stable')[:k],'random':np.random.default_rng(37+f).permutation(len(pool))[:k],'normal':greedy(M[pool],k),'full_jacobian':np.array(chosen)}
            if Dmode:post['fusion']=greedy(M[pool],k,.25+.75*ds[pool,2])
            for arm,idx in post.items():
                post_sums[arm].extend(B[idx]);fi['postmatch'][arm]=dict(selected_ids=ids[pool[idx]].tolist(),count=len(idx))
        # Three historical manual lines are a sparse, selected diagnostic only.
        for ref in ann['records']:
            if int(ref['frame'])!=f or not ref.get('manual_identity_confirmed'):continue
            xyz=np.array(ref['xyz_lidar']);uvref=np.array(ref['uv_image']);q=xyz@X[:3,:3].T+X[:3,3];pr=project(q,K,(h,w))
            if pr is None:continue
            v=uvref[1]-uvref[0];v/=np.linalg.norm(v);nn=np.array([-v[1],v[0]]);err=float(np.mean(abs((pr[0]-uvref[0])@nn)))
            status,i,detail=match(pr[0],s)
            manual.append(dict(frame=f,id=ref['id'],manual_perp_px_B4c=err,match_status=status,selected_image_id=int(ids[i]) if i is not None else None))
        for arm,choice in choices.items():
            mats=[];overlay=img.copy();sel=set(map(int,choice));S=M[choice].sum(0)/max(1,len(choice));matched=sel&set(unique)
            for i in choice:
                col=(0,255,0) if i in matched else (0,180,255);cv2.line(overlay,*np.rint(s[i]).astype(int),col,2,cv2.LINE_AA)
            for i in sorted(matched):
                _,lid,xyz,uv,qclip,detail=unique[i];q=np.linspace(qclip[0],qclip[1],5);J=jac(q,l[i],K,X);mats.append(J.T@J/len(J));cv2.line(overlay,*np.rint(uv).astype(int),(0,0,255),1,cv2.LINE_AA)
                rows.append(dict(frame=f,arm=arm,image_id=int(ids[i]),lidar_id=lid,nx=n[i,0],ny=n[i,1],nz=n[i,2],**detail))
                reviews.append((f,int(ids[i]),lid,s[i],uv,xyz))
            cv2.imwrite(str(a.out/f'{f:06d}_{arm}.png'),overlay)
            fi['arms'][arm]=dict(selected=len(choice),unique_proposed=len(matched),proxy_eigenvalues=np.linalg.eigvalsh(S).tolist(),selected_ids=ids[choice].tolist());sums[arm].extend(mats)
        for i in range(len(s)):
            rows.append(dict(frame=f,arm='all_image',image_id=int(ids[i]),lidar_id='',nx=n[i,0],ny=n[i,1],nz=n[i,2],length_px=float(length[i]),depth_jump=float(ds[i,0]) if Dmode else '',depth_crease=float(ds[i,1]) if Dmode else '',depth_support=float(ds[i,2]) if Dmode else ''))
        frames[str(f)]=fi
    # Equal mass per matched line, no duplicate image IDs within arm. Not covariance.
    summary={'warning':'Image-only proxy ignores depth/rotation; all matched information conditional on UNVERIFIED proposals. No calibration fit. Depth proxy not a physical normal. C0 cloud candidates generated in previous H2-era archive, no H2 transform used for rematching. Raw B4 prior fixed scales, no statistical independence claim.','baseline_cost_verified':cost,'K':K.tolist(),'X_B4c':X.tolist(),'depth_enabled':Dmode,'depth_external_provenance_unverified':bool(a.depth_dir),'budget':a.budget,'baseline_information':metrics(H),'frames':frames,'arms':{}}
    for arm,v in sums.items():
        count=len(v);S=np.sum(v,axis=0)/max(1,count) if count else np.zeros((6,6));summary['arms'][arm]={'unique_proposals':count,'selected':sum(x['arms'][arm]['selected'] for x in frames.values()),'mean_proxy_min_eigenvalue':float(np.mean([x['arms'][arm]['proxy_eigenvalues'][0] for x in frames.values()])), 'conditional_information':{str(w):metrics(H+w*S) for w in (.01,.1,1.)}}
    for arm in choices:
        if arm not in summary['arms']:summary['arms'][arm]={'unique_proposals':0,'selected':sum(x['arms'][arm]['selected'] for x in frames.values()),'mean_proxy_min_eigenvalue':float(np.mean([x['arms'][arm]['proxy_eigenvalues'][0] for x in frames.values()])), 'conditional_information':{str(w):metrics(H) for w in (.01,.1,1.)}}
    summary['postmatch_arms']={arm:dict(count=len(v),conditional_information={str(w):metrics(H+w*np.mean(v,axis=0)) for w in (.01,.1,1.)}) for arm,v in post_sums.items()}
    summary['manual_reference_diagnostic']=manual
    # De-duplicated review rows; no automatic correctness label.
    unique_review={(f,i,j):(s,u,x) for f,i,j,s,u,x in reviews};rd=a.out/'review';rd.mkdir()
    with (a.out/'review.csv').open('w',newline='') as fp:
        writer=csv.writer(fp);writer.writerow(['frame','image_id','lidar_id','label','notes'])
        for (f,i,j),(s,u,x) in sorted(unique_review.items()):
            panel=c0.image(f'frame_{f:06d}/comparison.png');h=panel.shape[0]//6-25;im=panel[25:25+h].copy();cv2.line(im,*np.rint(s).astype(int),(0,255,0),2);cv2.line(im,*np.rint(u).astype(int),(0,0,255),2);cv2.imwrite(str(rd/f'{f:06d}_{i}_{j}.png'),im);writer.writerow([f,i,j,'',''])
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (a.out/'features.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary['arms'],indent=2))
if __name__=='__main__':main()

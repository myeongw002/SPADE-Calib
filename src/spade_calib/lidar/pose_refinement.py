"""Historical D1 B/C1 scan-level robust joint refinement (not legacy Joint/PGO).

The numerical estimator below preserves the reference D1 B/C1 thresholds and
anchor scan 5. It is NOT an IMU-based continuous-time solver. Requires scipy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix, vstack as svstack, diags
from scipy.sparse.linalg import lsmr

SEED = 20261106

def vox_sample(p,voxel=.14):
    key=np.floor(p/voxel).astype('int32')
    _,ind=np.unique(key,axis=0,return_index=True)
    return np.sort(ind)

def normals_for_scan(p):
    # PCA only to provide targets normals, not final plane extraction
    tree=cKDTree(p)
    dist,near=tree.query(p,k=24,workers=-1)
    q=p[near]
    cen=q.mean(axis=1);arr=q-cen[:,None,:]
    cv=np.einsum('nki,nkj->nij',arr,arr)/24
    eig,vec=np.linalg.eigh(cv)
    ratio=eig[:,0]/np.maximum(eig[:,1],1.e-12)
    normal=vec[:,:,0]
    ok=(ratio<.28)&(dist[:,-1]<1.6)&(eig[:,1]>.0003)
    return normal,ok

def ptransform(p,x,c):
    return (p-c)@Rotation.from_rotvec(x[3:]).as_matrix().T+c+x[:3]

def sample_model(A,scan,exclude_wall=True):
    # Conservative ROI and exclude held-out wall in training; keeps ground+other building
    idx=[];centers=[];nset=[];normset=[];okset=[]
    for s in range(10):
        original=np.flatnonzero(scan==s)
        p=A[original]
        mask=(np.abs(p[:,0])<22)&(np.abs(p[:,1])<16)&(p[:,2]>-2)&(p[:,2]<5)
        if exclude_wall:
            wall=(p[:,0]>10.3)&(p[:,0]<14.9)&(p[:,1]>4.3)&(p[:,1]<9.4)&(p[:,2]>-1.2)&(p[:,2]<4.6)
            mask &= ~wall
        valid=original[mask];sel=vox_sample(A[valid],.14);sel=valid[sel]
        if len(sel)>6000:
            rng=np.random.default_rng(SEED+s)
            sel=np.sort(rng.choice(sel,6000,replace=False))
        q=A[sel]
        n,ok=normals_for_scan(q)
        centers.append(A[scan==s].mean(axis=0)); idx.append(sel);nset.append(n);okset.append(ok)
        print('scan',s,'selected',len(sel),'normal valid',np.sum(ok),flush=True)
    return idx,np.array(centers),nset,okset

def match_pairs(A,scan,inds,cent,normals,oks,X,max_per_pair=750):
    pairset={(i,i+1) for i in range(9)}|{(i,i+2) for i in range(8)}|{(min(i,5),max(i,5)) for i in range(10) if i!=5}
    matches=[]
    for i,j in sorted(pairset):
        ip=inds[i];jp=inds[j]; p=A[ip];q=A[jp]
        P=ptransform(p,X[i],cent[i]);Q=ptransform(q,X[j],cent[j]);
        tree=cKDTree(Q[oks[j]])
        valid_j=np.flatnonzero(oks[j]);valid_i=np.flatnonzero(oks[i])
        if len(valid_i)<50 or len(valid_j)<50:continue
        dist,loc=tree.query(P[valid_i],workers=1)
        nbr=valid_j[loc]; nj=normals[j][nbr]@Rotation.from_rotvec(X[j,3:]).as_matrix().T
        ni=normals[i][valid_i]@Rotation.from_rotvec(X[i,3:]).as_matrix().T
        pln=np.abs(np.einsum('ij,ij->i',nj,P[valid_i]-Q[nbr]))
        accept=(dist<.42)&(np.abs(np.einsum('ij,ij->i',ni,nj))>.87)&(pln<.15)
        indices=np.flatnonzero(accept)
        if len(indices)<65:continue
        # stable, spatially distributed matches
        pi=valid_i[indices]
        chosen=vox_sample(P[pi],.25)
        indices=indices[chosen]
        if len(indices)>max_per_pair:
            rng=np.random.default_rng(SEED+i*100+j)
            indices=np.sort(rng.choice(indices,max_per_pair,replace=False))
        ui=valid_i[indices];vj=nbr[indices]
        matches.append((i,j,A[ip[ui]],A[jp[vj]],normals[j][vj]))
    return matches

def solve_step(matches,cent,X):
    rows=[];cols=[];vals=[];rhs=[];wt=[];rnum=0
    for i,j,p,q,norig in matches:
        Rj=Rotation.from_rotvec(X[j,3:]).as_matrix(); normal=norig@Rj.T
        P=ptransform(p,X[i],cent[i]);Q=ptransform(q,X[j],cent[j]);
        resid=np.einsum('ij,ij->i',normal,P-Q)
        # Robust IRLS weights, 7cm transition
        w=1/np.sqrt(1+(resid/.07)**2)
        jac_i=np.hstack([normal,np.cross(P-cent[i]-X[i,:3],normal)])
        jac_j=-np.hstack([normal,np.cross(Q-cent[j]-X[j,:3],normal)])
        N=len(resid);indices=np.arange(rnum,rnum+N)
        for s,mat in ((i,jac_i),(j,jac_j)):
            if s==5:continue
            k=s if s<5 else s-1
            for d in range(6):
                rows.extend(indices.tolist());cols.extend(np.full(N,k*6+d,dtype=int).tolist());vals.extend((mat[:,d]*w).tolist())
        rhs.extend((-resid*w).tolist());rnum+=N
    data=coo_matrix((vals,(rows,cols)),shape=(rnum,54)).tocsr();y=np.array(rhs)
    # Tikhonov and adjacent-scan smoothing, in metric units
    Aextra=[];bextra=[]
    def mkreg(sd_t,sd_r):
        return [1/sd_t]*3+[1/sd_r]*3
    def ridx(s):return (s if s<5 else s-1)*6
    for s in range(10):
        if s==5:continue
        for d,v in enumerate(mkreg(.22,np.deg2rad(3.5))):
            row=np.zeros(54);row[ridx(s)+d]=v
            Aextra.append(row);bextra.append(-X[s,d]*v)
    # penalize additional deformation between neighboring corrections
    for i in range(9):
        j=i+1
        for d,v in enumerate(mkreg(.22,np.deg2rad(3.0))):
            row=np.zeros(54)
            if i!=5:row[ridx(i)+d]+=v
            if j!=5:row[ridx(j)+d]-=v
            Aextra.append(row);bextra.append(-(X[i,d]-X[j,d])*v)
    from scipy.sparse import csr_matrix
    a=svstack([data,csr_matrix(np.array(Aextra))],format='csr');b=np.r_[y,bextra]
    result=lsmr(a,b,atol=1e-8,btol=1e-8,maxiter=450)
    dx=np.zeros((10,6))
    for s in range(10):
        if s==5:continue
        dx[s]=result[0][ridx(s):ridx(s)+6]
    # cap each update step prevents catastrophic geometry collapse
    for s in range(10):
        t=np.linalg.norm(dx[s,:3]);r=np.linalg.norm(dx[s,3:])
        if t>.10:dx[s,:3]*=.10/t
        if r>np.deg2rad(1.5):dx[s,3:]*=np.deg2rad(1.5)/r
    return dx,dict(matches=sum(len(p) for i,j,p,q,n in matches),pairs=len(matches),resid_rmse_cm=float(np.sqrt(np.mean(np.hstack([np.einsum('ij,ij->i',n@Rotation.from_rotvec(X[j,3:]).as_matrix().T,ptransform(p,X[i],cent[i])-ptransform(q,X[j],cent[j])) for i,j,p,q,n in matches])**2))*100))

def get_clock(records):
    base=records[5]['stamp'];return np.array([(r['stamp']-base)*1e-9+np.median(r['dt']) for r in records])

def apply(A,scan,records,X,cent,mode):
    out=np.empty_like(A,dtype=np.float64)
    if mode=='A':return A.copy()
    if mode=='B':
        for s in range(10):
            m=scan==s;out[m]=ptransform(A[m],X[s],cent[s])
        return out
    # Continuous correction C(t) as piecewise-linear translation and rotvec
    # Note: this is interpolation, NOT IMU-aided continuous optimization.
    times=get_clock(records);tt=np.hstack([np.full(len(rec['dt']),(rec['stamp']-records[5]['stamp'])*1e-9)+rec['dt'] for rec in records]);
    correction=np.stack([np.interp(tt,times,X[:,k]) for k in range(6)],axis=1)
    # rotations by rotvec of interpolated correction about interpolated scan center
    cc=np.stack([np.interp(tt,times,cent[:,k]) for k in range(3)],axis=1)
    out=Rotation.from_rotvec(correction[:,3:]).apply(A-cc)+cc+correction[:,:3]
    return out

def fit_plane(p):
    c=p.mean(axis=0);eig,vec=np.linalg.eigh((p-c).T@(p-c)/len(p));n=vec[:,0]
    return n,-n@c



@dataclass(frozen=True)
class ScanRefinementResult:
    corrections: np.ndarray   # (10,6): tx,ty,tz,rotvec(rad)
    centers: np.ndarray       # scan centroids in target LiDAR frame
    history: tuple[dict, ...]


def refine_d1_b(cloud, *, iterations: int = 5, verbose: bool = False) -> ScanRefinementResult:
    """Reimplement B in the original D1 experiment with equal random seeds.

    Keeps anchor scan 5 fixed. Training excludes the historical wall ROI;
    do not present training residual as independent alignment ground truth.
    """
    if iterations < 1:
        raise ValueError('iterations must be >= 1')
    scan = cloud.scan_id
    if set(np.unique(scan).tolist()) != set(range(10)):
        raise ValueError('Historical D1 B expects scan IDs 0..9')
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    inds, centers, normals, good = sample_model(xyz, scan)
    X = np.zeros((10, 6), dtype=np.float64)
    history = []
    for iteration in range(iterations):
        pairs = match_pairs(xyz, scan, inds, centers, normals, good, X)
        if not pairs:
            raise ValueError('No geometrically valid scan pairs; D1 refinement is underconstrained')
        dx, report = solve_step(pairs, centers, X)
        X += dx
        report.update(iteration=iteration,
                      translation_step_max_cm=float(np.linalg.norm(dx[:, :3], axis=1).max()*100),
                      rotation_step_max_deg=float(np.rad2deg(np.linalg.norm(dx[:, 3:], axis=1).max())))
        history.append(report)
        if verbose:
            print('ITER', report, flush=True)
        if np.linalg.norm(dx) < 1e-5:
            break
    return ScanRefinementResult(X, centers, tuple(history))


def apply_d1_corrections(cloud, scans, result: ScanRefinementResult, mode: str) -> np.ndarray:
    """Produce A/B/C1 on fixed source returns, with no point removal.

    B = per-scan small-angle correction around fitted scan centers.
    C1 = linearly interpolated B correction about interpolated centers.
    This is a temporal application method, not separately estimated poses.
    """
    if mode not in {'A','B','C1'}:
        raise ValueError('mode must be A, B or C1')
    records = [{'stamp': s.timestamp_ns, 'dt': s.time_offset_s} for s in scans]
    return apply(np.asarray(cloud.xyz, dtype=np.float64), cloud.scan_id,
                 records, result.corrections, result.centers, mode)

"""Calibration-assisted approximate ring recovery and multi-view edge evidence.

Raw KITTI XYZ is never recalibrated. XML angles are degrees, offsets cm.
Offset model is the common single-distance beam model, not a complete inverse
of the original KITTI decoder (two-point distance correction, deskew unknown).
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def load_beams(path):
    """Read Velodyne factory db.xml, sorted by increasing vertical angle."""
    root=ET.parse(Path(path)).getroot()
    rows=[]
    for e in root.iter():
        fields={x.tag.split("}")[-1].rstrip("_"):x.text for x in e}
        if "id" not in fields or "vertCorrection" not in fields:continue
        rows.append((int(fields["id"]),float(fields["vertCorrection"]),
                     float(fields.get("vertOffsetCorrection",0))/100.,
                     float(fields.get("horizOffsetCorrection",0))/100.))
    b=np.asarray(rows,float).reshape(-1,4)
    if len(b)!=64 or len(np.unique(b[:,0]))!=64 or not np.isfinite(b).all():
        raise ValueError("Expected 64 unique HDL-64E XML laser records")
    if np.any(abs(b[:,1])>=45) or np.any(abs(b[:,2:])>1):
        raise ValueError("Unexpected beam angle/offset units")
    b=b[np.argsort(b[:,1],kind="stable")]
    if np.any(np.diff(b[:,1])<=0):raise ValueError("Duplicate vertical angles")
    return b


def recover_rings(xyz,beams,*,model="offset",max_error_deg=.20,margin_deg=.04,
                  allowed_ranks=None):
    """Classify against ALL 64 channels, then optionally reject unused ranks.

    Rank 0 is lowest elevation. Laser ID is separately stored. Offset model:
    rho^2 = a^2 + h^2, z = a*tan(alpha)+v/cos(alpha).
    Score is equivalent elevation mismatch, not a calibrated uncertainty.
    No forced classification for ambiguous returns.
    """
    p=np.asarray(xyz,float);b=np.asarray(beams,float)
    if (p.ndim!=2 or p.shape[1]!=3 or b.shape!=(64,4)
        or not np.isfinite(p).all() or not np.isfinite(b).all()
        or model not in ("angle","offset") or not np.isfinite([max_error_deg,margin_deg]).all()
        or max_error_deg<=0 or margin_deg<0):
        raise ValueError("Invalid ring inputs")
    if allowed_ranks is not None:
        allowed=np.asarray(allowed_ranks)
        if allowed.ndim!=1 or allowed.dtype.kind not in "iu" or np.any((allowed<0)|(allowed>=64)):
            raise ValueError("Invalid allowed ranks")
    a=np.deg2rad(b[:,1]);v=b[:,2];h=b[:,3]
    best=np.empty(len(p),int);err=np.empty(len(p));gap=np.empty(len(p))
    for start in range(0,len(p),8192):
        sl=slice(start,start+8192);q=p[sl]
        rho=np.hypot(q[:,0],q[:,1])[:,None];z=q[:,2,None]
        if model=="angle":
            residual=abs(np.rad2deg(np.arctan2(z,rho))-b[:,1])
        else:
            radial=np.sqrt(np.maximum(rho*rho-h*h,0))
            residual=abs(np.rad2deg(np.arctan2(z-v/np.cos(a),radial))-b[:,1])
            residual=np.where(rho>abs(h),residual,1e6)
        order=np.argsort(residual,axis=1,kind="stable")[:,:2]
        first,second=np.take_along_axis(residual,order,axis=1).T
        best[sl]=order[:,0];err[sl]=first;gap[sl]=second-first
    valid=(err<=max_error_deg)&(gap>=margin_deg)
    if allowed_ranks is not None:valid &= np.isin(best,allowed)
    return dict(ring=np.where(valid,best,-1),best_rank=best,
                laser_id=np.where(valid,b[best,0].astype(int),-1),
                error_deg=err,margin_deg=gap)


def ring_edges(xyz,ring,*,az_gap_deg=.8,jump_abs=1.,jump_rel=.1,
               same_abs=.4,same_rel=.05):
    """Same inferred ring, cyclic azimuth neighbors, observed near/far/back.

    No unknown returns or arbitrary inter-ring jumps. Only horizontal
    range discontinuities: misses boundaries without azimuth contrast.
    """
    p=np.asarray(xyz,float);ring=np.asarray(ring)
    vals=[az_gap_deg,jump_abs,jump_rel,same_abs,same_rel]
    if (p.ndim!=2 or p.shape[1]!=3 or ring.shape!=(len(p),)
        or ring.dtype.kind not in "iu" or not np.isfinite(p).all()
        or not np.isfinite(vals).all() or min(vals)<=0 or az_gap_deg>=90):
        raise ValueError("Invalid edge inputs")
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    r=np.linalg.norm(p,axis=1);out=[]
    for k in np.unique(ring[ring>=0]):
        idx=np.flatnonzero(ring==k)
        if len(idx)<3:continue
        i=idx[np.argsort(az[idx],kind="stable")]
        prev=np.roll(i,1);nxt=np.roll(i,-1)
        gp=(az[i]-az[prev])%360;gn=(az[nxt]-az[i])%360
        ok=(gp>1e-8)&(gn>1e-8)&(gp<=az_gap_deg)&(gn<=az_gap_deg)
        for far,back in ((prev,nxt),(nxt,prev)):
            keep=ok&(r[far]-r[i]>=np.maximum(jump_abs,jump_rel*r[i]))
            keep &= abs(r[back]-r[i])<=np.maximum(same_abs,same_rel*r[i])
            out.append(np.column_stack((i[keep],far[keep],back[keep])))
    return np.concatenate(out) if out else np.empty((0,3),int)


def recurrence(xyz,scan_ids,origins,*,radius=.25):
    """Count distinct scans with edge candidates in a metric neighborhood.

    One nearest return per scan; own scan counted once. No physical identity
    guarantee. Angle is maximum deviation from own ray to supporting origins,
    not full pairwise span. RMS is spread, not a localization error bound.
    """
    p=np.asarray(xyz,float);s=np.asarray(scan_ids);o=np.asarray(origins,float)
    if (p.ndim!=2 or p.shape[1]!=3 or s.shape!=(len(p),) or s.dtype.kind not in "iu"
        or o.ndim!=2 or o.shape[1]!=3 or np.any(s<0) or np.any(s>=len(o))
        or not np.isfinite(p).all() or not np.isfinite(o).all()
        or not np.isfinite(radius) or radius<=0):
        raise ValueError("Invalid recurrence inputs")
    n=len(p);m=len(o)
    support=np.full((n,m),-1,int);distance=np.full((n,m),np.nan)
    ids=np.arange(n)
    support[ids,s]=ids;distance[ids,s]=0.
    for j in range(m):
        jj=np.flatnonzero(s==j);query=np.flatnonzero(s!=j)
        if not len(jj) or not len(query):continue
        d,k=cKDTree(p[jj]).query(p[query],distance_upper_bound=np.nextafter(radius,np.inf))
        good=np.isfinite(d)
        support[query[good],j]=jj[k[good]]
        distance[query[good],j]=d[good]
    count=(support>=0).sum(1)
    rays=p-o[s];length=np.linalg.norm(rays,axis=1)
    if np.any(length<=1e-8):raise ValueError("Candidate at source origin")
    unit=rays/length[:,None];angle=np.zeros(n)
    for j in range(m):
        q=p-o[j];norm=np.linalg.norm(q,axis=1)
        ok=(support[:,j]>=0)&(norm>1e-8)
        dot=np.sum(unit[ok]*q[ok]/norm[ok,None],axis=1)
        angle[ok]=np.maximum(angle[ok],np.rad2deg(np.arccos(np.clip(dot,-1,1))))
    return dict(support_indices=support,support_distance_m=distance,scan_count=count,
                rms_m=np.sqrt(np.nansum(distance**2,axis=1)/np.maximum(count,1)),
                view_angle_deg=angle)


def projected_candidates(xyz,X,K,shape,reference_depth,*,max_depth=40.,margin=.5):
    """Nearest edge candidate per pixel; reject known reference occlusion.

    Reference is full accumulated cloud z-buffer, not DA2. Unknown pixels
    remain unknown. Same-pixel check only, no full surface visibility.
    """
    from spade_calib.calibration.point_line_occlusion import project_grid
    d=np.asarray(reference_depth,float)
    if (d.shape!=tuple(shape) or np.isinf(d).any() or np.any(d[np.isfinite(d)]<=0)
        or not np.isfinite(margin) or margin<0):
        raise ValueError("Invalid reference depth")
    q=project_grid(xyz,X,K,shape,np.ones(len(xyz),bool),cell=1,max_depth=max_depth)
    pix=np.floor(q["uv"]).astype(int)
    ref=d[pix[:,1],pix[:,0]];known=np.isfinite(ref)
    visible=~known|(q["z"]<=ref+margin)
    return dict(candidate_indices=q["ids"],uv=q["uv"],z_m=q["z"],
                reference_z_m=ref,reference_known=known,visible=visible)

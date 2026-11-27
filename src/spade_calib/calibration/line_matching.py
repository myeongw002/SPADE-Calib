"""Fixed-initializer 3D-to-2D segment matching, no GT or label snapping."""
from __future__ import annotations
import numpy as np


def transform_line(xyz,Ri,ti,Rj,tj,X):
    """World-from-LiDAR poses; X is camera-from-LiDAR."""
    p=np.asarray(xyz,float)
    return (((p@Ri.T+ti-tj)@Rj)@X[:3,:3].T+X[:3,3])


def clip_projected_line(q,K,shape,min_depth=.5,min_length=15.):
    """Clip finite 3D segment at near plane, then projected segment at image."""
    q=np.asarray(q,float).copy();K=np.asarray(K,float)
    if q.shape!=(2,3) or K.shape!=(3,3) or not np.isfinite(q).all() or not np.isfinite(K).all():
        raise ValueError("Invalid projection inputs")
    if min_depth<=0 or min_length<=0: raise ValueError("Invalid clipping settings")
    h,w=shape
    if min(h,w)<2: raise ValueError("Invalid image shape")
    if np.all(q[:,2]<min_depth): return None
    for i in range(2):
        if q[i,2]<min_depth:
            j=1-i
            a=(min_depth-q[i,2])/(q[j,2]-q[i,2])
            q[i]+=a*(q[j]-q[i])
    uv=(q@K.T)[:,:2]/q[:,2,None]
    a,b=uv;d=b-a;lo,hi=0.,1.
    # Liang-Barsky interval clipping.
    for p,r in ((-d[0],a[0]),(d[0],w-1-a[0]),(-d[1],a[1]),(d[1],h-1-a[1])):
        if abs(p)<1e-12:
            if r<0: return None
        elif p<0: lo=max(lo,r/p)
        else: hi=min(hi,r/p)
        if lo>hi: return None
    out=np.stack((a+lo*d,a+hi*d))
    return out if np.linalg.norm(out[1]-out[0])>=min_length else None


def rank_candidates(projected,segments,*,max_distance=20.,max_angle=10.,
                    min_overlap=.3,min_length=15.,ambiguity_gap=.1,top_k=5):
    """Gate and rank using finite visible projection, unoriented angle.

    Distance: mean absolute perpendicular distance of the two projected
    endpoints to the candidate's infinite line. Overlap: intersection of
    projected candidate interval with visible LiDAR segment / its length.
    Score lower is better. No detector score or manual 2D label used.
    """
    p=np.asarray(projected,float);s=np.asarray(segments,float).reshape(-1,2,2)
    if p.shape!=(2,2) or not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("Invalid segment coordinates")
    if not np.isfinite([max_distance,max_angle,min_overlap,min_length,ambiguity_gap]).all() or not (
            max_distance>0 and 0<max_angle<=90 and 0<=min_overlap<=1 and min_length>0 and ambiguity_gap>=0 and top_k>=2):
        raise ValueError("Invalid matching settings")
    v=p[1]-p[0];L=np.linalg.norm(v)
    if L<min_length: raise ValueError("Projection too short")
    v/=L
    dv=s[:,1]-s[:,0];length=np.linalg.norm(dv,axis=1)
    ids=np.flatnonzero(length>=min_length)
    u=dv[ids]/length[ids,None]
    normal=np.column_stack((-u[:,1],u[:,0]))
    diff=p[None,:,:]-s[ids,0,None,:]
    distance=np.mean(abs(np.einsum("nki,ni->nk",diff,normal)),axis=1)
    angle=np.rad2deg(np.arccos(np.clip(abs(u@v),0,1)))
    interval=(s[ids]-p[0])@v
    overlap=np.maximum(0,np.minimum(L,interval.max(axis=1))-np.maximum(0,interval.min(axis=1)))/L
    good=(distance<=max_distance)&(angle<=max_angle)&(overlap>=min_overlap)
    cost=(distance/max_distance)**2+(angle/max_angle)**2+(1-overlap)**2
    order=np.lexsort((ids,cost));order=order[good[order]]
    candidates=[dict(segment_index=int(ids[i]),uv=s[ids[i]].tolist(),
                     distance_px=float(distance[i]),angle_deg=float(angle[i]),
                     overlap=float(overlap[i]),cost=float(cost[i]),length_px=float(length[ids[i]]))
                for i in order[:top_k]]
    gap=float(cost[order[1]]-cost[order[0]]) if len(order)>1 else None
    status="no_candidate" if not len(order) else ("ambiguous" if gap is not None and gap<ambiguity_gap else "proposed")
    return dict(status=status,candidates=candidates,gated_count=int(len(order)),cost_gap=gap,
                selected_index=candidates[0]["segment_index"] if status=="proposed" else None)


def reject_conflicts(rows):
    """No single image segment can automatically represent two 3D records."""
    groups={}
    for r in rows:
        if r["status"]=="proposed":
            groups.setdefault((r["frame"],r["selected_index"]),[]).append(r)
    for g in groups.values():
        if len(g)>1:
            for r in g:
                r["status"]="conflict";r["selected_index"]=None

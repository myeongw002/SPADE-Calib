"""C0: image-line-guided local LiDAR plane extraction.

Image geometry selects spatial cells, never modifies 3D plane coefficients.
A proposed match is not a verified physical correspondence.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from spade_calib.calibration.lidar_line_extraction import Settings, ransac_plane, intersection
from spade_calib.calibration.line_matching import clip_projected_line, rank_candidates


@dataclass(frozen=True)
class GuideSettings:
    min_image_length_px: float = 40.
    roi_width_px: float = 30.
    endpoint_margin_px: float = 20.
    min_cell_votes: int = 5
    max_distance_px: float = 20.
    max_angle_deg: float = 10.
    min_overlap: float = .30
    ambiguity_gap: float = .10
    min_projection_length_px: float = 15.

    def validate(self):
        vals=list(vars(self).values())
        if not np.isfinite(vals).all() or not (
            self.min_image_length_px>0 and self.roi_width_px>0 and self.endpoint_margin_px>=0
            and self.min_cell_votes>=1 and self.max_distance_px>0
            and 0<self.max_angle_deg<=90 and 0<=self.min_overlap<=1
            and self.ambiguity_gap>=0 and 0<self.min_projection_length_px<=self.min_image_length_px):
            raise ValueError("Invalid guide settings")


def active_cells(points, segments, X, K, shape, cfg, guide):
    """Map finite pixel corridors to occupied 3D cells via projected returns.

    Returns all selected cell keys and guiding segment IDs. No depth
    collapse: cells at different ranges remain distinct. No image-based
    point removal inside a selected cell during subsequent plane fitting.
    """
    cfg.validate();guide.validate()
    p=np.asarray(points,float);s=np.asarray(segments,float).reshape(-1,2,2)
    X=np.asarray(X,float);K=np.asarray(K,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or not np.isfinite(s).all() or X.shape!=(4,4) or K.shape!=(3,3)
            or not np.isfinite(X).all() or not np.isfinite(K).all()
            or not np.allclose(K[2],[0,0,1])):
        raise ValueError("Invalid projection data")
    h,w=shape
    if min(h,w)<2:raise ValueError("Invalid image")
    inrange=np.flatnonzero(np.linalg.norm(p,axis=1)<=cfg.max_range_m)
    if not len(inrange):return {},inrange,dict(range_points=0,projected_points=0,occupied_cells=0,eligible_image_lines=0)
    cells,inv=np.unique(np.floor(p[inrange]/cfg.cell_m).astype(np.int64),axis=0,return_inverse=True)
    q=p[inrange]@X[:3,:3].T+X[:3,3]
    keep=np.flatnonzero(q[:,2]>.5)
    uv=(q[keep]@K.T)[:,:2]/q[keep,2,None]
    # Permit projected returns slightly outside the image border if in
    # the extended finite corridor.
    margin=guide.roi_width_px+guide.endpoint_margin_px
    good=(uv[:,0]>=-margin)&(uv[:,0]<=w-1+margin)&(uv[:,1]>=-margin)&(uv[:,1]<=h-1+margin)
    uv=uv[good];cell_ids=inv[keep[good]]
    chosen=defaultdict(dict)
    lengths=np.linalg.norm(s[:,1]-s[:,0],axis=1)
    for i in np.flatnonzero(lengths>=guide.min_image_length_px):
        a,b=s[i];v=(b-a)/lengths[i];n=np.array([-v[1],v[0]])
        delta=uv-a
        t=np.einsum("ij,j->i",delta,v);d=abs(np.einsum("ij,j->i",delta,n))
        hits=(d<=guide.roi_width_px)&(t>=-guide.endpoint_margin_px)&(t<=lengths[i]+guide.endpoint_margin_px)
        ids,count=np.unique(cell_ids[hits],return_counts=True)
        for j,k in zip(ids,count):
            if k>=guide.min_cell_votes:chosen[tuple(map(int,cells[j]))][int(i)]=int(k)
    return dict(chosen),inrange,dict(range_points=len(inrange),projected_points=len(uv),
                                    occupied_cells=len(cells),eligible_image_lines=int((lengths>=guide.min_image_length_px).sum()))


def extract_guided(points,scans,segments,X,K,shape,cfg=Settings(),guide=GuideSettings()):
    p=np.asarray(points,float);scans=np.asarray(scans)
    if scans.shape!=(len(p),) or scans.dtype.kind not in "iu":
        raise ValueError("Invalid scan IDs")
    selected,ids,stats=active_cells(p,segments,X,K,shape,cfg,guide)
    counts=Counter();lines=[]
    if not len(ids):return [],dict(**stats,active_cells=0,eligible_cells=0,raw_lines=0)
    tree=cKDTree(p[ids])
    for cell,votes in sorted(selected.items()):
        center=(np.asarray(cell)+.5)*cfg.cell_m
        local=ids[np.asarray(tree.query_ball_point(center,cfg.cell_m*(.5+cfg.halo_fraction),p=np.inf),int)]
        if len(local)<2*cfg.min_inliers:continue
        counts["eligible_cells"]+=1
        # Cell-local random seed, invariant to selection/order of other cells.
        seed=np.random.SeedSequence([cfg.seed,*[int(x)&0xffffffff for x in cell]])
        rng=np.random.default_rng(seed)
        planes=[];remaining=np.sort(local)
        for _ in range(cfg.max_planes):
            plane=ransac_plane(p,remaining,cfg,rng)
            if plane is None:break
            planes.append(plane)
            remaining=np.setdiff1d(remaining,plane["ids"],assume_unique=True)
            if len(remaining)<cfg.min_inliers:break
        counts["planes"]+=len(planes)
        for i in range(len(planes)):
            for j in range(i+1,len(planes)):
                line,reason=intersection(p,scans,planes[i],planes[j],cfg)
                counts[reason]+=1
                if line is not None:
                    line["cells"]=[list(cell)]
                    line["guide_votes"]=votes
                    lines.append(line)
    return lines,dict(**stats,active_cells=len(selected),raw_lines=len(lines),**counts)


def associate(lines,segments,X,K,shape,guide=GuideSettings(),require_guide=False):
    """Rank against all eligible image lines, not only the triggering one.

    Multiple local 3D fragments can project to the same image line.
    Retain them as unverified proposals and report multiplicity; do not
    mistake a 2D line ID for a unique 3D identity.
    """
    guide.validate()
    s=np.asarray(segments,float).reshape(-1,2,2)
    eligible=np.flatnonzero(np.linalg.norm(s[:,1]-s[:,0],axis=1)>=guide.min_image_length_px)
    rows=[]
    for i,line in enumerate(lines):
        xyz=np.asarray(line["xyz"],float)
        uv=clip_projected_line(xyz@X[:3,:3].T+X[:3,3],K,shape,
                               min_length=guide.min_projection_length_px)
        row=dict(id=i,xyz_lidar=xyz.tolist(),projected_uv=None,status="not_projectable",
                 selected_index=None,candidates=[],guide_votes=line.get("guide_votes",{}))
        if uv is not None:
            row["projected_uv"]=uv.tolist()
            rank=rank_candidates(uv,s[eligible],max_distance=guide.max_distance_px,
                 max_angle=guide.max_angle_deg,min_overlap=guide.min_overlap,
                 min_length=guide.min_projection_length_px,ambiguity_gap=guide.ambiguity_gap)
            for c in rank["candidates"]:c["segment_index"]=int(eligible[c["segment_index"]])
            if rank["selected_index"] is not None:
                rank["selected_index"]=int(eligible[rank["selected_index"]])
            row.update(rank)
            if require_guide and row["status"]=="proposed" and row["selected_index"] not in row["guide_votes"]:
                row["status"]="guide_mismatch";row["selected_index"]=None
        rows.append(row)
    groups=Counter(r["selected_index"] for r in rows if r["status"]=="proposed")
    for r in rows:
        r["image_line_proposal_count"]=groups.get(r["selected_index"],0)
    return rows

"""LOAM-inspired same-ring smoothness; not a full LOAM implementation."""
from __future__ import annotations
import numpy as np


def ring_smoothness(xyz, ring, *, half_window=5, az_gap_deg=.8,
                    occlusion_abs=.3, occlusion_rel=.03):
    """Normalized LOAM-paper-style local difference norm.

    c_i = ||sum_{j != i}(p_j - p_i)|| / (2*m*||p_i||).
    Full circular same-ring window required, no invalid angular gaps.
    Clean candidates reject any strong adjacent radial jump in that window.
    This is an occlusion proxy, not a physical crease classifier.
    """
    p=np.asarray(xyz,float); ring=np.asarray(ring)
    if (p.ndim!=2 or p.shape[1]!=3 or ring.shape!=(len(p),)
        or ring.dtype.kind not in "iu" or not np.isfinite(p).all()
        or not isinstance(half_window,(int,np.integer)) or half_window<1
        or not np.isfinite([az_gap_deg,occlusion_abs,occlusion_rel]).all()
        or min(az_gap_deg,occlusion_abs,occlusion_rel)<=0 or az_gap_deg>=90):
        raise ValueError("Invalid smoothness inputs")
    r=np.linalg.norm(p,axis=1)
    if np.any(r<=1e-8):raise ValueError("Zero range")
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    score=np.full(len(p),np.nan)
    clean=np.zeros(len(p),bool)
    m=half_window
    for k in np.unique(ring[ring>=0]):
        idx=np.flatnonzero(ring==k)
        if len(idx)<2*m+1:continue
        i=idx[np.argsort(az[idx],kind="stable")]
        nxt=np.roll(i,-1)
        gap=(az[nxt]-az[i])%360
        gap_ok=(gap>1e-8)&(gap<=az_gap_deg)
        jump=abs(r[nxt]-r[i])
        continuous=jump<np.maximum(occlusion_abs,occlusion_rel*np.minimum(r[nxt],r[i]))
        valid=np.ones(len(i),bool); no_jump=valid.copy()
        delta=np.zeros((len(i),3))
        for d in range(-m,m+1):
            if d:
                delta += p[np.roll(i,-d)]-p[i]
            if d<m:
                valid &= np.roll(gap_ok,-d)
                no_jump &= np.roll(continuous,-d)
        score[i[valid]]=np.linalg.norm(delta[valid],axis=1)/(2*m*r[i[valid]])
        clean[i]=valid&no_jump
    return score,clean


def select_top(score, eligible, budget):
    """Deterministic top-K, no invented positives or fixed GT-tuned threshold."""
    s=np.asarray(score,float);e=np.asarray(eligible,bool)
    if s.ndim!=1 or e.shape!=s.shape or not isinstance(budget,(int,np.integer)) or budget<0:
        raise ValueError("Invalid candidate budget")
    ids=np.flatnonzero(e&np.isfinite(s))
    order=np.lexsort((ids,-s[ids]))
    return ids[order[:budget]]

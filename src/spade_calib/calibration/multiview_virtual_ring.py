"""Multi-view virtual scan helpers; immutable accumulated return identities."""
from __future__ import annotations
import numpy as np


def reframe_points(xyz,R_anchor,t_anchor,R_view,t_view):
    """Pose convention world-from-LiDAR, row-vector points.

    p_view = R_view.T @ (R_anchor @ p_anchor + t_anchor - t_view).
    No registration, interpolation or point synthesis.
    """
    p=np.asarray(xyz,float);ra=np.asarray(R_anchor,float);rv=np.asarray(R_view,float)
    ta=np.asarray(t_anchor,float);tv=np.asarray(t_view,float)
    if (p.ndim!=2 or p.shape[1]!=3 or ra.shape!=(3,3) or rv.shape!=(3,3)
        or ta.shape!=(3,) or tv.shape!=(3,)
        or not all(np.isfinite(x).all() for x in (p,ra,rv,ta,tv))):
        raise ValueError("Invalid rigid frame inputs")
    for r in (ra,rv):
        if not np.allclose(r.T@r,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7):
            raise ValueError("Expected proper rotation")
    return p@(ra.T@rv)+(ta-tv)@rv


def view_indices(n,target,stride):
    """Center and symmetric stride offsets within the available scan window."""
    if not all(isinstance(x,(int,np.integer)) for x in (n,target,stride)) or not 0<=target<n or stride<1:
        raise ValueError("Invalid view selection")
    return np.flatnonzero((np.arange(n)-target)%stride==0)


def union_view_edges(triples,n_points):
    """Unique near-return IDs and number of distinct *virtual* views.

    Multiple hypotheses in one view count once. Views share the input cloud;
    this is not independent scan support or a physical boundary identity.
    """
    if not isinstance(n_points,(int,np.integer)) or n_points<0:
        raise ValueError("Invalid point count")
    counts=np.zeros(n_points,np.int32)
    for e in triples:
        e=np.asarray(e)
        if e.ndim!=2 or e.shape[1]!=3 or e.dtype.kind not in "iu" or np.any((e<0)|(e>=n_points)):
            raise ValueError("Invalid edge triple indices")
        counts[np.unique(e[:,0])]+=1
    return np.flatnonzero(counts),counts

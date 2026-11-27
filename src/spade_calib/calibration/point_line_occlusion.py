"""Camera-view depth-jump point proposals. No 3D line fitting."""
import numpy as np

def project_grid(xyz, X, K, shape, non_ground, cell=4, max_depth=40.):
    p=np.asarray(xyz,float); X=np.asarray(X,float); K=np.asarray(K,float)
    mask=np.asarray(non_ground,bool)
    if p.ndim!=2 or p.shape[1]!=3 or mask.shape!=(len(p),) or not np.isfinite(p).all() or cell<1 or max_depth<=.5:
        raise ValueError("Invalid cloud/grid")
    h,w=shape
    q=p@X[:3,:3].T+X[:3,3]
    ids=np.flatnonzero((q[:,2]>=.5)&(q[:,2]<=max_depth))
    uv=(q[ids]@K.T)[:,:2]/q[ids,2,None]
    ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids,uv=ids[ok],uv[ok]
    gh,gw=(h+cell-1)//cell,(w+cell-1)//cell
    ij=np.floor(uv/cell).astype(int); key=ij[:,1]*gw+ij[:,0]
    order=np.lexsort((ids,q[ids,2],key))
    _,first=np.unique(key[order],return_index=True)
    sel=order[first]
    ids,uv=ids[sel],uv[sel]
    grid=np.full((gh,gw),-1,int);grid.flat[key[sel]]=np.arange(len(ids))
    return dict(ids=ids,uv=uv,z=q[ids,2],non_ground=mask[ids],grid=grid)

def occlusion_edges(projected, *, jump_abs=1., jump_rel=.1, gap=2, same_rel=.05, same_abs=.4):
    """Foreground original returns, observed far side + same-depth back support.
    Empty cells are unknown, never sufficient to create an edge.
    Cardinal axes only. Direction proxy points from near towards far.
    """
    g=projected["grid"];z=projected["z"];ng=projected["non_ground"]
    if gap<1 or min(jump_abs,jump_rel,same_rel,same_abs)<=0:
        raise ValueError("Invalid discontinuity settings")
    n=len(z);dirs=np.zeros((n,4),bool)
    h,w=g.shape
    yy,xx=np.nonzero(g>=0);i=g[yy,xx]
    for d,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
        far=np.full(len(i),-1,int);back=far.copy()
        for step in range(1,gap+1):
            for sign,arr in ((1,far),(-1,back)):
                x=xx+sign*step*dx;y=yy+sign*step*dy
                ok=(x>=0)&(x<w)&(y>=0)&(y<h)&(arr<0)
                arr[ok]=g[y[ok],x[ok]]
        valid=(far>=0)&(back>=0)&ng[i]
        a=i[valid];b=far[valid];c=back[valid]
        good=(z[b]-z[a]>=np.maximum(jump_abs,jump_rel*z[a])) & ng[c] & (abs(z[c]-z[a])<=np.maximum(same_abs,same_rel*z[a]))
        dirs[a[good],d]=True
    return dirs

def match_points(projected, edge_dirs, segments, *, gate=20., use_edges=True, angle_deg=45., ambiguity=4., bin_px=8., min_bins=3, min_coverage=.25):
    """Unique nearest finite line with ambiguity rejection and along-line bins.
    A retained proposal is NOT a verified correspondence.
    """
    s=np.asarray(segments,float).reshape(-1,2,2)
    if not np.isfinite(s).all() or gate<=0 or not 0<angle_deg<=90 or ambiguity<0 or bin_px<=0 or min_bins<1 or not 0<=min_coverage<=1:
        raise ValueError("Invalid match settings")
    uv=projected["uv"]; ids=np.flatnonzero(projected["non_ground"] & (edge_dirs.any(axis=1) if use_edges else True))
    length=np.linalg.norm(s[:,1]-s[:,0],axis=1)
    if np.any(length<=0):raise ValueError("Degenerate line")
    tangent=(s[:,1]-s[:,0])/length[:,None]
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    best=np.full(len(ids),np.inf);second=best.copy();choice=np.full(len(ids),-1,int);along=np.zeros(len(ids));signed=np.zeros(len(ids))
    for j in range(len(s)):
        q=uv[ids]-s[j,0];a=q@tangent[j];r=q@normal[j];dist=abs(r)
        ok=(a>=0)&(a<=length[j])&(dist<=gate)
        if use_edges:
            axis=np.array([abs(normal[j,0]),abs(normal[j,0]),abs(normal[j,1]),abs(normal[j,1])])
            ok &= np.any(edge_dirs[ids] & (axis[None]>=np.cos(np.deg2rad(angle_deg))-1e-12),axis=1)
        better=ok&(dist<best);other=ok&~better
        second[other]=np.minimum(second[other],dist[other]);second[better]=best[better]
        best[better]=dist[better];choice[better]=j;along[better]=a[better];signed[better]=r[better]
    accepted=choice>=0
    good=accepted.copy()
    good[accepted]=(second[accepted]-best[accepted]>=ambiguity)
    groups=[]
    for j in range(len(s)):
        a=np.flatnonzero(good&(choice==j))
        if not len(a):continue
        # One nearest candidate per occupied along-line bin; cap correlation.
        bins=np.minimum(np.floor(along[a]/bin_px).astype(int),int(np.ceil(length[j]/bin_px))-1)
        order=np.lexsort((ids[a],best[a],bins));_,first=np.unique(bins[order],return_index=True)
        keep=a[order[first]]
        coverage=len(keep)/int(np.ceil(length[j]/bin_px))
        if len(keep)<min_bins or coverage<min_coverage:continue
        groups.append(dict(line=j,point_indices=projected["ids"][ids[keep]],uv=uv[ids[keep]],signed_px=signed[keep],coverage=float(coverage),raw_count=len(a)))
    return groups,dict(eligible_points=len(ids),gated_points=int(accepted.sum()),unambiguous_points=int(good.sum()),retained_points=sum(len(g["point_indices"]) for g in groups),matched_lines=len(groups))

def evaluate_fixed(xyz, groups, segments, X, K, shape):
    """Evaluate already-selected pairs at a diagnostic transform, no rematching."""
    out=[]
    for g in groups:
        p=xyz[g["point_indices"]]@X[:3,:3].T+X[:3,3]
        valid=p[:,2]>.5;uv=np.full((len(p),2),np.nan);uv[valid]=(p[valid]@K.T)[:,:2]/p[valid,2,None]
        s=segments[g["line"]];v=s[1]-s[0];L=np.linalg.norm(v);v/=L;n=np.array([-v[1],v[0]])
        a=(uv-s[0])@v;r=(uv-s[0])@n
        inside=valid&(uv[:,0]>=0)&(uv[:,0]<shape[1])&(uv[:,1]>=0)&(uv[:,1]<shape[0])&(a>=0)&(a<=L)
        out.append(dict(line=g["line"],signed_px=r,inside=inside))
    return out

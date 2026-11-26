"""EP0: independent camera and sparse LiDAR edge-point proposals.

No image segments, 3D line fitting, cross-modal associations or optimization.
Directions are image-plane directions, not physical surface normals.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import cKDTree
from spade_calib.calibration.point_line_occlusion import project_grid


def plane_normals(uv, normal, K):
    """Local tangent interpretation-plane normal; not a surface normal."""
    uv, n, K = np.asarray(uv,float), np.asarray(normal,float), np.asarray(K,float)
    if uv.shape != n.shape or uv.ndim != 2 or uv.shape[1] != 2 or K.shape != (3,3):
        raise ValueError("Invalid edge geometry")
    l = np.column_stack((n, -np.sum(n*uv,axis=1)))
    q = l @ K
    norm = np.linalg.norm(q,axis=1)
    if not np.isfinite(q).all() or np.any(norm<=1e-12):
        raise ValueError("Invalid edge direction")
    return q/norm[:,None]


def _remap(a,x,y):
    return cv2.remap(a,np.asarray(x,np.float32),np.asarray(y,np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def _nms_candidates(response,nx,ny,margin):
    h,w=response.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    p=_remap(response,x+nx,y+ny)
    m=_remap(response,x-nx,y-ny)
    # Deterministic asymmetric tie-breaking; no connecting pixels into lines.
    good=(response>1e-7)&(response>=p)&(response>m)
    good &= (x>=margin)&(x<w-margin)&(y>=margin)&(y<h-margin)
    iy,ix=np.nonzero(good)
    return np.column_stack((ix,iy)).astype(float)


def _profiles(D,uv,n,offsets):
    """Bilateral affine fits in outward pixel coordinates."""
    A=np.column_stack((np.ones(len(offsets)),offsets))
    P=np.linalg.pinv(A)
    out=np.empty((len(uv),2),float)
    for start in range(0,len(uv),4096):
        sl=slice(start,start+4096)
        pos=uv[sl,None,:]+n[sl,None,:]*offsets[None,:,None]
        neg=uv[sl,None,:]-n[sl,None,:]*offsets[None,:,None]
        a=_remap(D,pos[...,0],pos[...,1])@P.T
        b=_remap(D,neg[...,0],neg[...,1])@P.T
        # Positive jump means minus side has larger inverse depth (near).
        out[sl,0]=b[:,0]-a[:,0]
        out[sl,1]=(offsets[-1]-offsets[0])*(a[:,1]+b[:,1])
    return out


def dense_edges(depth,K,*,sigma=1.2,jump_min=.02,crease_min=.02,
                offset_min=3,offset_max=9):
    """DA2 relative inverse-depth jump and crease *proxies*.

    Normalize by global P95-P5, smooth for candidate direction/NMS only,
    verify local bilateral profiles on the original normalized prediction.
    Jump direction points near->far. Crease direction is unoriented.
    Neither score is a calibrated confidence or ground-truth edge type.
    """
    D=np.asarray(depth,np.float32)
    if (D.ndim!=2 or not np.isfinite(D).all() or
        not np.isfinite([sigma,jump_min,crease_min]).all() or
        min(sigma,jump_min,crease_min)<=0 or not 1<=offset_min<offset_max or
        min(D.shape)<=2*(offset_max+2)):
        raise ValueError("Invalid dense depth/settings")
    lo,hi=np.percentile(D,[5,95]);scale=float(hi-lo)
    if scale<=1e-8:raise ValueError("Insufficient inverse-depth variation")
    D=((D-lo)/scale).astype(np.float32)
    smooth=cv2.GaussianBlur(D,(0,0),sigma)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=.125)
    mag=np.hypot(gx,gy)
    nx=-gx/np.maximum(mag,1e-12);ny=-gy/np.maximum(mag,1e-12)
    xx=cv2.Sobel(gx,cv2.CV_32F,1,0,ksize=3,scale=.125)
    yy=cv2.Sobel(gy,cv2.CV_32F,0,1,ksize=3,scale=.125)
    xy=.5*(cv2.Sobel(gx,cv2.CV_32F,0,1,ksize=3,scale=.125)+
           cv2.Sobel(gy,cv2.CV_32F,1,0,ksize=3,scale=.125))
    # Principal absolute Hessian eigen-direction. No eigendecomposition per pixel.
    theta=.5*np.arctan2(2*xy,xx-yy)
    mid=.5*(xx+yy);rad=np.hypot(.5*(xx-yy),xy)
    plus,minus=mid+rad,mid-rad
    use_minus=abs(minus)>abs(plus)
    cx=np.where(use_minus,-np.sin(theta),np.cos(theta)).astype(np.float32)
    cy=np.where(use_minus,np.cos(theta),np.sin(theta)).astype(np.float32)
    curvature=np.maximum(abs(plus),abs(minus)).astype(np.float32)
    offsets=np.arange(offset_min,offset_max+1,dtype=float)
    result={}
    for name,response,dx,dy,threshold in (
        ("jump",mag,nx,ny,jump_min),("crease",curvature,cx,cy,crease_min)):
        uv=_nms_candidates(response,dx,dy,offset_max+2)
        ix,iy=uv.astype(int).T
        n=np.column_stack((dx[iy,ix],dy[iy,ix])).astype(float)
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        profile=_profiles(D,uv,n,offsets)
        score=profile[:,0] if name=="jump" else abs(profile[:,1])
        keep=score>=threshold
        uv,n,profile,score=uv[keep],n[keep],profile[keep],score[keep]
        result[name]=dict(uv=uv,normal_uv=n,plane_normal=plane_normals(uv,n,K),
                          score=score,jump_signed=profile[:,0],crease_signed=profile[:,1])
    # These can overlap; a jump also generates curvature. Do not claim
    # exclusive physical classes or interpret curvature polarity as near/far.
    return result


def sparse_projection(xyz,X,K,shape,*,max_depth=40.):
    """Original-resolution z-buffer; unknown pixels are NaN / index -1."""
    p=project_grid(xyz,X,K,shape,np.ones(len(xyz),bool),cell=1,max_depth=max_depth)
    depth=np.full(shape,np.nan,np.float32); index=np.full(shape,-1,np.int64)
    y,x=np.nonzero(p["grid"]>=0);i=p["grid"][y,x]
    depth[y,x]=p["z"][i];index[y,x]=p["ids"][i]
    return p,depth,index


def sparse_edges(projected,xyz,K,*,radius=12.,jump_abs=1.,jump_rel=.1,
                 same_abs=.4,same_rel=.05,foreground_mask=None):
    """Nearest observed neighbor in each of eight angular sectors.

    Accept observed near->far jump only if opposite sector also has a
    similar-depth foreground point. Unknown cells never count as evidence.
    Preserve all directional hypotheses, including repeats of a near ID.
    Pixel bracket width is a sampling proxy, not an error bound.
    """
    p=projected;uv=np.asarray(p["uv"],float);z=np.asarray(p["z"],float)
    ids=np.asarray(p["ids"],int);xyz=np.asarray(xyz,float)
    if (not np.isfinite([radius,jump_abs,jump_rel,same_abs,same_rel]).all()
        or min(radius,jump_abs,jump_rel,same_abs,same_rel)<=0):
        raise ValueError("Invalid sparse edge settings")
    fg=np.ones(len(xyz),bool) if foreground_mask is None else np.asarray(foreground_mask,bool)
    if fg.shape!=(len(xyz),):raise ValueError("Invalid foreground mask")
    # Exact Euclidean radius search, no interpolation or fabricated depth.
    # Process chunks to bound memory on dense accumulated clouds.
    tree=cKDTree(uv)
    triples=[]
    for start in range(0,len(uv),1024):
        stop=min(start+1024,len(uv))
        lists=tree.query_ball_point(uv[start:stop],radius)
        counts=np.fromiter((len(v) for v in lists),int,count=stop-start)
        if counts.sum()==0:continue
        a=np.repeat(np.arange(start,stop),counts)
        b=np.concatenate(lists).astype(int)
        delta=uv[b]-uv[a];dist=np.linalg.norm(delta,axis=1)
        keep=(a!=b)&(dist>1e-8)
        a,b,delta,dist=a[keep],b[keep],delta[keep],dist[keep]
        sector=(np.floor((np.arctan2(delta[:,1],delta[:,0])+np.pi/8)/(np.pi/4)).astype(int)%8)
        key=(a-start)*8+sector
        order=np.lexsort((ids[b],dist,key))
        _,first=np.unique(key[order],return_index=True)
        nearest=np.full((stop-start,8),-1,int)
        nearest.flat[key[order[first]]]=b[order[first]]
        for s in range(8):
            aa=np.arange(start,stop);bb=nearest[:,s];cc=nearest[:,(s+4)%8]
            ok=(bb>=0)&(cc>=0)&fg[ids[aa]]
            aa,bb,cc=aa[ok],bb[ok],cc[ok]
            good=(z[bb]-z[aa]>=np.maximum(jump_abs,jump_rel*z[aa]))
            good &= fg[ids[cc]]&(abs(z[cc]-z[aa])<=np.maximum(same_abs,same_rel*z[aa]))
            if good.any():triples.append(np.column_stack((aa[good],bb[good],cc[good])))
    t=np.concatenate(triples,axis=0) if triples else np.empty((0,3),int)
    a,b,c=t.T
    width=np.linalg.norm(uv[b]-uv[a],axis=1)
    n=(uv[b]-uv[a])/width[:,None]
    return dict(point_indices=ids[a],background_indices=ids[b],support_indices=ids[c],
                xyz_lidar=xyz[ids[a]],background_xyz_lidar=xyz[ids[b]],
                uv=uv[a],background_uv=uv[b],support_uv=uv[c],
                z_m=z[a],background_z_m=z[b],depth_jump_m=z[b]-z[a],
                inverse_depth_jump=1/z[a]-1/z[b],normal_uv=n,
                plane_normal=plane_normals(uv[a],n,K),bracket_width_px=width)

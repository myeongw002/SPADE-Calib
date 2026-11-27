"""Read actual COLMAP observations and fixed intrinsics for local BA."""
from pathlib import Path
import struct
import numpy as np

# Supported perspective COLMAP models; fail closed for other models.
MODELS={0:("SIMPLE_PINHOLE",3),1:("PINHOLE",4),2:("SIMPLE_RADIAL",4),
        3:("RADIAL",5),4:("OPENCV",8),5:("OPENCV_FISHEYE",8),
        6:("FULL_OPENCV",12),7:("FOV",5),8:("SIMPLE_RADIAL_FISHEYE",4),
        9:("RADIAL_FISHEYE",5),10:("THIN_PRISM_FISHEYE",12)}
SUPPORTED={"SIMPLE_PINHOLE","PINHOLE","SIMPLE_RADIAL","RADIAL","OPENCV"}


def _read(f,n):
    b=f.read(n)
    if len(b)!=n: raise ValueError("Truncated COLMAP model")
    return b


def load_cameras(path):
    path=Path(path);out={}
    if path.suffix==".bin":
        with path.open("rb") as f:
            count=struct.unpack("<Q",_read(f,8))[0]
            for _ in range(count):
                cid,mid,w,h=struct.unpack("<iiQQ",_read(f,24))
                if mid not in MODELS: raise ValueError(f"Unsupported camera model ID {mid}")
                name,n=MODELS[mid]
                out[cid]=(name,np.array(struct.unpack("<"+"d"*n,_read(f,8*n))))
    else:
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"): continue
            a=line.split();out[int(a[0])]=(a[1],np.array(a[4:],float))
    return out


def load_observations(path,wanted):
    """Return image ID -> (camera ID, landmark IDs, pixel XY); matched only."""
    path=Path(path);out={};wanted=set(wanted)
    def put(i,c,pids,uv):
        if i in wanted:
            m=(pids>=0)&np.isfinite(uv).all(axis=1)
            out[i]=(c,pids[m],uv[m])
    if path.suffix==".bin":
        dt=np.dtype([("x","<f8"),("y","<f8"),("pid","<i8")])
        with path.open("rb") as f:
            count=struct.unpack("<Q",_read(f,8))[0]
            for _ in range(count):
                v=struct.unpack("<i4d3di",_read(f,64))
                while _read(f,1)!=b"\x00": pass
                n=struct.unpack("<Q",_read(f,8))[0]
                if n>(path.stat().st_size-f.tell())//24: raise ValueError("Invalid observation count")
                if v[0] not in wanted: f.seek(n*24,1);continue
                a=np.frombuffer(_read(f,n*24),dtype=dt)
                put(v[0],v[-1],a["pid"],np.column_stack((a["x"],a["y"])))
    else:
        with path.open() as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"): continue
                a=line.split(maxsplit=9);i,c=int(a[0]),int(a[8])
                row=f.readline()
                if not row: raise ValueError("Missing POINTS2D line")
                if i not in wanted: continue
                a=row.split()
                if len(a)%3: raise ValueError("Malformed POINTS2D")
                put(i,c,np.array(a[2::3],dtype=np.int64),
                    np.column_stack((np.array(a[0::3],float),np.array(a[1::3],float))))
    if set(out)!=wanted: raise ValueError("Missing selected COLMAP images")
    return out


def camera_parameters(model):
    name,p=model;p=np.asarray(p,float)
    counts={n:k for n,k in MODELS.values()}
    if name not in SUPPORTED or len(p)!=counts[name] or not np.isfinite(p).all():
        raise ValueError(f"Unsupported/invalid BA camera: {name}")
    if name in {"PINHOLE","OPENCV"}:
        fx,fy,cx,cy=p[:4];d=p[4:]
    else:
        fx=fy=p[0];cx,cy=p[1:3];d=p[3:]
    if min(fx,fy)<=0: raise ValueError("Invalid focal length")
    k=np.zeros(4);k[:len(d)]=d
    return np.r_[fx,fy,cx,cy,k]


def project(xyz,params):
    """Pinhole/radial/OpenCV projection; smooth positive-depth barrier separate."""
    z=xyz[:,2];x,y=(xyz[:,:2]/np.maximum(z[:,None],.05)).T
    fx,fy,cx,cy,k1,k2,p1,p2=params.T
    r2=x*x+y*y;rad=1+k1*r2+k2*r2*r2
    return np.column_stack((fx*(x*rad+2*p1*x*y+p2*(r2+2*x*x))+cx,
                            fy*(y*rad+p1*(r2+2*y*y)+2*p2*x*y)+cy))


def unique_view_observations(point_indices, pixels, errors):
    """One observation per landmark in a view, chosen by initial pixel error.

    Call AFTER the positive-depth and reprojection gate. Preserve measured
    pixels (no averaging), choose earliest row on exact error ties, and
    return the number of removed duplicate observations. No GT is used.
    This is a fixed input-cleaning rule, not a claim of physical identity.
    """
    p=np.asarray(point_indices)
    uv=np.asarray(pixels,dtype=float)
    e=np.asarray(errors,dtype=float)
    if (p.ndim!=1 or p.dtype.kind not in "iu" or uv.shape!=(len(p),2)
            or e.shape!=(len(p),) or not np.isfinite(uv).all()
            or not np.isfinite(e).all() or np.any(e<0)):
        raise ValueError("Invalid per-view observations")
    order=np.lexsort((np.arange(len(p)),e,p))
    _,first=np.unique(p[order],return_index=True)
    keep=np.sort(order[first])
    return p[keep],uv[keep],int(len(p)-len(keep))

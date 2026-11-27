"""Subset a monocular COLMAP text model without refitting its geometry."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def read_text_model(path):
    path=Path(path)
    cameras={}
    for line in (path/"cameras.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            fields=line.split();cameras[int(fields[0])]=fields
    images={}
    with (path/"images.txt").open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):continue
            fields=line.strip().split(maxsplit=9)
            if len(fields)!=10:raise ValueError("Malformed image")
            obs=f.readline().split()
            if len(obs)%3:raise ValueError("Malformed image observations")
            name=fields[9].replace("\\","/").removeprefix("./")
            if name in images or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Duplicate/unsafe image name")
            images[name]=(fields,obs)
    points={}
    for line in (path/"points3D.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):continue
        v=line.split()
        if len(v)<8 or (len(v)-8)%2:raise ValueError("Malformed point track")
        points[int(v[0])]=v
    return cameras,images,points


def subset_text_model(model,names,out):
    """Preserve IDs, camera poses, point XYZ and 2D observation indices.

    Prune tracks to selected images, retain points with >=2 distinct views,
    set deleted point references to -1 (never renumber observations).
    """
    cameras,images,points=model;names=list(names);out=Path(out)
    if len(names)<2 or len(set(names))!=len(names) or not set(names)<=images.keys():
        raise ValueError("Need unique registered segment image names")
    selected=[images[n] for n in names]
    byid={int(v[0]):obs for v,obs in selected}
    if len(byid)!=len(selected):raise ValueError("Duplicate image ID")
    kept={}
    for pid,v in points.items():
        track=[]
        for i,j in zip(v[8::2],v[9::2]):
            iid,k=int(i),int(j)
            if iid not in byid:continue
            obs=byid[iid]
            if k<0 or 3*k+2>=len(obs) or int(obs[3*k+2])!=pid:
                raise ValueError(f"Inconsistent COLMAP track {pid}/{iid}/{k}")
            track.extend([i,j])
        if len(set(track[::2]))>=2:kept[pid]=v[:8]+track
    if not kept:raise ValueError("No sparse points with >=2 selected observations")
    # Validate every retained observation links back to a retained track.
    links={(int(i),int(j),pid) for pid,v in kept.items() for i,j in zip(v[8::2],v[9::2])}
    used={int(v[8]) for v,_ in selected}
    out.mkdir(parents=True,exist_ok=False)
    with (out/"cameras.txt").open("w") as f:
        for i in sorted(used):f.write(" ".join(cameras[i])+"\n")
    counts=[]
    with (out/"images.txt").open("w") as f:
        for v,obs in selected:
            obs=obs.copy();iid=int(v[0]);n=0
            for k in range(2,len(obs),3):
                pid=int(obs[k])
                if pid not in kept:obs[k]="-1"
                elif (iid,k//3,pid) not in links:
                    raise ValueError("Missing reciprocal track")
                else:n+=1
            counts.append(n)
            f.write(" ".join(v)+"\n"+" ".join(obs)+"\n")
    with (out/"points3D.txt").open("w") as f:
        for v in kept.values():f.write(" ".join(v)+"\n")
    return dict(images=len(names),points=len(kept),observations=sum(counts),
                min_observations_per_image=min(counts),cameras=len(used))


def anchor_metric(xyz,Rcw,tcw,scale):
    """World SfM gauge -> anchor camera metric, with t_cw (NOT center)."""
    p=np.asarray(xyz,float);R=np.asarray(Rcw,float);t=np.asarray(tcw,float)
    if (p.ndim!=2 or p.shape[1]!=3 or R.shape!=(3,3) or t.shape!=(3,)
        or not np.isfinite(p).all() or not np.isfinite(R).all() or not np.isfinite(t).all()
        or not np.isfinite(scale) or scale<=0
        or not np.allclose(R@R.T,np.eye(3),atol=1e-6) or np.linalg.det(R)<0):
        raise ValueError("Invalid anchor transform/scale")
    return scale*(p@R.T+t)

"""S2: fixed visual-state multiview line controls.

Same physical tracks, same anchor-local metric reference gauge, equal
weight per track, independent of number of accepted observation frames.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from spade_calib.calibration.crossmodal_lines import sample_line,image_line
from spade_calib.calibration.segmented_ground import _robust
from spade_calib.calibration.ba_diagnostics import scale_coupling_information


def prepare_tracks(segments,manual,reviewed):
    anchors={int(s.frames[s.anchor]):k for k,s in enumerate(segments)}
    records={r["id"]:r for r in manual}
    if len(records)!=len(manual):raise ValueError("Duplicate M0 records")
    groups={};seen=set();skipped=[]
    for r in reviewed:
        rid=r["source_id"]
        if rid not in records or not r["manual_identity_confirmed"]:
            raise ValueError("Unknown or unconfirmed reviewed line")
        src=records[rid];f=int(r["frame"]);anchor=int(src["frame"])
        if (int(r["anchor_frame"])!=anchor or r["split"]!=src["split"] or
                not np.array_equal(r["xyz_lidar"],src["xyz_lidar"]) or not src["manual_identity_confirmed"]):
            raise ValueError("Reviewed line does not match M0")
        if src["split"] not in ("train","test"):raise ValueError("Invalid split")
        key=(rid,f)
        if key in seen:raise ValueError("Duplicate track/frame")
        seen.add(key);image_line(r["uv_image"])
        if anchor not in anchors:raise ValueError("M0 frame is not a BA segment anchor")
        k=anchors[anchor];s=segments[k]
        if f==anchor:
            skipped.append(dict(id=r["id"],reason="manual_M0_anchor_used"));continue
        index=np.flatnonzero(s.frames==f)
        if len(index)!=1:
            skipped.append(dict(id=r["id"],reason="not_in_same_archived_BA_support"));continue
        groups.setdefault(rid,[]).append(dict(frame=f,view=int(index[0]),uv=r["uv_image"]))
    tracks=[]
    for rid,views in sorted(groups.items()):
        src=records[rid];k=anchors[int(src["frame"])];s=segments[k]
        # Use original manually confirmed anchor image line in every arm.
        image_line(src["uv_image"])
        tracks.append(dict(id=rid,segment=k,split=src["split"],xyz=sample_line(src["xyz_lidar"]),
            views=[dict(frame=int(src["frame"]),view=s.anchor,uv=src["uv_image"])]+sorted(views,key=lambda r:r["frame"])))
    return tracks,skipped


def camera_points(p,X,R,t,ratio):
    return (p@X[:3,:3].T+X[:3,3]-ratio*t)@R


def glim_points(p,X,R,t):
    return ((p-t)@R)@X[:3,:3].T+X[:3,3]


def raw_line(q,K,uv):
    z=q[:,2];pixel=(q@K.T)[:,:2]/np.maximum(z[:,None],.1)
    l=image_line(uv)
    return pixel@l[:2]+l[2],np.maximum(.5-z,0)*100.


def line_rows(segments,states,tracks,X,ratios,K,mode,sigma=2.,robust=True):
    if mode not in ("anchor","glim","camera"):raise ValueError("Unknown line mode")
    blocks=[];details=[]
    for tr in tracks:
        k=tr["segment"];s=segments[k];R,t,_=states[k]
        # Ra/ta rows are exactly in s.centers order, anchor identity.
        lookup={int(j):i for i,j in enumerate(s.centers)}
        views=tr["views"][:1] if mode=="anchor" else tr["views"]
        for view in views:
            j=view["view"];p=tr["xyz"]
            if mode=="camera":
                q=camera_points(p,X,R[j],t[j],ratios[k])
            elif j==s.anchor:
                q=p@X[:3,:3].T+X[:3,3]
            else:
                i=lookup[j];q=glim_points(p,X,s.Ra[i],s.ta[i])
            d,b=raw_line(q,K,view["uv"])
            r=d/sigma
            if robust:r=_robust(r,"soft_l1")
            blocks.append(np.column_stack((r,b)).ravel()/np.sqrt(len(tracks)*len(views)*len(p)))
            details.append(dict(id=tr["id"],segment=k,frame=view["frame"],
                                rmse_px=float(np.sqrt(np.mean(d*d))),near_samples=int(np.count_nonzero(b))))
    return np.concatenate(blocks) if blocks else np.empty(0),details


def report_lines(segments,states,tracks,X,ratios,K):
    out={}
    for mode in ("anchor","glim","camera"):
        _,rows=line_rows(segments,states,tracks,X,ratios,K,mode,robust=False)
        # Equal physical-track mass, same number of samples per observation.
        ids=sorted({r["id"] for r in rows})
        mse=[np.mean([r["rmse_px"]**2 for r in rows if r["id"]==i]) for i in ids]
        out[mode]=dict(tracks=len(ids),observations=len(rows),rmse_px=float(np.sqrt(np.mean(mse))) if mse else None,
                       near_samples=sum(r["near_samples"] for r in rows),per_observation=rows)
    return out


def solve_fixed(segments,states,tracks,X0,scales,he,start,K,*,kind="scene",mode="none",weight=0.,
                sigma=2.,motion_weight=1.,ground_weight=.1,sigma_scene=.15,sigma_ground=.15,
                max_log_scale=.15,max_nfev=120,fixed_scale=False):
    scales=np.asarray(scales,float);start=np.asarray(start,float).copy()
    n=len(scales)
    if (start.shape!=(6+n,) or len(segments)!=n or len(states)!=n or
            not np.isfinite(start).all() or not np.isfinite(scales).all() or np.any(scales<=0) or
            mode not in ("none","anchor","glim","camera") or (mode=="none")!=(weight==0)
            or weight<0 or (weight>0 and not tracks) or sigma<=0):
        raise ValueError("Invalid S2 settings")
    def unpack(x):
        X=X0.copy();X[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X0[:3,:3];X[:3,3]+=x[3:6]
        return X,np.exp(x[6:])
    def fun(x,raw=False,parts=False):
        X,ratio=unpack(x);blocks=[]
        for k,(s,(R,t,P)) in enumerate(zip(segments,states)):
            r=s.motion(R,t,X,ratio[k],kind,he)
            if kind=="scene":r=r/sigma_scene
            if not raw:
                if kind=="he":r=_robust(r,he.robust_loss)
                else:r=r*np.sqrt(2/(np.hypot(1,np.linalg.norm(r,axis=1))+1))[:,None]
            blocks.append(r.ravel()*np.sqrt(motion_weight/(n*len(r))))
            if ground_weight>0:
                g=np.array([s.ground(P,X,ratio[k])/sigma_ground])
                blocks.append((g if raw else _robust(g,"soft_l1"))*np.sqrt(ground_weight/n))
        base=np.concatenate(blocks)
        line=np.empty(0)
        if weight>0:
            line,_=line_rows(segments,states,tracks,X,ratio,K,mode,sigma,not raw)
            line*=np.sqrt(weight)
        return (base,line) if parts else np.r_[base,line]
    lo=-np.r_[np.full(3,np.deg2rad(5)),np.full(3,1.),np.full(n,max_log_scale)];hi=-lo
    if np.any(start<lo-1e-8) or np.any(start>hi+1e-8):raise ValueError("Archived start outside original bounds")
    start=np.clip(start,lo,hi)
    mask=np.ones(6+n,bool)
    if fixed_scale:mask[6:]=False
    def expand(x):
        z=start.copy();z[mask]=x;return z
    # Same-state local information at archived H state, all scales treated
    # free for comparison even for the fixed-scale diagnostic arm.
    def jac(raw):
        h=1e-5
        return np.column_stack([(fun(start+d,raw)-fun(start-d,raw))/(2*h) for d in np.eye(len(start))*h])
    info={label:scale_coupling_information(csr_matrix(jac(raw)),n,start[:3])
          for label,raw in (("raw",True),("robust",False))}
    sol=least_squares(lambda x:fun(expand(x)),start[mask],bounds=(lo[mask],hi[mask]),
                      x_scale="jac",max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    final=expand(sol.x)
    def report(x):
        X,ratio=unpack(x);base,line=fun(x,parts=True)
        return dict(T_camera_lidar=X.tolist(),scales=(ratio*scales).tolist(),base_cost=float(.5*(base@base)),
                    line_cost=float(.5*(line@line)),cost=float(.5*(base@base+line@line)),
                    lines=report_lines(segments,states,tracks,X,ratio,K))
    initial=report(start);end=report(final)
    depth_ok=mode=="none" or end["lines"][mode]["near_samples"]==0
    return dict(success=bool(sol.success and depth_ok),solver_converged=bool(sol.success),nfev=int(sol.nfev),
                status=int(sol.status),message=str(sol.message),bound_hit=bool(np.any(sol.active_mask)),
                active_indices=np.flatnonzero(mask)[sol.active_mask!=0].tolist(),
                mode=mode,weight=weight,fixed_scale=fixed_scale,initial=initial,final=end,
                same_state_information=info,
                warning="Fixed archived camera poses/landmarks, visual cost omitted as constant. Information at common archived state, not covariance. Cross-mode line costs not directly comparable.")

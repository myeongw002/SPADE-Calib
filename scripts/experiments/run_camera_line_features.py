#!/usr/bin/env python3
"""E1: camera-only ELSED / relative-depth / normal diversity pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from spade_calib.camera.line_features import (
    line_geometry, depth_profiles, select_arms, geometry_metrics,
)

DEFAULT_FRAMES = (69, 189, 289, 389, 509, 659, 819, 995)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_intrinsics(path, camera_index=2):
    # Read P2 (or requested camera projection) ONLY, never Tr or baseline.
    key = f"P{camera_index}"
    matches = []
    for line in path.read_text().splitlines():
        name, sep, values = line.partition(":")
        if sep and name.strip() == key:
            p = np.array([float(v) for v in values.split()])
            if p.shape != (12,):
                raise ValueError(f"Invalid {key} in {path}")
            matches.append(p.reshape(3, 4)[:, :3])
    if len(matches) != 1:
        raise ValueError(f"Expected one {key} in {path}")
    K = matches[0]
    line_geometry(np.array([[[0., 0.], [1., 0.]]]), K)
    return K


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save {path}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def draw(image, segments, ids, selected, label_ids=True):
    result = image.copy()
    for i in selected:
        p, q = np.rint(segments[i]).astype(int)
        cv2.line(result, tuple(p), tuple(q), (0, 255, 0), 1 if not label_ids else 2,
                 cv2.LINE_AA)
        if label_ids:
            m = tuple(np.rint(segments[i].mean(axis=0)).astype(int))
            cv2.putText(result, str(ids[i]), m, cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, str(ids[i]), m, cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def panel(image, label):
    out = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def depth_visual(depth):
    lo, hi = np.percentile(depth, [5, 95])
    u = np.clip((depth-lo) / max(hi-lo, 1e-8), 0, 1)
    return cv2.applyColorMap(np.rint(u*255).astype(np.uint8), cv2.COLORMAP_TURBO)


def finite_or_none(x):
    return float(x) if np.isfinite(x) else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True, help="KITTI calib.txt, P camera intrinsics only")
    p.add_argument("--camera-index", type=int, default=2)
    p.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--min-length", type=float, default=40.)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--elsed-dir", type=Path, help="Optional camera-only archived frame_*/original.png and elsed_segments.npz")
    depth = p.add_mutually_exclusive_group()
    depth.add_argument("--depth-dir", type=Path, help="Raw original-resolution relative inverse-depth .npy")
    depth.add_argument("--checkpoint", type=Path, help="Official DA2 relative checkpoint")
    p.add_argument("--da2-repo", type=Path)
    p.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--input-size", type=int, default=518)
    a = p.parse_args(argv)
    frames = sorted(set(a.frames))
    if (not frames or min(frames) < 0 or a.seed < 0 or a.budget < 1
            or not np.isfinite(a.min_length) or a.min_length <= 0):
        p.error("Invalid frames/seed/budget/min-length")
    if a.checkpoint and (not a.checkpoint.is_file() or not a.da2_repo
                         or not (a.da2_repo/"depth_anything_v2"/"dpt.py").is_file()):
        p.error("Provide existing --checkpoint and official --da2-repo")
    K = read_intrinsics(a.calib, a.camera_index)
    for f in frames:
        paths = [a.image_dir/f"{f:06d}.png"]
        if a.elsed_dir:
            folder = a.elsed_dir/f"frame_{f:06d}"
            paths += [folder/"original.png", folder/"elsed_segments.npz"]
        if a.depth_dir:
            paths += [a.depth_dir/f"{f:06d}.npy"]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    if not a.elsed_dir:
        try:
            import pyelsed
        except ImportError as exc:
            raise SystemExit("Install official ELSED (pyelsed), or provide --elsed-dir") from exc
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError(f"Use a fresh output: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    summary = dict(
        warning="Camera-only. No physical identity labels, metric depth, extrinsic, LiDAR or calibration information. "
                "Depth support is not ground truth; RGB and predicted depth are correlated. "
                "Normal diversity is a 2D proxy. Thresholds and weights are pilot heuristics.",
        K=K.tolist(), camera_index=a.camera_index,
        settings=dict(budget=a.budget, min_length_px=a.min_length, seed=a.seed,
                      offsets_px=[3, 9], stations=17, min_valid_stations=5,
                      range_floor=.03, fusion_weight_floor=.25, grid=[6, 3]),
        detector="archive" if a.elsed_dir else str(pyelsed.__file__),
        depth_mode="external_unverified" if a.depth_dir else ("DA2" if a.checkpoint else "disabled"),
        frames={})
    rows = []
    for f in frames:
        ip = a.image_dir/f"{f:06d}.png"
        img = read_image(ip)
        h, w = img.shape[:2]
        folder = a.out/f"frame_{f:06d}"
        folder.mkdir()
        if a.elsed_dir:
            src = a.elsed_dir/f"frame_{f:06d}"
            if not np.array_equal(read_image(src/"original.png"), img):
                raise ValueError(f"Archived ELSED image mismatch at {f}")
            with np.load(src/"elsed_segments.npz", allow_pickle=False) as z:
                raw, scores = z["segments"], z["scores"]
        else:
            raw, scores = pyelsed.detect(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        raw = np.asarray(raw, dtype=float).reshape(-1, 2, 2)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(raw) or not np.isfinite(raw).all() or not np.isfinite(scores).all():
            raise ValueError(f"Invalid ELSED result at {f}")
        length = np.linalg.norm(raw[:, 1]-raw[:, 0], axis=1)
        inside = np.all((raw[..., 0]>=0)&(raw[..., 0]<=w-1)&
                        (raw[..., 1]>=0)&(raw[..., 1]<=h-1), axis=1)
        ids = np.flatnonzero((length >= a.min_length) & inside)
        s = raw[ids]
        length, normals = line_geometry(s, K)
        save_image(folder/"original.png", img)
        np.savez_compressed(folder/"elsed_segments.npz", segments=raw.reshape(-1,4), scores=scores)
        panels = [panel(img, "Original")]
        D = None
        meta = None
        if a.checkpoint:
            from spade_calib.camera.depth_inference import infer_da2
            D, meta = infer_da2(ip, a.checkpoint, encoder=a.encoder,
                               input_size=a.input_size, device=a.device, da2_repo=a.da2_repo)
        elif a.depth_dir:
            dp = a.depth_dir/f"{f:06d}.npy"
            D = np.load(dp, allow_pickle=False)
            meta = dict(source=str(dp), source_sha256=sha(dp), provenance_verified=False)
        profiles = None
        if D is not None:
            if D.shape != (h, w):
                raise ValueError(f"Depth shape mismatch at {f}: {D.shape} != {(h,w)}")
            profiles = depth_profiles(D, s)
            np.save(folder/"relative_depth.npy", D, allow_pickle=False)
            write_json(folder/"relative_depth.json", meta)
            dvis = depth_visual(D)
            save_image(folder/"depth_visual.png", dvis)
            panels.append(panel(dvis, "Relative inverse-depth proxy (visualization only)"))
        arms, pool = select_arms(length, normals, a.budget, a.seed+f, profiles)
        info = dict(image_sha256=sha(ip), elsed_sha256=sha(folder/"elsed_segments.npz"),
                    depth_sha256=sha(folder/"relative_depth.npy") if D is not None else None,
                    shape=[h,w], detected=len(raw), eligible=len(s),
                    common_ranked_pool=len(pool), arms={})
        memberships = {name: set(map(int, idx)) for name, idx in arms.items()}
        for name, idx in arms.items():
            m = geometry_metrics(s[idx], normals[idx], (h,w))
            m["selected_ids"] = ids[idx].tolist()
            if profiles is not None:
                valid = idx[np.isfinite(profiles[idx, :3]).all(axis=1)]
                m["valid_depth_count"] = len(valid)
                m["median_depth_support"] = float(np.median(profiles[valid,2])) if len(valid) else None
            info["arms"][name] = m
            overlay = draw(img, s, ids, idx, label_ids=name!="all")
            save_image(folder/f"{name}.png", overlay)
            eig = m["normal_eigenvalues"]
            label = f"{name}: {len(idx)} lines, min eig " + (f"{eig[0]:.4f}" if eig else "n/a")
            panels.append(panel(overlay, label))
        save_image(folder/"comparison.png", np.vstack(panels))
        for i, sid in enumerate(ids):
            d = profiles[i] if profiles is not None else [np.nan]*4
            row = dict(frame=f, image_id=int(sid), u0=float(s[i,0,0]), v0=float(s[i,0,1]),
                       u1=float(s[i,1,0]), v1=float(s[i,1,1]), length_px=float(length[i]),
                       elsed_score=float(scores[sid]), nx=float(normals[i,0]),
                       ny=float(normals[i,1]), nz=float(normals[i,2]),
                       jump=finite_or_none(d[0]), crease=finite_or_none(d[1]),
                       depth_support=finite_or_none(d[2]), valid_fraction=finite_or_none(d[3]),
                       arms=";".join(name for name in arms if i in memberships[name]),
                       geometry_label="", localization_label="", notes="")
            rows.append(row)
        summary["frames"][str(f)] = info
        write_json(folder/"summary.json", info)
        print(f"{f:06d}: {len(s)} eligible, {len(pool)} common pool, "
              f"{len(arms['length'])} per ranked arm", flush=True)
    keys = ["frame","image_id","u0","v0","u1","v1","length_px","elsed_score","nx","ny","nz",
            "jump","crease","depth_support","valid_fraction","arms",
            "geometry_label","localization_label","notes"]
    for name, data in (("features.csv", rows), ("review.csv", [r for r in rows if r["arms"]!="all"])):
        with (a.out/name).open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    names = next(iter(summary["frames"].values()))["arms"]
    summary["aggregate"] = {}
    for name in names:
        ms = [x["arms"][name] for x in summary["frames"].values()]
        e = [x["normal_eigenvalues"][0] for x in ms if x["normal_eigenvalues"] is not None]
        summary["aggregate"][name] = dict(
            count=sum(x["count"] for x in ms), nonempty_frames=len(e),
            mean_proxy_min_eigenvalue=float(np.mean(e)) if e else None,
            mean_grid_occupancy=float(np.mean([x["grid_occupancy"] for x in ms])))
    write_json(a.out/"summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

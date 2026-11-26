# S2 matching pilot

Reuse M0 manual 3D lines and existing ELSED pilot arrays, no new detector,
no CLAHE, no GT, no BA. GLIM world-from-LiDAR poses transform each
anchor line into each observation frame, then frozen M0 X projects it.
Near-plane and image clipping; no occlusion inference. Gate 20px mean
perpendicular endpoint distance, 10deg unoriented angle, 30% visible
projection overlap, minimum 15px. Equal normalized costs, ambiguity
gap .1; duplicate automatic assignments rejected. Thresholds provisional.
Each source record ID is the physical-track key, not the reused feature
name. Neighboring views are independently matched to the same 3D line;
this is not an independent temporal identity check.

Run prepare using original M0 and edge directories plus original image
directory. See --help. Writes matches.json and per-observation PNG
(original above overlay), review.csv. Red projection, green auto proposal,
yellow alternatives. Candidate labels are ELSED array indices. Review
each row: decision accept/reject/skip; chosen_index blank accepts only
an unambiguous proposed index, otherwise specify a saved candidate index.
No candidate can be manufactured for a failed gate; skip/reject it.
Finalize writes separately reviewed associations; records retain
anchor-frame 3D coordinates and observation-frame 2D endpoints.
Do not feed directly to M1 (different coordinate semantics).
GLIM hash and current segment-array hashes recorded; original pilot
did not hash segment arrays, so historical array integrity is not proven.
Manual identity review is essential; no matching accuracy claim before it.

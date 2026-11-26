# MV7 — visibility-only local surface reconstruction

Run fixed MV6 U0.2, W21, same anchor/GLIM/ghost filter and projection.
    python scripts/experiments/run_surface_visibility.py \
      --ring-calib /actual/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
      --out outputs/edgepoints07_MV7
Optional --frames 389 for pilot. Defaults --save-mode review, --el-step .2,
--bin-deg .2, same FOV and range thresholds as MV6. All thresholds fixed
before data analysis; adjustable for separate sensitivity tests.

Raw: exact U0.2 H/V/HV re-run; MV1 controls. Splat: minimum range of actual
grid returns within .3deg Euclidean azimuth/elevation angle-plane distance
from each query return (not spherical solid-angle distance). Surface:
four triangles (choose 3 of the four corner cells at +/-1 row/col around
the query cell) using actual XYZ; intersect actual unit ray from anchor,
require intersection inside triangle, all edges <=.5m, nondegenerate,
abs(normal dot ray) >=.15. Minimum valid positive range, unknown=inf.
Local stencil is deliberately conservative and is not a full mesh, plane
fit, or complete occlusion solution. Triangle may still bridge a real gap.
Splat and Surface computed independently from the same original nearest
grid, not iterative or combined. All eligible raw returns are queried,
including non-winners. Reject if r > predicted range + .5m, except direct
target scan returns protected (sensor offsets/finite sampling make this
an approximate constraint). Unknown retained. Rebuild actual-return-only
grid after each mask; no interpolated depths ever enter edge extraction.
Same MV6 H/V rules; removed bins can become unknown, which existing
neighbor rules can skip. Camera reference depth kept identical (EP0),
not rerendered, so evaluation isolates anchor visibility intervention.

Report supported rays, conflicts, protected conflicts, removed returns,
raw grid lost/replaced cells, H/V/HV candidate counts, and fixed raw
hypothesis invalidation (any removed near/far/back); lost near ID means
all raw hypotheses invalid. Also report new/lost near IDs after re-extract.
Separate projected fixed-surviving and fixed-lost near sets; independent
projection means pixel counts need not add exactly. DA2 proximity not
physical precision; no ground truth visibility labels or optimization.

minimal: per-frame and root JSON only; review (default): JSON, compact
source features, directional triples, source IDs and projected NPZ, one
JPEG comparison per frame; full: also raw cloud masks and grid NPZ and
individual overlays. No repeated camera overlays. In review, full grid
not needed to assess primary metrics. Source feature depth arrays may
contain inf for unknown. No changes to previous runners.

# Canny vs ELSED pilot

Use original rectified camera images, M0 manual image lines only.
Install official Python wrapper separately (not a mandatory project dependency):
```bash
sudo apt install -y libopencv-dev build-essential cmake pkg-config
python -m pip install setuptools
python -m pip install git+https://github.com/iago-suarez/ELSED.git
python -c "import pyelsed; print(pyelsed.__file__)"
```
Run:
```bash
KITTI=/media/myungw00/256GB_SSD/kitti
python scripts/experiments/run_edge_detector_pilot.py \
  --data outputs/lines07_M0_ghost \
  --image-dir "$KITTI/data_odometry_color/sequences/07/image_2" \
  --out outputs/edge07_canny_elsed
```
For anchors only omit image-dir and pass --offsets 0.
Same grayscale image, full resolution, Canny 50/150 L2 gradient,
ELSED official defaults, no filtering or Hough fitting. Rasterize ELSED
at one-pixel width; nearest detected pixel to <=1px samples on each
manual line, coverage at 1/2/3/5px, median/p90/RMSE, longest uncovered
span approximation and local band pixel count. Nearby wrong edges can
increase coverage; sparse annotations cannot measure precision or actual
false positive rate. Adjacent frames +/-5,10 only visual inspection,
not repeatability score. Original image directory checked against M0
anchor pixels. Timing median after warmup, detector-only (excludes
ELSED rasterization, image I/O and scoring); OpenCV one thread, ELSED
internal thread count not controlled. Save raw segment coordinates/scores,
masks, overlays and stacked comparison.png per frame, summary.json.
No matching or extrinsic optimization. Pilot only; do not tune on three
selected corners and claim generalization. Record package provenance.

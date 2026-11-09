"""Opt-in D1 artifact-shape checks; never require private data on public CI."""

from __future__ import annotations

import os

import pytest

from spade_calib.io.dataset import inspect_dataset, load_dataset_config


@pytest.mark.skipif(
    not os.getenv("SPADE_D1_CONFIG"),
    reason="SPADE_D1_CONFIG not set; original D1 data are private",
)
def test_recovered_d1_artifact_contract() -> None:
    config = load_dataset_config(os.environ["SPADE_D1_CONFIG"])
    report = inspect_dataset(config)
    artifacts = report["artifacts"]
    assert artifacts["joint_cloud"]["point_count"] == 215504
    assert artifacts["pgo_cloud"]["point_count"] == 215504
    assert artifacts["camera_planes"]["plane_count"] == 24
    assert artifacts["camera_masks"]["mask_shape"] == [180, 320]
    assert artifacts["sfm_anchors"]["anchor_count"] == 6050
    assert artifacts["metric_depth"]["shape"] == [720, 1280]

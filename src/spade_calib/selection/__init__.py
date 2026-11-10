"""Trajectory observability proxy and scene-selection API."""
from .trajectory import (
    SelectionSettings, MatchedTrajectories, match_trajectories,
    load_trajectory_data, from_recording_index, build_motion_pairs,
    select_scenes, write_scene_selection,
)
__all__ = [
    'SelectionSettings','MatchedTrajectories','match_trajectories',
    'load_trajectory_data','from_recording_index','build_motion_pairs',
    'select_scenes','write_scene_selection',
]
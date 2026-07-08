"""Shared tracking pipeline.

Extracted from ``main.py`` so it can be reused by both the output-video renderer
(``main.py``) and the tracking editor (``editor/generate.py``). ``run_pipeline``
runs the full detect -> track -> position -> camera-movement -> view-transform ->
ball-interpolation -> team-assignment sequence and returns the raw pieces the
callers need. Rendering / ball-control / speed metrics stay in ``main.py``.

Stub caching is keyed to the input video filename so that processing a different
video never silently serves another video's cached tracks. The sample video's
stubs were copied to this naming convention (``stubs/<stem>_tracks.pkl`` /
``stubs/<stem>_camera.pkl``) so the sample workflow still runs without GPU
inference.
"""

import os

from utils import read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def _video_stem(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def track_stub_path(video_path):
    return os.path.join("stubs", f"{_video_stem(video_path)}_tracks.pkl")


def camera_stub_path(video_path):
    return os.path.join("stubs", f"{_video_stem(video_path)}_camera.pkl")


def run_pipeline(video_path, model_path="models/best.pt", read_from_stub=True):
    """Run the tracking pipeline for ``video_path``.

    Returns ``(video_frames, tracks, camera_movement_per_frame)`` where
    ``tracks`` has been populated with per-frame ``position``,
    ``position_adjusted``, ``position_transformed`` and per-player ``team`` /
    ``team_color`` fields, and the ball bbox has been interpolated across frames.
    """
    # Read Video
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=read_from_stub,
                                       stub_path=track_stub_path(video_path))

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=camera_stub_path(video_path))
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    return video_frames, tracks, camera_movement_per_frame

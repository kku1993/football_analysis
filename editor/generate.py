"""Phase 1: produce clean frame JPEGs + a seed pixel-space ``annotations.json``.

Run as::

    venv/bin/python -m editor.generate [--video <path>] [--output-dir <dir>] [--force]

``--video`` defaults to the sample video referenced by ``main.py`` so the sample
workflow needs no arguments. Nothing about the video (path, resolution, fps,
frame count) is hardcoded downstream: it is measured here and recorded in
``annotations.json``, from which the server, frontend and exporter read it.

Output goes to ``editor_data-<video filename>/`` by default (``--output-dir``
overrides this), so multiple videos can be generated and edited side by side
without one clobbering another's frames/annotations. Point ``editor.server``
at the resulting directory to edit it.

The frames written here are *clean* (no boxes baked in); the editor draws boxes
as canvas overlays from ``annotations.json`` so they stay editable.
"""

import argparse
import json
import os

import cv2

from pipeline import run_pipeline
from view_transformer import (
    DEFAULT_PIXEL_VERTICES,
    DEFAULT_TARGET_VERTICES,
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
)

# Team integer (from the pipeline) -> schema enum. Established in the deleted
# tracking_output.py; recover with `git show 5ad880f:tracking_output.py`.
TEAM_NAME = {1: "Defence", 2: "Offence"}

DEFAULT_VIDEO = "input_videos/arg-egy-14_57-goal.mp4"

# Legacy flat default, still used as export.py's fallback --annotations path.
# generate() itself defaults to a per-video directory (see default_data_dir).
EDITOR_DATA_DIR = "editor_data"
FRAMES_DIR = os.path.join(EDITOR_DATA_DIR, "frames")
ANNOTATIONS_PATH = os.path.join(EDITOR_DATA_DIR, "annotations.json")


def default_data_dir(video_path):
    """The editor_data directory a video generates into, named after it so
    several videos can be worked on side by side (each editor.server instance
    points at one) without clobbering each other's frames/annotations.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"editor_data-{stem}"


def default_calibration():
    """The seed calibration block written into ``annotations.json``.

    Mirrors the per-sample hardcoded values that ``ViewTransformer`` used to
    carry alone, so a freshly-generated file reproduces the existing export
    output exactly. The human is expected to refine this in the editor (the
    pixel trapezoid only spans ~23 m of the pitch; recalibrating to span more
    -- or to a different camera view -- is the whole point of editing it).
    """
    return {
        "pitch_length": DEFAULT_PITCH_LENGTH,
        "pitch_width": DEFAULT_PITCH_WIDTH,
        "points": [
            {"pixel": list(DEFAULT_PIXEL_VERTICES[i]),
             "pitch": list(DEFAULT_TARGET_VERTICES[i])}
            for i in range(len(DEFAULT_PIXEL_VERTICES))
        ],
    }

# Zero-padding width for frame filenames (supports up to 99,999 frames). Frames
# are indexed by integer everywhere; filenames are never parsed back to indices.
FRAME_PAD = 5


def _video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


def _seed_frame_annotation(players_track, ball_track):
    """Build one frame's seed annotation object (pixel space) from tracks."""
    players = []
    for tid, info in players_track.items():
        team_int = info.get("team")
        team = TEAM_NAME.get(team_int, "Defence")
        players.append({
            "track_id": str(tid),
            "team": team,
            "role": "Outfield",
            "bbox": [float(v) for v in info["bbox"]],
        })

    ball = None
    ball_info = ball_track.get(1)
    if ball_info and ball_info.get("bbox") and len(ball_info["bbox"]) == 4:
        ball = {
            "bbox": [float(v) for v in ball_info["bbox"]],
            "state": "Active",
            "kick": None,
        }

    return {"players": players, "ball": ball}


def generate(video_path, force=False, output_dir=None):
    data_dir = output_dir or default_data_dir(video_path)
    frames_dir = os.path.join(data_dir, "frames")
    annotations_path = os.path.join(data_dir, "annotations.json")

    if os.path.exists(annotations_path) and not force:
        raise SystemExit(
            f"{annotations_path} already exists; refusing to overwrite human "
            f"edits. Pass --force to regenerate.")

    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    print(f"Running pipeline on {video_path} ...")
    video_frames, tracks, _camera = run_pipeline(video_path)

    frame_count = len(video_frames)
    height, width = video_frames[0].shape[:2]
    fps = _video_fps(video_path)

    os.makedirs(frames_dir, exist_ok=True)

    print(f"Writing {frame_count} clean frames to {frames_dir}/ ...")
    frames_out = []
    for i, frame in enumerate(video_frames):
        fname = os.path.join(frames_dir, f"frame_{i:0{FRAME_PAD}d}.jpg")
        cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frames_out.append(_seed_frame_annotation(tracks["players"][i], tracks["ball"][i]))

    annotations = {
        "video": video_path,
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": frame_count,
        "frames": frames_out,
        "calibration": default_calibration(),
    }

    tmp = annotations_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(annotations, fh, indent=2)
    os.replace(tmp, annotations_path)

    print(f"Wrote {annotations_path} ({frame_count} frames, {width}x{height}, {fps:.2f} fps).")
    print(f"Edit it with: venv/bin/python -m editor.server {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate editor frames + seed annotations.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write frames/ and annotations.json into. "
                             "Defaults to 'editor_data-<video filename>'.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing annotations.json.")
    args = parser.parse_args()
    generate(args.video, force=args.force, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

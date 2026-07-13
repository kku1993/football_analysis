"""Step 1 of the editor pipeline: video -> per-frame frames + bounding boxes.

Given a broadcast video, this:

1. Resamples the video to a target frame rate (default 12 fps) by selecting the
   native frames closest to each output timestamp.
2. Runs the CV pipeline (YOLO detection + ByteTrack tracking + team clustering +
   ball interpolation) on the *resampled* frames.
3. Writes an output directory named after the video stem:

       <stem>/
         fps.txt                      # the fps used for extraction
         frames/frame_001.png ...     # the resampled frames
         bbox/frame_001_bbox.json ... # player + ball boxes per bbox-schema.json

Referees are intentionally dropped -- we only track players and the ball.

The tracker's raw output is cached to ``stubs/<stem>_<fps>fps_tracks.pkl`` so
re-running (e.g. after tweaking the writer) is instant and does not re-run
inference. Pass ``--no-stub`` to force a fresh detection pass.

Usage:
    python -m editor.generate input_videos/eng-nor-49-offside-no-goal.mp4 [--fps 12]
"""

import argparse
import json
import os
import pickle

import cv2

from trackers import Tracker
from team_assigner import TeamAssigner

DEFAULT_FPS = 12.0
TEAM_LABEL = {1: "A", 2: "B"}


def video_stem(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def resample_frames(video_path, target_fps):
    """Read ``video_path`` and return the frames nearest each 1/target_fps tick.

    Returns ``(frames, native_fps)``. The output is the video "converted" to
    ``target_fps``: for output index ``i`` we take the native frame nearest to
    timestamp ``i / target_fps``. When the target fps meets or exceeds the
    native rate every native frame is kept.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps

    native = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        native.append(frame)
    cap.release()
    if not native:
        raise SystemExit(f"no frames decoded from {video_path}")

    duration = len(native) / native_fps
    n_out = max(1, int(round(duration * target_fps)))
    frames = []
    for i in range(n_out):
        src = int(round(i * native_fps / target_fps))
        if src >= len(native):
            break
        frames.append(native[src])
    return frames, native_fps


def build_tracks(frames, model_path, stub_path, read_from_stub):
    """Detect + track players and ball, cluster teams, interpolate the ball."""
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, read_from_stub=read_from_stub,
                                       stub_path=stub_path)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    # Seed the two team colors from the first frame that actually has players.
    seed = next((pf for pf in tracks["players"] if pf), None)
    seed_frame = next((frames[i] for i, pf in enumerate(tracks["players"]) if pf), None)
    if seed is not None:
        team_assigner.assign_team_color(seed_frame, seed)
        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                track["team"] = team_assigner.get_player_team(
                    frames[frame_num], track["bbox"], player_id)
    return tracks


def bbox_record(player_track, ball_track):
    """One frame's bbox JSON conforming to bbox-schema.json."""
    players = []
    for player_id, track in sorted(player_track.items(), key=lambda kv: int(kv[0])):
        x1, y1, x2, y2 = track["bbox"]
        players.append({
            "id": str(int(player_id)),
            "team": TEAM_LABEL.get(track.get("team", 1), "A"),
            "x_min": round(float(x1), 2),
            "y_min": round(float(y1), 2),
            "x_max": round(float(x2), 2),
            "y_max": round(float(y2), 2),
        })

    record = {"players": players}
    ball_bbox = ball_track.get(1, {}).get("bbox") if ball_track else None
    if ball_bbox and len(ball_bbox) == 4 and all(v == v for v in ball_bbox):
        x1, y1, x2, y2 = ball_bbox
        record["ball"] = {
            "x_min": round(float(x1), 2),
            "y_min": round(float(y1), 2),
            "x_max": round(float(x2), 2),
            "y_max": round(float(y2), 2),
        }
    return record


def write_output(out_dir, frames, tracks, fps):
    frames_dir = os.path.join(out_dir, "frames")
    bbox_dir = os.path.join(out_dir, "bbox")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    with open(os.path.join(out_dir, "fps.txt"), "w") as f:
        f.write(f"{fps}\n")

    for i, frame in enumerate(frames):
        name = f"frame_{i + 1:03d}"
        cv2.imwrite(os.path.join(frames_dir, f"{name}.png"), frame)
        record = bbox_record(tracks["players"][i], tracks["ball"][i])
        with open(os.path.join(bbox_dir, f"{name}_bbox.json"), "w") as f:
            json.dump(record, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to the input video file")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help=f"frame rate for extraction (default {DEFAULT_FPS})")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <video stem> in cwd)")
    ap.add_argument("--model", default="models/best.pt", help="YOLO weights path")
    ap.add_argument("--no-stub", action="store_true",
                    help="ignore any cached tracks and re-run detection")
    args = ap.parse_args()

    stem = video_stem(args.video)
    out_dir = args.out or stem

    print(f"resampling {args.video} to {args.fps} fps ...")
    frames, native_fps = resample_frames(args.video, args.fps)
    print(f"native fps {native_fps:.2f} -> {len(frames)} frames at {args.fps} fps")

    os.makedirs("stubs", exist_ok=True)
    stub_path = os.path.join("stubs", f"{stem}_{args.fps:g}fps_tracks.pkl")
    print(f"running CV pipeline (stub: {stub_path}) ...")
    tracks = build_tracks(frames, args.model, stub_path,
                          read_from_stub=not args.no_stub)

    print(f"writing output to {out_dir}/ ...")
    write_output(out_dir, frames, tracks, args.fps)
    print(f"done: {out_dir}/ ({len(frames)} frames)")


if __name__ == "__main__":
    main()

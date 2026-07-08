"""Phase 4: export pixel-space annotations.json -> schema-conformant JSON.

Run as::

    venv/bin/python -m editor.export [-o output_videos/tracking_data.json]

Converts the editor's pixel-space working file (``editor_data/annotations.json``)
into the metric JSON described by ``tracking-schema.json``. Coordinates are in
meters from the pitch center: (0,0) = center, -x left / +x right, -y up / +y down.

Adapted from the deleted ``tracking_output.py`` (``git show
5ad880f:tracking_output.py``), which already solved metric conversion,
center-origin offset and gap interpolation; here we read boxes from
``annotations.json`` instead of the pipeline's ``tracks``.

GENERALIZATION CAVEAT: the metric conversion is only as good as
``ViewTransformer.pixel_vertices``, a hardcoded calibration for the SAMPLE
video's camera view. The pixel-space editor (generate / server / frontend) is
fully video-agnostic, but this export step inherits the pipeline's per-video
calibration and must be recalibrated per camera setup, exactly as ``main.py``
already requires. We do not attempt auto-calibration.

Role limitation: goalkeepers were merged into "players" by the tracker with no
preserved flag, so ``GoalKeeper`` cannot be inferred here; every player is
exported with ``role: "Outfield"``.
"""

import argparse
import json
import os

import cv2
import numpy as np

from view_transformer import ViewTransformer
from camera_movement_estimator import CameraMovementEstimator
from pipeline import camera_stub_path
from editor.generate import ANNOTATIONS_PATH

DEFAULT_OUTPUT = os.path.join("output_videos", "tracking_data.json")


# --- metric helpers (from the recovered tracking_output.py) ----------------
def _raw_transform(view_transformer, point):
    """Perspective transform WITHOUT the inside-polygon filter.

    Yields real metric coordinates even for a slightly off-pitch object (e.g. an
    out-of-bounds ball) rather than returning None.
    """
    mat = getattr(view_transformer, "persepctive_trasnformer", None)
    if mat is None:
        mat = getattr(view_transformer, "perspective_transformer")
    src = np.array([[float(point[0]), float(point[1])]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, mat)
    return dst.reshape(-1, 2)[0].tolist()


def _center_offset(view_transformer):
    """(cx, cy) offset converting top-left-origin metric -> pitch-center origin."""
    verts = np.array(view_transformer.target_vertices, dtype=np.float32)
    xs, ys = verts[:, 0], verts[:, 1]
    return (float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0))


def _to_center_coords(point, offset):
    cx, cy = offset
    return (point[0] - cx, point[1] - cy)


def _interpolate_gaps(positions):
    """Linearly interpolate interior None gaps in a list of (x,y) | None.

    Leading/trailing Nones are left untouched (object not yet / no longer present).
    """
    n = len(positions)
    valid = [i for i, p in enumerate(positions) if p is not None]
    if not valid:
        return list(positions)
    first, last = valid[0], valid[-1]
    result = list(positions)

    prev_valid = [None] * n
    next_valid = [None] * n
    stack_prev = None
    for i in range(first, last + 1):
        if positions[i] is not None:
            stack_prev = i
        prev_valid[i] = stack_prev
    stack_next = None
    for i in range(last, first - 1, -1):
        if positions[i] is not None:
            stack_next = i
        next_valid[i] = stack_next

    for i in range(first, last + 1):
        if result[i] is not None:
            continue
        p, q = prev_valid[i], next_valid[i]
        if p is None or q is None or q == p:
            continue
        t = (i - p) / float(q - p)
        x = positions[p][0] + t * (positions[q][0] - positions[p][0])
        y = positions[p][1] + t * (positions[q][1] - positions[p][1])
        result[i] = (x, y)
    return result


# --- camera movement -------------------------------------------------------
def _first_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read first frame of {video_path}")
    return frame


def _camera_movement(video_path, frame_count):
    """Per-frame [dx, dy] camera offsets for the video named in annotations.

    Uses the per-video camera stub (same convention as Phase 1) with
    read_from_stub=True; only the first video frame is needed to construct the
    estimator when the stub exists.
    """
    stub = camera_stub_path(video_path)
    frame0 = _first_video_frame(video_path)
    est = CameraMovementEstimator(frame0)
    if os.path.exists(stub):
        cam = est.get_camera_movement([frame0], read_from_stub=True, stub_path=stub)
    else:
        # No stub: fall back to reading the whole video (slow but correct).
        from utils import read_video
        cam = est.get_camera_movement(read_video(video_path), read_from_stub=True, stub_path=stub)
    if len(cam) < frame_count:
        # Pad defensively so indexing never fails.
        cam = list(cam) + [[0, 0]] * (frame_count - len(cam))
    return cam


def _id_sort_key(tid):
    try:
        return (0, int(tid))
    except (ValueError, TypeError):
        return (1, tid)


# --- export ----------------------------------------------------------------
def export(annotations_path=ANNOTATIONS_PATH, output_path=DEFAULT_OUTPUT):
    if not os.path.exists(annotations_path):
        raise SystemExit(f"{annotations_path} not found. Run editor.generate first.")
    with open(annotations_path) as fh:
        ann = json.load(fh)

    video_path = ann["video"]
    frames = ann["frames"]
    n = len(frames)

    vt = ViewTransformer()
    offset = _center_offset(vt)
    cam = _camera_movement(video_path, n)

    # --- Players: per-track top-left-origin metric position, then interpolate.
    player_pos = {}     # tid -> [ (x,y)|None ] * n
    player_team = {}    # tid -> "Offence"|"Defence"
    for f, fr in enumerate(frames):
        for p in fr["players"]:
            tid = p["track_id"]
            player_team.setdefault(tid, p["team"])
            seq = player_pos.setdefault(tid, [None] * n)
            x1, y1, x2, y2 = p["bbox"]
            foot = np.array([(x1 + x2) / 2.0 - cam[f][0], y2 - cam[f][1]], dtype=np.float32)
            t = vt.transform_point(foot)   # polygon-filtered: off-pitch -> None
            if t is not None:
                x, y = np.array(t).squeeze().tolist()
                seq[f] = (float(x), float(y))
    for tid in player_pos:
        player_pos[tid] = _interpolate_gaps(player_pos[tid])

    # --- Ball: raw (unfiltered) transform so out-of-bounds still gets coords.
    ball_pos = [None] * n
    ball_state = [None] * n
    ball_kick = [None] * n
    for f, fr in enumerate(frames):
        b = fr.get("ball")
        if not b:
            continue
        x1, y1, x2, y2 = b["bbox"]
        center = [(x1 + x2) / 2.0 - cam[f][0], (y1 + y2) / 2.0 - cam[f][1]]
        raw = _raw_transform(vt, center)
        ball_pos[f] = (float(raw[0]), float(raw[1]))
        ball_state[f] = b.get("state")
        ball_kick[f] = b.get("kick")
    ball_pos = _interpolate_gaps(ball_pos)
    valid = [i for i, p in enumerate(ball_pos) if p is not None]
    if valid:
        first, last = valid[0], valid[-1]
        for i in range(first):
            ball_pos[i] = ball_pos[first]
        for i in range(last + 1, n):
            ball_pos[i] = ball_pos[last]
    else:
        ball_pos = [(0.0, 0.0)] * n

    # --- Top-level players: unique ids that appear on the pitch at least once.
    appearing = [tid for tid, seq in player_pos.items() if any(p is not None for p in seq)]
    players_list = [{"id": tid, "role": "Outfield"} for tid in sorted(appearing, key=_id_sort_key)]

    # --- Per-frame output.
    frames_out = []
    for f in range(n):
        bp = ball_pos[f]
        state = ball_state[f] or "OutOfBounds"
        if bp is not None:
            bx, by = _to_center_coords(bp, offset)
        else:
            bx, by = 0.0, 0.0
            state = "OutOfBounds"
        ball_obj = {"x": bx, "y": by, "state": state}
        kick = ball_kick[f]
        if isinstance(kick, dict) and (kick.get("byPlayerId") or kick.get("toPlayerId")):
            ball_obj["kick"] = kick

        frame_players = []
        for tid, seq in player_pos.items():
            p = seq[f]
            if p is None:
                continue
            x, y = _to_center_coords(p, offset)
            frame_players.append({"id": tid, "team": player_team[tid], "x": x, "y": y})

        frames_out.append({"ball": ball_obj, "players": frame_players})

    output = {"frames": frames_out, "players": players_list}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(output, fh, indent=2)
    os.replace(tmp, output_path)
    print(f"Wrote {output_path} ({n} frames, {len(players_list)} players).")
    return output


def main():
    parser = argparse.ArgumentParser(description="Export annotations.json to schema JSON.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--annotations", default=ANNOTATIONS_PATH, help="Input annotations.json.")
    args = parser.parse_args()
    export(args.annotations, args.output)


if __name__ == "__main__":
    main()

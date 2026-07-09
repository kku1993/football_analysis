"""Phase 4: export pixel-space annotations.json -> schema-conformant JSON.

Run as::

    venv/bin/python -m editor.export [-o output_videos/tracking_data.json]

Converts the editor's pixel-space working file (``editor_data/annotations.json``)
into the metric JSON described by ``tracking-schema.json``.

COORDINATE SYSTEM: meters from the CENTER of a standard 105 m x 68 m pitch.
``(0, 0)`` is the pitch center; ``-x`` is toward the left goal, ``+x`` toward the
right goal; ``-y`` toward the top touchline, ``+y`` toward the bottom touchline.

The ``ViewTransformer`` homography maps the image onto the pitch plane, in real
meters, but with its origin at the near goal-line / top-touchline *corner* of the
pitch (its ``target_vertices`` run 0..length along x and 0..width along y from
that corner). We therefore shift by half a pitch (``PITCH_LENGTH/2`` in x,
``PITCH_WIDTH/2`` in y) to move the origin to the pitch center. This assumes the
transform's ``(0,0)`` corner really is a pitch corner (goal line x touchline) --
true for the sample calibration (the visible trapezoid starts at the left goal
line and spans the full width), and it is what a per-camera recalibration should
preserve.

Adapted from the deleted ``tracking_output.py`` (``git show
5ad880f:tracking_output.py``), which already solved metric conversion and gap
interpolation; here we read boxes from ``annotations.json`` instead of the
pipeline's ``tracks``, and reference the origin to the true pitch center.

GENERALIZATION CAVEAT: the metric conversion is only as good as
``ViewTransformer.pixel_vertices``, a hardcoded calibration for the SAMPLE
video's camera view. The pixel-space editor (generate / server / frontend) is
fully video-agnostic, but this export step inherits the pipeline's per-video
calibration and must be recalibrated per camera setup, exactly as ``main.py``
already requires. We do not attempt auto-calibration.

Roles: the tracker merges goalkeepers into "players" with no preserved flag, so
``GoalKeeper`` cannot be auto-inferred; instead the human marks the goalkeeper on
a player box in the editor. A player exported as ``GoalKeeper`` if they are marked
GK on any frame; otherwise ``Outfield``.
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

# Standard soccer pitch dimensions (meters). x = length (goal to goal),
# y = width (touchline to touchline). Used to place the coordinate origin at the
# pitch center; see the module docstring's COORDINATE SYSTEM note.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


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


def _center_offset():
    """(cx, cy) offset converting corner-origin metric -> pitch-CENTER origin.

    The ViewTransformer's metric plane has its origin at the near goal-line /
    top-touchline corner of the pitch (real meters). The pitch center is half a
    pitch length/width from that corner, so subtracting (PITCH_LENGTH/2,
    PITCH_WIDTH/2) recenters to (0,0) at the pitch center. See the module
    docstring for the coordinate system and the calibration assumption.
    """
    return (PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0)


def _to_center_coords(point, offset):
    cx, cy = offset
    return (point[0] - cx, point[1] - cy)


def _clip_to_pitch(point):
    """Clamp a corner-origin metric (x,y) into the pitch rectangle.

    Corner-origin metric runs 0..PITCH_LENGTH in x and 0..PITCH_WIDTH in y, so a
    projected point beyond the touchlines / goal lines (typical for a box that
    was extrapolated from outside the calibration region) is pulled back onto the
    nearest boundary rather than reported meters off the field.
    """
    x = min(max(point[0], 0.0), PITCH_LENGTH)
    y = min(max(point[1], 0.0), PITCH_WIDTH)
    return (x, y)


def _player_box_position(view_transformer, bbox, cam_offset, offset):
    """Corner-origin metric position for a player box.

    Projects all four box corners onto the pitch plane (raw/unfiltered, so boxes
    outside the calibration region still extrapolate) and returns the one whose
    pitch position is closest to the pitch CENTER. For a box straddling the
    boundary this biases inward, giving a more conservative estimate than the
    bottom-center foot point. The winner is then clipped into the pitch.

    Note: the homography assumes points lie on the ground plane, which is only
    true for the two *bottom* corners (the feet); the top corners are the head
    and project as if they were on the ground. They are still considered here
    because the request is "the corner closest to pitch center", but in practice
    the nearest-to-center corner is almost always a bottom one.
    """
    x1, y1, x2, y2 = bbox
    dx, dy = cam_offset
    corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    best = None
    best_d = None
    for px, py in corners:
        raw = _raw_transform(view_transformer, [px - dx, py - dy])
        cx, cy = _to_center_coords(raw, offset)
        d = cx * cx + cy * cy   # squared distance to pitch center (0,0)
        if best_d is None or d < best_d:
            best_d = d
            best = raw
    return _clip_to_pitch(best)


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
    offset = _center_offset()
    cam = _camera_movement(video_path, n)

    # --- Players: per-track top-left-origin metric position, then interpolate.
    # Position comes from the box corner nearest the pitch center, projected with
    # the raw (unfiltered) transform so a player outside the calibration polygon
    # still gets metric coordinates: the homography encodes the camera angle
    # fitted from the calibration region, and extrapolating it projects off-region
    # boxes onto the same pitch plane. The result is clipped into the pitch.
    player_pos = {}     # tid -> [ (x,y)|None ] * n
    player_team = {}    # tid -> "Offence"|"Defence"
    player_role = {}    # tid -> "GoalKeeper" if marked GK in ANY frame, else "Outfield"
    for f, fr in enumerate(frames):
        for p in fr["players"]:
            tid = p["track_id"]
            player_team.setdefault(tid, p["team"])
            if p.get("role") == "GoalKeeper":
                player_role[tid] = "GoalKeeper"
            else:
                player_role.setdefault(tid, "Outfield")
            seq = player_pos.setdefault(tid, [None] * n)
            pos = _player_box_position(vt, p["bbox"], cam[f], offset)
            seq[f] = (float(pos[0]), float(pos[1]))
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
    players_list = [{"id": tid, "role": player_role.get(tid, "Outfield")}
                    for tid in sorted(appearing, key=_id_sort_key)]

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

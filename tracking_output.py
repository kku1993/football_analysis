import json
import numpy as np
import cv2

from utils import get_center_of_bbox


TEAM_NAME = {1: "Defence", 2: "Offence"}


def _raw_transform(view_transformer, point):
    """Apply the perspective transform without the inside-polygon filter.

    This yields real metric coordinates even when the object is slightly
    off the pitch (e.g. an out-of-bounds ball), rather than returning None.
    """
    mat = getattr(view_transformer, "persepctive_trasnformer", None)
    if mat is None:
        mat = getattr(view_transformer, "perspective_transformer")
    src = np.array([[float(point[0]), float(point[1])]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, mat)
    return dst.reshape(-1, 2)[0].tolist()


def _center_offset(view_transformer):
    """Return (cx, cy) offset to convert top-left origin -> pitch-center origin."""
    verts = np.array(view_transformer.target_vertices, dtype=np.float32)
    xs = verts[:, 0]
    ys = verts[:, 1]
    return (float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0))


def _to_center_coords(point, offset):
    """Convert a top-left-origin metric point to center-origin coordinates.

    (0,0) is the center of the pitch, -x is left, +x is right, -y is up, +y is down.
    """
    cx, cy = offset
    return (point[0] - cx, point[1] - cy)


def _interpolate_gaps(positions):
    """Linearly interpolate interior None gaps in a list of (x, y) | None.

    Leading/trailing Nones are left untouched (object not yet on / no longer on pitch).
    """
    n = len(positions)
    valid = [i for i, p in enumerate(positions) if p is not None]
    if not valid:
        return positions  # all None

    first, last = valid[0], valid[-1]
    result = list(positions)

    # Fill interior gaps with linear interpolation between the surrounding
    # known samples. idx arrays sorted ascending.
    next_valid = [None] * n
    prev_valid = [None] * n
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
        p = prev_valid[i]
        q = next_valid[i]
        if p is None or q is None or q == p:
            continue
        t = (i - p) / float(q - p)
        x = positions[p][0] + t * (positions[q][0] - positions[p][0])
        y = positions[p][1] + t * (positions[q][1] - positions[p][1])
        result[i] = (x, y)
    return result


def _player_team_map(tracks):
    """Map each player track_id to its static team (1/2)."""
    player_team = {}
    for frame in tracks["players"]:
        for tid, info in frame.items():
            if "team" in info and tid not in player_team:
                player_team[tid] = int(info["team"])
    return player_team


def _player_positions(tracks, view_transformer):
    """Per player track_id -> list(len num_frames) of top-left-origin (x,y) | None.

    Uses the polygon-filtered `position_transformed` already computed by
    ViewTransformer, so that off-pitch detections (spectators, staff warming up
    along the touchline, etc.) are excluded: those frames yield None and are only
    filled by interpolation when the player is actually on the pitch in nearby
    frames. Frames where the player was not detected at all also remain None and
    are interpolated in place to bridge temporary disappearances.
    """
    num_frames = len(tracks["players"])
    out = {tid: [None] * num_frames for tid in _collect_player_ids(tracks)}
    for f, frame in enumerate(tracks["players"]):
        for tid, info in frame.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            out[tid][f] = (float(pos[0]), float(pos[1]))
    for tid in out:
        out[tid] = _interpolate_gaps(out[tid])
    return out


def _collect_player_ids(tracks):
    ids = set()
    for frame in tracks["players"]:
        ids.update(frame.keys())
    return ids


def _ball_positions(tracks, view_transformer, camera_movement_per_frame):
    """Per-frame ball position (top-left-origin metric (x,y)) and state.

    The ball bbox was already interpolated by Tracker.interpolate_ball_positions,
    but its transformed position wasn't recomputed afterward, so we recompute it
    here from the bbox center, stabilised against camera motion using the same
    formula the rest of the pipeline uses for players, and then through the same
    polygon-filtered perspective transform (so the ball stays on the pitch and
    is consistent with player coordinates). Interior gaps are linearly
    interpolated and the ends are held (forward/back-filled) with the last known
    position so every frame yields numeric coordinates. State is reported as
    "Active"; out-of-bounds detection isn't reliable from interpolated data and
    isn't required by the task.
    """
    num_frames = len(tracks["ball"])
    positions = [None] * num_frames
    for f in range(num_frames):
        ball = tracks["ball"][f].get(1)
        if not ball or "bbox" not in ball or len(ball["bbox"]) < 4:
            continue
        cx, cy = get_center_of_bbox(ball["bbox"])
        cam = camera_movement_per_frame[f]
        stab = np.array([cx - cam[0], cy - cam[1]], dtype=np.float32)
        transformed = view_transformer.transform_point(stab)
        if transformed is not None:
            x, y = transformed.squeeze().tolist()
            positions[f] = (float(x), float(y))

    positions = _interpolate_gaps(positions)
    # Hold last known position at leading/trailing gaps so every frame has coords.
    valid = [i for i, p in enumerate(positions) if p is not None]
    if valid:
        first, last = valid[0], valid[-1]
        for i in range(0, first):
            positions[i] = positions[first]
        for i in range(last + 1, num_frames):
            positions[i] = positions[last]
    if not valid:
        positions = [(0.0, 0.0)] * num_frames

    states = ["Active"] * num_frames
    return positions, states


def _possession_per_frame(tracks):
    """Per-frame player track_id holding the ball, or None."""
    num_frames = len(tracks["players"])
    possession = [None] * num_frames
    for f in range(num_frames):
        for tid, info in tracks["players"][f].items():
            if info.get("has_ball"):
                possession[f] = tid
                break
    return possession


def _detect_goalkeepers(tracks, view_transformer, player_team):
    """Return set of track_ids identified as goalkeepers (conservatively).

    Goalkeepers were merged into the "players" tracks by the tracker with no
    preserved flag, so a position/movement heuristic is the best signal. A
    goalkeeper is, per team, a player who:
      * is observed on the pitch for at least MIN_FRAMES frames,
      * stays near a goal line (mean |x| within NEAR_GOAL metres of an end),
      * moves very little (total path among the lowest on the team).

    We only designate a GK when a clearly stationary candidate near a goal line
    exists; otherwise that team has no GK flagged (all its players stay Outfield),
    which keeps us from mislabeling a striker parked at the opponent's goal.
    """
    MIN_FRAMES = 15
    NEAR_GOAL = 3.0  # metres from a goal line

    verts = np.array(view_transformer.target_vertices, dtype=np.float32)
    half_length = float((verts[:, 0].max() - verts[:, 0].min()) / 2.0)
    near_goal_threshold = half_length - NEAR_GOAL

    # Collect, per player, the sequence of top-left-origin x samples and the
    # total path length travelled on the pitch, plus the team.
    samples_x = {}
    path = {}
    for f, frame in enumerate(tracks["players"]):
        for tid, info in frame.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            samples_x.setdefault(tid, []).append(pos[0])

    # Recompute path length properly per track across contiguous observed frames.
    seq = {}
    for f, frame in enumerate(tracks["players"]):
        for tid, info in frame.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            seq.setdefault(tid, []).append((pos[0], pos[1]))
    for tid, pts in seq.items():
        if len(pts) >= 2:
            import math
            path[tid] = sum(
                math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                for i in range(1, len(pts))
            )
        else:
            path[tid] = 0.0

    gks = set()
    by_team = {1: [], 2: []}
    for tid, xs in samples_x.items():
        team = player_team.get(tid)
        if team not in by_team or len(xs) < MIN_FRAMES:
            continue
        mean_x = sum(xs) / len(xs)
        by_team[team].append(
            {"tid": tid, "mean_x": mean_x, "path": path.get(tid, 0.0)}
        )

    for team, lst in by_team.items():
        # Near a goal line in center-origin terms: |mean x - center| large.
        candidates = [c for c in lst if abs(c["mean_x"] - half_length) > near_goal_threshold]
        if not candidates:
            candidates = lst
        if not candidates:
            continue
        gk = min(candidates, key=lambda c: c["path"])
        # Only flag as GK if clearly stationary relative to the team's typical
        # movement, to avoid mislabeling an attacker loitering near a goal.
        others = sorted(c["path"] for c in lst)
        if len(others) >= 4:
            import statistics
            med = statistics.median(others)
            if med > 0 and gk["path"] <= 0.35 * med:
                gks.add(gk["tid"])
        elif gk["path"] <= 8.0:
            gks.add(gk["tid"])
    return gks


def generate_tracking_json(tracks, view_transformer, camera_movement_per_frame, output_path):
    """Build a tracking-data JSON conforming to tracking-schema.json and write it.

    Coordinate system: (0,0) is the center of the pitch, -x is left, +x is right,
    -y is up, +y is down, all in meters. Only players (incl. goalkeepers) are
    tracked -- referees and spectators are excluded. Positions for players or the
    ball that temporarily disappear between frames are linearly interpolated.
    """
    offset = _center_offset(view_transformer)
    num_frames = len(tracks["players"])

    player_team = _player_team_map(tracks)
    player_pos = _player_positions(tracks, view_transformer)
    ball_pos, ball_states = _ball_positions(tracks, view_transformer, camera_movement_per_frame)
    possession = _possession_per_frame(tracks)
    gk_ids = _detect_goalkeepers(tracks, view_transformer, player_team)

    # Top-level players list: only players that actually appear on the pitch in
    # at least one frame (i.e. have a real or interpolated position). This
    # excludes off-pitch detections such as spectators/staff warming up.
    appearing = [
        tid
        for tid, pos_list in player_pos.items()
        if any(p is not None for p in pos_list)
    ]
    players_list = []
    for tid in sorted(appearing):
        role = "GoalKeeper" if tid in gk_ids else "Outfield"
        players_list.append({"id": str(tid), "role": role})

    # Per-frame output.
    frames_out = []
    prev_holder = None
    for f in range(num_frames):
        # --- Ball ---
        bp = ball_pos[f]
        state = ball_states[f]
        if bp is not None:
            bx, by = _to_center_coords(bp, offset)
        else:
            bx, by = 0.0, 0.0
            state = "OutOfBounds"

        ball_obj = {"x": bx, "y": by, "state": state}

        # Kick detection: a kick is recorded on the frame where possession
        # switches from a previously tracked holder to a different player.
        # We attribute the kick to the previous holder (the kicker).
        holder = possession[f]
        if (
            holder is not None
            and prev_holder is not None
            and holder != prev_holder
        ):
            ball_obj["kick"] = {"byPlayerId": str(prev_holder)}
        prev_holder = holder if holder is not None else prev_holder

        # --- Players present in this frame (real or interpolated) ---
        frame_players = []
        for tid, pos_list in player_pos.items():
            p = pos_list[f]
            if p is None:
                continue
            team = player_team.get(tid, 1)
            x, y = _to_center_coords(p, offset)
            frame_players.append(
                {
                    "id": str(tid),
                    "team": TEAM_NAME.get(team, "Defence"),
                    "x": x,
                    "y": y,
                }
            )

        frames_out.append({"ball": ball_obj, "players": frame_players})

    output = {"frames": frames_out, "players": players_list}

    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    return output
"""Build the tracking-schema.json export from the editor state.

The user labels a set of (possibly non-consecutive) "to process" frames with
bounding boxes and per-frame homographies. Each labeled frame becomes a
keyframe: the box anchor pixel (feet for players, center for the ball, unless
overridden) is projected to pitch meters through that frame's homography.

The frame number parsed from each filename places keyframes on a dense
timeline; the first (lexicographically first / lowest-numbered) keyframe is the
start and gaps between consecutive keyframes are filled by extrapolation:

- players move on a straight line between labeled positions, with a smooth
  random acceleration bump plus per-frame jitter; defenders are pulled slightly
  toward the ball when near, else toward the nearest attacker
- the ball rides 0.3-0.5 m ahead of the player in possession until a keyframe
  marks a kick (kick.byPlayerId), then flies linearly in (x, y) to its next
  labeled position, with a parabolic z arc when the z axis is involved

Adapted from football-pitch-calibration/label-system/export.py; the difference
is that homographies are computed on the fly from the editor's calibration dots
rather than read from exported files.
"""

import math
import random
import re

from .homography import CalibrationError, compute_homography


def frame_index(name):
    m = re.search(r"(\d+)(?=\.[^.]+$)", name)
    if not m:
        raise ValueError(f"cannot parse a frame number from '{name}'")
    return int(m.group(1))


def id_sort_key(pid):
    m = re.search(r"(\d+)$", pid)
    return (0, int(m.group(1))) if m else (1, pid)


def lerp(a, b, t):
    return a + (b - a) * t


# ---------- anchor + projection ----------

def anchor_px(box, kind):
    """Pixel point of ``box`` to project. ``kind`` is 'player' or 'ball'.

    Players default to feet (bottom-center); the ball defaults to its center.
    A per-box ``anchor`` of 'feet' or 'center' overrides the default.
    """
    cx = (box["x_min"] + box["x_max"]) / 2.0
    anchor = box.get("anchor")
    if anchor == "center" or (anchor is None and kind == "ball"):
        return cx, (box["y_min"] + box["y_max"]) / 2.0
    return cx, box["y_max"]  # feet


def homography_for(dots):
    """3x3 homography from calibration dots, or None if fewer than 4."""
    if len(dots) < 4:
        return None
    return compute_homography([(d["px"], d["py"]) for d in dots],
                              [(d["x"], d["y"]) for d in dots])


def px_to_xy(H, px, py):
    X = H[0][0] * px + H[0][1] * py + H[0][2]
    Y = H[1][0] * px + H[1][1] * py + H[1][2]
    W = H[2][0] * px + H[2][1] * py + H[2][2]
    return X / W, Y / W


# ---------- keyframe collection ----------

def collect_keyframes(state):
    """Project each labeled to-process frame into pitch coordinates.

    Returns a list of keyframes sorted by frame index, each
    ``{"idx", "name", "players": {id: (x, y, z)}, "ball": {...}|None}``.
    Raises ValueError if a labeled frame lacks a valid homography.
    """
    to_process = state.get("toProcess") or sorted(state.get("frames", {}).keys())
    calib = state.get("calibration", {})
    frames_state = state.get("frames", {})

    keyframes = []
    missing_h = []
    for name in sorted(to_process, key=frame_index):
        fs = frames_state.get(name)
        if not fs or (not fs.get("players") and not fs.get("ball")):
            continue
        try:
            H = homography_for(calib.get(name, []))
        except CalibrationError as e:
            raise ValueError(f"{name}: {e}")
        if H is None:
            missing_h.append(name)
            continue
        Hl = [[float(v) for v in row] for row in H]

        players = {}
        for box in fs.get("players", []):
            px, py = anchor_px(box, "player")
            x, y = px_to_xy(Hl, px, py)
            players[box["id"]] = (x, y, float(box.get("z") or 0))

        ball = None
        bd = fs.get("ball")
        if bd:
            px, py = anchor_px(bd, "ball")
            x, y = px_to_xy(Hl, px, py)
            ball = {"x": x, "y": y, "z": float(bd.get("z") or 0),
                    "state": bd.get("state", "Active"),
                    "kick": bd.get("kick") or None}
        keyframes.append({"idx": frame_index(name), "name": name,
                          "players": players, "ball": ball})

    if missing_h:
        raise ValueError("to-process frames need 4+ calibration points before "
                         "export: " + ", ".join(missing_h))
    keyframes.sort(key=lambda kf: kf["idx"])
    return keyframes


# ---------- extrapolation ----------

def build_player_tracks(keyframes, lo, n):
    """Dense per-player (x, y, z) tracks; None where the player is absent."""
    pids = {pid for kf in keyframes for pid in kf["players"]}
    tracks = {pid: [None] * n for pid in pids}
    for pid in pids:
        pkfs = [(kf["idx"], kf["players"][pid]) for kf in keyframes
                if pid in kf["players"]]
        for idx, pos in pkfs:
            tracks[pid][idx - lo] = pos
        for (i, pa), (j, pb) in zip(pkfs, pkfs[1:]):
            steps = j - i
            if steps <= 1:
                continue
            ang = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.05, 0.4)
            bx, by = math.cos(ang) * amp, math.sin(ang) * amp
            for k in range(1, steps):
                t = k / steps
                env = math.sin(math.pi * t)
                x = lerp(pa[0], pb[0], t) + bx * env + random.gauss(0, 0.03)
                y = lerp(pa[1], pb[1], t) + by * env + random.gauss(0, 0.03)
                z = lerp(pa[2], pb[2], t)
                tracks[pid][i - lo + k] = (x, y, z)
    return tracks


def approximate_ball(keyframes, lo, n):
    """Cheap linear ball path used only to steer defenders."""
    bkfs = [(kf["idx"], kf["ball"]) for kf in keyframes if kf["ball"]]
    approx = [None] * n
    for (i, ba), (j, bb) in zip(bkfs, bkfs[1:]):
        for k in range(j - i + 1):
            t = k / max(j - i, 1)
            approx[i - lo + k] = (lerp(ba["x"], bb["x"], t),
                                  lerp(ba["y"], bb["y"], t))
    if not bkfs:
        return [None] * n
    first_i, first_b = bkfs[0]
    last_i, last_b = bkfs[-1]
    for di in range(n):
        if approx[di] is None:
            b = first_b if di < first_i - lo else last_b
            approx[di] = (b["x"], b["y"])
    return approx


def adjust_defenders(tracks, teams, ball_approx, keyframes, lo):
    """Pull defenders toward the ball if near, else the nearest attacker.

    Applied only to synthesized frames, scaled to vanish at labeled endpoints.
    """
    kf_indices = sorted(kf["idx"] for kf in keyframes)
    gap_t = {}
    for i, j in zip(kf_indices, kf_indices[1:]):
        for k in range(1, j - i):
            gap_t[i - lo + k] = k / (j - i)
    offence = [pid for pid, team in teams.items() if team == "Offence"]

    for pid, team in teams.items():
        if team != "Defence":
            continue
        track = tracks[pid]
        for di, t in gap_t.items():
            v = track[di]
            if v is None:
                continue
            bx, by = ball_approx[di] if ball_approx[di] else (v[0], v[1])
            target = None
            if math.hypot(bx - v[0], by - v[1]) < 8:
                target = (bx, by)
            else:
                best_d = 6.0
                for opid in offence:
                    ov = tracks[opid][di]
                    if ov is None:
                        continue
                    d = math.hypot(ov[0] - v[0], ov[1] - v[1])
                    if d < best_d:
                        best_d, target = d, (ov[0], ov[1])
            if target is None:
                continue
            pull = 0.25 * math.sin(math.pi * t)
            dx = (target[0] - v[0]) * pull
            dy = (target[1] - v[1]) * pull
            cap = 0.6
            scale = min(1.0, cap / max(math.hypot(dx, dy), 1e-9))
            track[di] = (v[0] + dx * scale, v[1] + dy * scale, v[2])


def build_ball_track(keyframes, tracks, lo, n):
    """Dense (x, y, z, state) ball track following possession/kick rules."""
    bkfs = [(kf["idx"], kf["ball"]) for kf in keyframes if kf["ball"]]
    ball = [None] * n
    for i, b in bkfs:
        ball[i - lo] = (b["x"], b["y"], b["z"], b["state"])

    for (i, ba), (j, bb) in zip(bkfs, bkfs[1:]):
        steps = j - i
        if steps <= 1:
            continue
        kicked = bool(ba.get("kick") and ba["kick"].get("byPlayerId"))
        if kicked:
            fill_pass(ball, ba, bb, i - lo, steps)
        else:
            fill_possession(ball, ba, bb, tracks, i - lo, steps)

    first_i, first_b = bkfs[0]
    last_i, last_b = bkfs[-1]
    for di in range(n):
        if ball[di] is None:
            b = first_b if di < first_i - lo else last_b
            ball[di] = (b["x"], b["y"], b["z"], b["state"])
    return ball


def fill_pass(ball, ba, bb, di0, steps):
    """Linear (x, y) flight; parabolic z arc when the z axis is involved."""
    peak = max(ba["z"], bb["z"])
    for k in range(1, steps):
        t = k / steps
        x = lerp(ba["x"], bb["x"], t)
        y = lerp(ba["y"], bb["y"], t)
        z = lerp(ba["z"], bb["z"], t)
        if peak > 0.05:
            z += 4 * peak * t * (1 - t)
        ball[di0 + k] = (x, y, z, ba["state"])


def fill_possession(ball, ba, bb, tracks, di0, steps):
    """Ball rides 0.3-0.5 m ahead of the possessing player."""
    poss, poss_d = None, float("inf")
    for pid, track in tracks.items():
        v = track[di0]
        if v is None:
            continue
        d = math.hypot(v[0] - ba["x"], v[1] - ba["y"])
        if d < poss_d:
            poss, poss_d = pid, d
    if poss is None:
        fill_pass(ball, ba, bb, di0, steps)
        return

    track = tracks[poss]
    for k in range(1, steps):
        t = k / steps
        p = track[di0 + k]
        if p is None:
            x = lerp(ba["x"], bb["x"], t)
            y = lerp(ba["y"], bb["y"], t)
        else:
            prev = track[di0 + k - 1] or p
            nxt = track[min(di0 + k + 1, len(track) - 1)] or p
            dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                dx, dy = bb["x"] - p[0], bb["y"] - p[1]
                norm = math.hypot(dx, dy)
            if norm < 1e-6:
                dx, dy, norm = 1.0, 0.0, 1.0
            ahead = random.uniform(0.3, 0.5)
            x = p[0] + dx / norm * ahead
            y = p[1] + dy / norm * ahead
        ball[di0 + k] = (x, y, lerp(ba["z"], bb["z"], t), ba["state"])


# ---------- top-level ----------

def build(state, fps):
    """Assemble the tracking-schema.json document from the editor state."""
    keyframes = collect_keyframes(state)
    if not keyframes:
        raise ValueError("no labeled to-process frames to export")
    if not any(kf["ball"] for kf in keyframes):
        raise ValueError("label the ball in at least one frame before exporting")

    lo = keyframes[0]["idx"]
    n = keyframes[-1]["idx"] - lo + 1

    player_meta = state.get("players", {})
    mapping = state.get("teamMapping", {}) or {}
    teams = {}   # pid -> "Offence" / "Defence"
    for kf in keyframes:
        for pid in kf["players"]:
            ab = player_meta.get(pid, {}).get("team", "A")
            teams[pid] = mapping.get(ab, "Offence")

    tracks = build_player_tracks(keyframes, lo, n)
    ball_approx = approximate_ball(keyframes, lo, n)
    adjust_defenders(tracks, teams, ball_approx, keyframes, lo)
    ball = build_ball_track(keyframes, tracks, lo, n)

    kick_at = {kf["idx"]: kf["ball"]["kick"] for kf in keyframes
               if kf["ball"] and kf["ball"].get("kick")}

    frames_out = []
    for di in range(n):
        idx = lo + di
        players = []
        for pid in sorted(tracks, key=id_sort_key):
            v = tracks[pid][di]
            if v is None:
                continue
            players.append({"id": pid, "team": teams[pid],
                            "x": round(v[0], 2), "y": round(v[1], 2),
                            "z": round(v[2], 2)})
        b = ball[di]
        ball_out = {"x": round(b[0], 2), "y": round(b[1], 2),
                    "z": round(b[2], 2), "state": b[3]}
        if idx in kick_at:
            ball_out["kick"] = kick_at[idx]
        frames_out.append({"ball": ball_out, "players": players})

    players_out = [{"id": pid,
                    "role": player_meta.get(pid, {}).get("role", "Outfield")}
                   for pid in sorted(teams, key=id_sort_key)]
    return {"fps": fps, "frames": frames_out, "players": players_out}

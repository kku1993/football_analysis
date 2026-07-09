"""Phase 2: Flask backend for the tracking editor.

Run as::

    venv/bin/python -m editor.server

Serves on http://127.0.0.1:5000. Loads ``editor_data/annotations.json`` into
memory at startup (produced by ``editor.generate``). The frontend owns all
editing logic; this backend just serves frames + annotations and validates and
persists per-frame edits. Per-frame PUTs are the autosave mechanism -- there is
no separate "save" concept. Single user, single process: a module-level dict
guarded by a lock is sufficient.
"""

import json
import os
import threading

from flask import Flask, jsonify, request, send_from_directory, abort

from editor.generate import ANNOTATIONS_PATH, FRAMES_DIR, FRAME_PAD, default_calibration

# Flask's send_from_directory resolves relative dirs against the package root, so
# anchor to absolute paths derived from the current working directory instead.
FRAMES_ABS = os.path.abspath(FRAMES_DIR)

TEAM_ENUM = {"Offence", "Defence"}
ROLE_ENUM = {"Outfield", "GoalKeeper"}
BALL_STATE_ENUM = {"Active", "Goal", "OutOfBounds"}

app = Flask(__name__, static_folder="static", static_url_path="/static")

_lock = threading.Lock()
_annotations = None  # loaded at startup


def _load_annotations():
    global _annotations
    if not os.path.exists(ANNOTATIONS_PATH):
        raise SystemExit(
            f"{ANNOTATIONS_PATH} not found. Run `venv/bin/python -m editor.generate` first.")
    with open(ANNOTATIONS_PATH) as fh:
        _annotations = json.load(fh)
    # Exclusion areas are a global (per-video) edit, not per-frame; backfill the
    # field on older annotation files that predate the feature.
    _annotations.setdefault("exclusion_areas", [])
    # Calibration is a global per-video edit. Older files predate it: fall back
    # to the sample default so the UI still has something to show / refine.
    _annotations.setdefault("calibration", default_calibration())
    # Ball trajectories (per-video list of frame ranges the human marks as the
    # ball being in the air, with a max height; the exporter interpolates a
    # parabola for z within each range). Backfill on older files.
    _annotations.setdefault("ball_trajectories", [])


def _persist_locked():
    """Atomically write the whole annotations file. Caller must hold _lock."""
    tmp = ANNOTATIONS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_annotations, fh, indent=2)
    os.replace(tmp, ANNOTATIONS_PATH)


def _validate_bbox(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "bbox must be a list of 4 numbers"
    if not all(isinstance(v, (int, float)) for v in bbox):
        return "bbox values must be numbers"
    x1, y1, x2, y2 = bbox
    if not (x1 < x2 and y1 < y2):
        return "bbox must satisfy x1 < x2 and y1 < y2"
    return None


def _validate_kick(kick):
    if kick is None:
        return None
    if not isinstance(kick, dict):
        return "kick must be null or an object"
    keys = set(kick.keys())
    if keys == {"byPlayerId"}:
        if not isinstance(kick["byPlayerId"], str) or not kick["byPlayerId"]:
            return "kick.byPlayerId must be a non-empty string"
    elif keys == {"toPlayerId"}:
        if not isinstance(kick["toPlayerId"], str) or not kick["toPlayerId"]:
            return "kick.toPlayerId must be a non-empty string"
    else:
        return "kick must have exactly one of byPlayerId / toPlayerId"
    return None


def _validate_frame_annotation(obj):
    """Return an error string, or None if valid."""
    if not isinstance(obj, dict):
        return "annotation must be an object"
    if "players" not in obj or not isinstance(obj["players"], list):
        return "annotation.players must be a list"

    for p in obj["players"]:
        if not isinstance(p, dict):
            return "each player must be an object"
        tid = p.get("track_id")
        if not isinstance(tid, str) or not tid.strip():
            return "player.track_id must be a non-empty string"
        if p.get("team") not in TEAM_ENUM:
            return f"player.team must be one of {sorted(TEAM_ENUM)}"
        if p.get("role", "Outfield") not in ROLE_ENUM:
            return f"player.role must be one of {sorted(ROLE_ENUM)}"
        err = _validate_bbox(p.get("bbox"))
        if err:
            return f"player {tid}: {err}"

    ball = obj.get("ball")
    if ball is not None:
        if not isinstance(ball, dict):
            return "ball must be null or an object"
        err = _validate_bbox(ball.get("bbox"))
        if err:
            return f"ball: {err}"
        if ball.get("state") not in BALL_STATE_ENUM:
            return f"ball.state must be one of {sorted(BALL_STATE_ENUM)}"
        err = _validate_kick(ball.get("kick"))
        if err:
            return err

    return None


def _frame_count():
    return _annotations["frame_count"]


def _bbox_inside(area_bbox, box_bbox):
    """True when ``box_bbox`` is fully contained within ``area_bbox``."""
    return (area_bbox[0] <= box_bbox[0] and area_bbox[1] <= box_bbox[1]
            and area_bbox[2] >= box_bbox[2] and area_bbox[3] >= box_bbox[3])


def _purge_frame(frame, areas):
    """Remove every player and the ball from ``frame`` when their bbox is fully
    contained inside any of ``areas``. Mutates ``frame`` in place; returns count
    removed. Players and ball are both subject to exclusion.
    """
    removed = 0
    if not areas:
        return removed
    kept = []
    for p in frame.get("players", []):
        if any(_bbox_inside(a["bbox"], p["bbox"]) for a in areas):
            removed += 1
        else:
            kept.append(p)
    frame["players"] = kept
    ball = frame.get("ball")
    if ball is not None and any(_bbox_inside(a["bbox"], ball["bbox"]) for a in areas):
        frame["ball"] = None
        removed += 1
    return removed


def _purge_all(areas):
    """Run ``_purge_frame`` over every frame. Caller must hold _lock."""
    removed = 0
    for fr in _annotations["frames"]:
        removed += _purge_frame(fr, areas)
    return removed


def _next_area_id():
    mx = 0
    for a in _annotations.get("exclusion_areas", []):
        try:
            mx = max(mx, int(a["id"]))
        except (KeyError, ValueError, TypeError):
            continue
    return mx + 1


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "frame_count": _annotations["frame_count"],
        "width": _annotations["width"],
        "height": _annotations["height"],
        "fps": _annotations["fps"],
        "video": _annotations["video"],
        "exclusion_areas": _annotations.get("exclusion_areas", []),
        "calibration": _annotations.get("calibration"),
        "ball_trajectories": _annotations.get("ball_trajectories", []),
    })


def _validate_calibration(obj):
    """Return an error string, or None if valid. Used by GET/PUT below.

    Shape: ``{"pitch_length": float, "pitch_width": float, "points":
    [{"pixel": [x, y], "pitch": [mx, my]}, ...]}``. ``pixel`` coords are in
    frame-0 image-space pixels; ``pitch`` coords are in corner-origin meters,
    i.e. x in 0..pitch_length, y in 0..pitch_width (NOT centered).
    """
    if not isinstance(obj, dict):
        return "calibration must be an object"
    for k in ("pitch_length", "pitch_width"):
        v = obj.get(k)
        if not isinstance(v, (int, float)) or v <= 0:
            return f"{k} must be a positive number"
    pts = obj.get("points")
    if not isinstance(pts, list) or len(pts) < 4:
        return "points must be a list of at least 4 correspondences"
    seen = set()
    for i, p in enumerate(pts):
        if not isinstance(p, dict):
            return f"point {i} must be an object"
        pix, pit = p.get("pixel"), p.get("pitch")
        if (not isinstance(pix, list) or len(pix) != 2
                or not all(isinstance(v, (int, float)) for v in pix)):
            return f"point {i}.pixel must be [x, y] numbers"
        if (not isinstance(pit, list) or len(pit) != 2
                or not all(isinstance(v, (int, float)) for v in pit)):
            return f"point {i}.pitch must be [mx, my] numbers"
        key = (round(float(pix[0]), 3), round(float(pix[1]), 3),
               round(float(pit[0]), 3), round(float(pit[1]), 3))
        if key in seen:
            return f"point {i} duplicates an earlier correspondence"
        seen.add(key)
    return None


@app.route("/api/calibration", methods=["GET"])
def get_calibration():
    with _lock:
        return jsonify(_annotations.get("calibration", default_calibration()))


@app.route("/api/calibration", methods=["PUT"])
def put_calibration():
    obj = request.get_json(silent=True)
    if obj is None:
        return jsonify({"error": "request body must be JSON"}), 400
    err = _validate_calibration(obj)
    if err:
        return jsonify({"error": err}), 400
    normalized = {
        "pitch_length": float(obj["pitch_length"]),
        "pitch_width": float(obj["pitch_width"]),
        "points": [
            {"pixel": [float(p["pixel"][0]), float(p["pixel"][1])],
             "pitch": [float(p["pitch"][0]), float(p["pitch"][1])]}
            for p in obj["points"]
        ],
    }
    with _lock:
        _annotations["calibration"] = normalized
        _persist_locked()
    return jsonify({"ok": True, "calibration": normalized})


def _overlaps_trajectory(start, end, trajectories, exclude_id=None):
    """True when [start, end] overlaps any persisted trajectory range.

    Overlap is closed (a shared endpoint is considered conflict) because each
    frame must belong to at most one trajectory so the parabola z value is
    unambiguous in the exporter. Pass ``exclude_id`` to skip the trajectory
    being edited (PUT) so an in-place edit is allowed.
    """
    for t in trajectories:
        if exclude_id is not None and t["id"] == exclude_id:
            continue
        if not (end < t["start_frame"] or start > t["end_frame"]):
            return True
    return False


def _validate_ball_trajectory(obj, frame_count, trajectories, exclude_id=None):
    """Return an error string, or None if valid. Used by POST/PUT below.

    Shape: ``{"start_frame": int, "end_frame": int, "max_height": number}``.
    The exporter fits a parabola that is 0 at the endpoints and ``max_height``
    at the midpoint; the range must be inside the video and not overlap any
    other persisted trajectory so every frame has at most one z value.
    """
    if not isinstance(obj, dict):
        return "trajectory must be an object"
    s = obj.get("start_frame")
    e = obj.get("end_frame")
    h = obj.get("max_height")
    if (not isinstance(s, int) or isinstance(s, bool)
            or not isinstance(e, int) or isinstance(e, bool)):
        return "start_frame and end_frame must be integers"
    if not (0 <= s < frame_count) or not (0 <= e < frame_count):
        return "start_frame and end_frame must be within the frame range"
    if e <= s:
        return "end_frame must be greater than start_frame"
    if not isinstance(h, (int, float)) or isinstance(h, bool) or h <= 0:
        return "max_height must be a positive number"
    if _overlaps_trajectory(s, e, trajectories, exclude_id=exclude_id):
        return "frame range overlaps an existing ball trajectory"
    return None


def _next_trajectory_id():
    mx = 0
    for t in _annotations.get("ball_trajectories", []):
        try:
            mx = max(mx, int(t["id"]))
        except (KeyError, ValueError, TypeError):
            continue
    return mx + 1


@app.route("/api/ball_trajectories", methods=["GET"])
def get_ball_trajectories():
    with _lock:
        return jsonify(_annotations.get("ball_trajectories", []))


@app.route("/api/ball_trajectories", methods=["POST"])
def create_ball_trajectory():
    body = request.get_json(silent=True) or {}
    with _lock:
        trajs = _annotations.setdefault("ball_trajectories", [])
        err = _validate_ball_trajectory(body, _frame_count(), trajs)
        if err:
            return jsonify({"error": err}), 400
        tid = _next_trajectory_id()
        entry = {
            "id": tid,
            "start_frame": int(body["start_frame"]),
            "end_frame": int(body["end_frame"]),
            "max_height": float(body["max_height"]),
        }
        trajs.append(entry)
        _persist_locked()
    return jsonify(entry)


@app.route("/api/ball_trajectories/<int:tid>", methods=["PUT"])
def update_ball_trajectory(tid):
    body = request.get_json(silent=True) or {}
    with _lock:
        trajs = _annotations.get("ball_trajectories", [])
        target = None
        for t in trajs:
            if t["id"] == tid:
                target = t
                break
        if target is None:
            return jsonify({"error": "ball trajectory not found"}), 404
        err = _validate_ball_trajectory(body, _frame_count(), trajs, exclude_id=tid)
        if err:
            return jsonify({"error": err}), 400
        target["start_frame"] = int(body["start_frame"])
        target["end_frame"] = int(body["end_frame"])
        target["max_height"] = float(body["max_height"])
        _persist_locked()
    return jsonify({"ok": True, "trajectory": target})


@app.route("/api/ball_trajectories/<int:tid>", methods=["DELETE"])
def delete_ball_trajectory(tid):
    with _lock:
        trajs = _annotations.get("ball_trajectories", [])
        new_trajs = [t for t in trajs if t["id"] != tid]
        if len(new_trajs) == len(trajs):
            return jsonify({"error": "ball trajectory not found"}), 404
        _annotations["ball_trajectories"] = new_trajs
        _persist_locked()
    return jsonify({"ok": True})


@app.route("/api/next_track_id")
def next_track_id():
    """Max numeric track_id across ALL frames + 1, as a string.

    Computed server-side so newly created boxes get an id unique across every
    frame (not just the ones currently loaded in the browser).
    """
    with _lock:
        mx = 0
        for fr in _annotations["frames"]:
            for p in fr["players"]:
                try:
                    mx = max(mx, int(p["track_id"]))
                except (ValueError, TypeError):
                    continue
    return jsonify({"next_id": str(mx + 1)})


@app.route("/frames/<int:i>")
def frame_image(i):
    if not (0 <= i < _frame_count()):
        abort(404)
    return send_from_directory(FRAMES_ABS, f"frame_{i:0{FRAME_PAD}d}.jpg")


@app.route("/api/frame/<int:i>", methods=["GET"])
def get_frame(i):
    if not (0 <= i < _frame_count()):
        abort(404)
    with _lock:
        return jsonify(_annotations["frames"][i])


@app.route("/api/frame/<int:i>", methods=["PUT"])
def put_frame(i):
    if not (0 <= i < _frame_count()):
        abort(404)
    obj = request.get_json(silent=True)
    if obj is None:
        return jsonify({"error": "request body must be JSON"}), 400

    err = _validate_frame_annotation(obj)
    if err:
        return jsonify({"error": err}), 400

    # Normalize to the canonical shape we persist.
    normalized = {
        "players": [
            {
                "track_id": p["track_id"],
                "team": p["team"],
                "role": p.get("role", "Outfield"),
                "bbox": [float(v) for v in p["bbox"]],
            }
            for p in obj["players"]
        ],
        "ball": None,
    }
    ball = obj.get("ball")
    if ball is not None:
        normalized["ball"] = {
            "bbox": [float(v) for v in ball["bbox"]],
            "state": ball["state"],
            "kick": ball.get("kick"),
        }

    # Any box fully inside a persisted exclusion area is dropped before saving,
    # so the containment invariant is maintained on every edit -- not just when
    # an area is created/resized.
    areas = _annotations.get("exclusion_areas", [])
    removed = _purge_frame(normalized, areas)

    with _lock:
        _annotations["frames"][i] = normalized
        _persist_locked()
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/delete_player", methods=["POST"])
def delete_player():
    """Remove a player (by track_id) from ``from_frame`` through the last frame.

    Used when the human deletes a player box: the player should disappear from
    that frame onward, not just the current one. Frames before ``from_frame`` are
    left untouched. Idempotent; safe if the id is not present.
    """
    body = request.get_json(silent=True) or {}
    tid = body.get("track_id")
    frm = body.get("from_frame")
    if not isinstance(tid, str) or not tid:
        return jsonify({"error": "track_id must be a non-empty string"}), 400
    if not isinstance(frm, int) or not (0 <= frm < _frame_count()):
        return jsonify({"error": "from_frame out of range"}), 400

    with _lock:
        removed = 0
        for i in range(frm, _frame_count()):
            players = _annotations["frames"][i]["players"]
            kept = [p for p in players if p.get("track_id") != tid]
            removed += len(players) - len(kept)
            _annotations["frames"][i]["players"] = kept
        _persist_locked()
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/invert_teams", methods=["POST"])
def invert_teams():
    """Swap Offence <-> Defence for every player in every frame.

    Global toggle used when the human decides the two teams were assigned the
    wrong way round. Self-inverse: calling it again restores the original.
    """
    swap = {"Offence": "Defence", "Defence": "Offence"}
    with _lock:
        changed = 0
        for fr in _annotations["frames"]:
            for p in fr["players"]:
                new = swap.get(p.get("team"))
                if new is not None:
                    p["team"] = new
                    changed += 1
        _persist_locked()
    return jsonify({"ok": True, "changed": changed})


@app.route("/api/propagate", methods=["POST"])
def propagate_attr():
    """Apply attribute edits to a box from ``from_frame`` through the last frame.

    Persistent box attributes carry forward once edited, rather than being
    re-entered on every frame:
      * player: ``team``, ``role``, ``track_id`` (rename) -- matched by the
        ``track_id`` given in the request ("that player, from here on").
      * ball: ``state`` -- applied to every subsequent frame that has a ball.

    Bounding-box geometry and instantaneous events (ball ``kick``) are per-frame
    and are intentionally NOT propagated. Frames before ``from_frame`` are left
    untouched.
    """
    body = request.get_json(silent=True) or {}
    target = body.get("target")
    frm = body.get("from_frame")
    attrs = body.get("attrs")
    if target not in ("player", "ball"):
        return jsonify({"error": "target must be 'player' or 'ball'"}), 400
    if not isinstance(frm, int) or not (0 <= frm < _frame_count()):
        return jsonify({"error": "from_frame out of range"}), 400
    if not isinstance(attrs, dict) or not attrs:
        return jsonify({"error": "attrs must be a non-empty object"}), 400
    if "team" in attrs and attrs["team"] not in TEAM_ENUM:
        return jsonify({"error": f"team must be one of {sorted(TEAM_ENUM)}"}), 400
    if "role" in attrs and attrs["role"] not in ROLE_ENUM:
        return jsonify({"error": f"role must be one of {sorted(ROLE_ENUM)}"}), 400
    if "state" in attrs and attrs["state"] not in BALL_STATE_ENUM:
        return jsonify({"error": f"state must be one of {sorted(BALL_STATE_ENUM)}"}), 400
    if "track_id" in attrs and (not isinstance(attrs["track_id"], str) or not attrs["track_id"].strip()):
        return jsonify({"error": "track_id must be a non-empty string"}), 400

    match = body.get("track_id")
    if target == "player" and (not isinstance(match, str) or not match):
        return jsonify({"error": "track_id (match) must be a non-empty string"}), 400

    with _lock:
        changed = 0
        if target == "player":
            player_attrs = ("team", "role", "track_id")
            for i in range(frm, _frame_count()):
                for p in _annotations["frames"][i]["players"]:
                    if p.get("track_id") == match:
                        for k in player_attrs:
                            if k in attrs:
                                p[k] = attrs[k]
                        changed += 1
        else:  # ball
            for i in range(frm, _frame_count()):
                b = _annotations["frames"][i].get("ball")
                if b is not None and "state" in attrs:
                    b["state"] = attrs["state"]
                    changed += 1
        _persist_locked()
    return jsonify({"ok": True, "changed": changed})


@app.route("/api/exclusion_areas", methods=["POST"])
def create_exclusion_area():
    """Add a new exclusion area and immediately purge contained boxes.

    The area definition persists in ``annotations.json`` and applies to every
    frame. Existing boxes (players and ball) whose bbox is fully contained
    inside the new area are removed from all frames as part of this call.
    """
    body = request.get_json(silent=True) or {}
    bbox = body.get("bbox")
    err = _validate_bbox(bbox)
    if err:
        return jsonify({"error": err}), 400
    bbox = [float(v) for v in bbox]
    with _lock:
        areas = _annotations.setdefault("exclusion_areas", [])
        aid = _next_area_id()
        areas.append({"id": aid, "bbox": bbox})
        removed = _purge_all(areas)
        _persist_locked()
    return jsonify({"id": aid, "removed": removed})


@app.route("/api/exclusion_areas/<int:aid>", methods=["PUT"])
def update_exclusion_area(aid):
    """Move/resize an existing exclusion area; re-run the purge afterward.

    Previously-purged boxes are not restored (they no longer exist in any
    frame); only boxes that are now -- thanks to the geometry change -- fully
    contained inside the area get removed.
    """
    body = request.get_json(silent=True) or {}
    bbox = body.get("bbox")
    err = _validate_bbox(bbox)
    if err:
        return jsonify({"error": err}), 400
    bbox = [float(v) for v in bbox]
    with _lock:
        target = None
        for a in _annotations.get("exclusion_areas", []):
            if a["id"] == aid:
                target = a
                break
        if target is None:
            return jsonify({"error": "exclusion area not found"}), 404
        target["bbox"] = bbox
        removed = _purge_all(_annotations.get("exclusion_areas", []))
        _persist_locked()
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/exclusion_areas/<int:aid>", methods=["DELETE"])
def delete_exclusion_area(aid):
    """Remove an exclusion area definition. Boxes previously removed by it are
    NOT restored -- they have already been deleted from the frames.
    """
    with _lock:
        areas = _annotations.get("exclusion_areas", [])
        new_areas = [a for a in areas if a["id"] != aid]
        if len(new_areas) == len(areas):
            return jsonify({"error": "exclusion area not found"}), 404
        _annotations["exclusion_areas"] = new_areas
        _persist_locked()
    return jsonify({"ok": True})


def main():
    _load_annotations()
    print(f"Loaded {_frame_count()} frames from {ANNOTATIONS_PATH}.")
    print("Serving editor on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


# Load annotations on import too, so `flask run` / test harnesses work.
_load_annotations()


if __name__ == "__main__":
    main()

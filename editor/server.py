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

from editor.generate import ANNOTATIONS_PATH, FRAMES_DIR, FRAME_PAD

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
    })


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

    with _lock:
        _annotations["frames"][i] = normalized
        _persist_locked()
    return jsonify({"ok": True})


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


def main():
    _load_annotations()
    print(f"Loaded {_frame_count()} frames from {ANNOTATIONS_PATH}.")
    print("Serving editor on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


# Load annotations on import too, so `flask run` / test harnesses work.
_load_annotations()


if __name__ == "__main__":
    main()

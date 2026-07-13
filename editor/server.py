"""Editor backend (Step 2): label player/ball positions on frames.

Point it at a data directory produced by ``editor/generate.py`` (containing
``fps.txt``, ``frames/`` and ``bbox/``):

    python -m editor.server --dir eng-nor-49-offside-no-goal/ [--port 8000]

Then open http://localhost:8000. State autosaves to
``<dir>/label-system-state.json``.

The backend is deliberately thin: it serves frames + the initial bounding
boxes, persists the editor state, solves homographies from calibration dots
(so the frontend can project boxes to pitch coordinates in real time and so the
export can too), projects the standard pitch model for the calibration overlay,
and builds the final tracking-schema.json on export.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from . import export as export_mod
from .homography import CalibrationError, compute_homography

app = Flask(__name__, static_folder="static", static_url_path="/static")

cfg = {}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Standard pitch (meters), origin at the center spot; matches tracking-schema.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def list_frames():
    return sorted(f.name for f in cfg["frames_dir"].iterdir()
                  if f.suffix.lower() in IMAGE_EXTS)


def frame_bbox(name):
    """Load the CV bounding boxes for a frame, or an empty record."""
    stem = os.path.splitext(name)[0]
    path = cfg["bbox_dir"] / f"{stem}_bbox.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"players": [], "ball": None}


def read_fps():
    path = cfg["data_dir"] / "fps.txt"
    if path.exists():
        try:
            return float(path.read_text().strip())
        except ValueError:
            pass
    return 12.0


def default_state():
    return {
        "toProcess": [],
        "teamMapping": {"A": "Offence", "B": "Defence"},
        "players": {},        # id -> {team: "A"/"B", role: "Outfield"/"GoalKeeper"}
        "calibration": {},    # frameName -> [{px,py,x,y}]
        "frames": {},         # frameName -> {players:[box], ball:box|null}
        "correctSets": [],    # [{offence:[id], defence:[id], start, end}]
        "nextId": 1,
    }


def load_state():
    if cfg["state_path"].exists():
        try:
            return json.loads(cfg["state_path"].read_text())
        except json.JSONDecodeError:
            pass
    return default_state()


# ---------- routes ----------

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/frames/<path:name>")
def frame_image(name):
    return send_from_directory(cfg["frames_dir"], name)


@app.get("/api/bootstrap")
def bootstrap():
    frames = list_frames()
    return jsonify({
        "fps": read_fps(),
        "frames": frames,
        "bbox": {name: frame_bbox(name) for name in frames},
        "state": load_state(),
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
    })


@app.put("/api/state")
def save_state():
    state = request.get_json(force=True)
    fd, tmp = tempfile.mkstemp(dir=cfg["data_dir"], prefix=".label-state-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, cfg["state_path"])
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return jsonify({"ok": True})


@app.post("/api/homography")
def homography():
    """Solve a 3x3 homography (pixels -> pitch meters) from calibration dots."""
    dots = request.get_json(force=True).get("dots", [])
    if len(dots) < 4:
        return jsonify({"error": "need >= 4 dots", "homography": None})
    try:
        H = compute_homography([(d["px"], d["py"]) for d in dots],
                               [(d["x"], d["y"]) for d in dots])
    except CalibrationError as e:
        return jsonify({"error": str(e), "homography": None})
    return jsonify({"homography": [[float(v) for v in row] for row in H]})


def _pitch_model():
    """Standard 105x68 m pitch lines as polylines in pitch coordinates."""
    def seg(x1, y1, x2, y2, n=80):
        return [(x1 + (x2 - x1) * t / n, y1 + (y2 - y1) * t / n) for t in range(n + 1)]

    lines = [
        seg(-52.5, -34, -52.5, 34), seg(52.5, -34, 52.5, 34),
        seg(-52.5, 34, 52.5, 34), seg(-52.5, -34, 52.5, -34),
        seg(0, -34, 0, 34),
    ]
    for s in (-1, 1):
        gx = 52.5 * s
        bx = (52.5 - 16.5) * s
        ax = (52.5 - 5.5) * s
        lines += [seg(bx, -20.16, bx, 20.16), seg(gx, 20.16, bx, 20.16),
                  seg(gx, -20.16, bx, -20.16),
                  seg(ax, -9.16, ax, 9.16), seg(gx, 9.16, ax, 9.16),
                  seg(gx, -9.16, ax, -9.16)]
        spot = (52.5 - 11) * s
        a0 = math.degrees(math.asin(7.31 / 9.15))
        arc = [(spot + 9.15 * math.cos(math.radians(t)) * -s, 9.15 * math.sin(math.radians(t)))
               for t in [a0 * (2 * k / 59 - 1) for k in range(60)]]
        lines.append(arc)
    lines.append([(9.15 * math.cos(t), 9.15 * math.sin(t))
                  for t in [2 * math.pi * k / 160 for k in range(161)]])
    return lines


PITCH_MODEL = _pitch_model()


@app.post("/api/pitchlines")
def pitchlines():
    """Project the standard pitch model into image space for the given dots."""
    pts = request.get_json(force=True).get("dots", [])
    if len(pts) < 4:
        return jsonify({"error": "need >= 4 dots"}), 400
    try:
        H = compute_homography([(d["px"], d["py"]) for d in pts],
                               [(d["x"], d["y"]) for d in pts])
    except CalibrationError as e:
        return jsonify({"error": str(e)}), 400
    Hinv = np.linalg.inv(H)

    out = []
    for line in PITCH_MODEL:
        homog = np.column_stack([np.asarray(line), np.ones(len(line))])
        mapped = (Hinv @ homog.T).T
        run = []
        for u, v, w in mapped:
            if w > 1e-9 and -4000 < u / w < 8000 and -4000 < v / w < 6000:
                run.append([round(u / w, 1), round(v / w, 1)])
            else:
                if len(run) > 1:
                    out.append(run)
                run = []
        if len(run) > 1:
            out.append(run)
    return jsonify({"lines": out})


@app.post("/api/export")
def export():
    """Build tracking-schema.json from the posted state and return it."""
    state = request.get_json(force=True)
    try:
        data = export_mod.build(state, read_fps())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    resp = jsonify(data)
    resp.headers["Content-Disposition"] = "attachment; filename=tracking.json"
    return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="data directory from editor/generate.py "
                         "(has fps.txt, frames/, bbox/)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    base = Path(args.dir).resolve()
    cfg["data_dir"] = base
    cfg["frames_dir"] = base / "frames"
    cfg["bbox_dir"] = base / "bbox"
    cfg["state_path"] = base / "label-system-state.json"
    if not cfg["frames_dir"].is_dir():
        ap.error(f"missing frames directory: {cfg['frames_dir']}")

    print(f"serving {len(list_frames())} frames from {base} "
          f"on http://localhost:{args.port}")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()

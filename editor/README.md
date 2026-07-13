# Football label editor

Collaborate with the CV pipeline to label player and ball positions in
broadcast football video. Two steps:

1. **`generate.py`** turns a video into a data directory of frames + CV
   bounding boxes.
2. **`server.py`** serves a web editor to calibrate homographies, correct the
   boxes, and export pitch-coordinate tracking data.

All commands run from the repo root with the project venv.

## Step 1 — generate frames + bounding boxes

```bash
venv/bin/python -m editor.generate input_videos/eng-nor-49-offside-no-goal.mp4 --fps 12
```

- The video is resampled to `--fps` (default 12), then YOLO detection +
  ByteTrack + team-color clustering run on the resampled frames. Referees are
  dropped; the ball is interpolated across frames.
- Output goes to a directory named after the video stem:

  ```
  eng-nor-49-offside-no-goal/
    fps.txt                        # fps used
    frames/frame_001.png ...       # resampled frames
    bbox/frame_001_bbox.json ...   # players + ball per bbox-schema.json
  ```

- Raw tracks are cached to `stubs/<stem>_<fps>fps_tracks.pkl`; re-runs are
  instant. Pass `--no-stub` to force fresh detection.

## Step 2 — the editor

```bash
venv/bin/python -m editor.server --dir eng-nor-49-offside-no-goal/ --port 8000
```

Open http://localhost:8000. State autosaves after every change to
`<dir>/label-system-state.json`.

### Workflow

**1 · Select frames to process.** Browse every frame with ←/→ and press
**Space** (or the checkbox) to mark frames to process. They may be
non-consecutive. Click **Start labeling →**.

**2 · Calibrate.** For each to-process frame add ≥4 homography calibration
dots: click a landmark on the pitch map, then click its spot on the frame (its
meters get filled in), or click the frame and type the meters. Drag to adjust;
scroll to zoom, drag empty space to pan. Toggle the pitch overlay to check the
fit. "Copy dots from previous to-process frame" speeds up static cameras. The
dots + solved homography are stored per frame — the same structure a future
model can populate automatically.

**3 · Label boxes.** Switch to **Boxes**. Player/ball boxes are seeded from the
CV output. Drag a box body to move it, drag a corner handle to resize, or
**+ Player / + Ball** then click to add one. Select a box to edit:

- Player: id, team (A/B, shown as its Offence/Defence mapping), role
  (Outfield/GoalKeeper), per-frame z, and pitch (x, y) computed live from the
  homography. Team and role **propagate to that player in every frame**; z is
  per-frame.
- Ball: state (Active/Goal/OutOfBounds), kick (kicked-by / received-by a player
  id), z, live (x, y).
- **Anchor**: which pixel projects to the pitch — players default to feet, the
  ball to center; override per box.
- **Team A = Offence/Defence** toggles the whole mapping in one click.
- **Delete** a player removes their box from the current frame and all later
  frames.
- **Isolate selected** hides every other box across to-process frames.

**Correct rosters.** Define the offence/defence ids that should be on the pitch
from the current frame onward (or up to an end frame). Frames in range that are
missing or have extra ids are flagged with jump buttons.

### Export

**Export tracking.json** downloads data conforming to `tracking-schema.json`.
To-process frames are keyframes on a dense timeline (frame number from the
filename); gaps are filled by extrapolation:

- players move in near-straight lines with slight acceleration noise; defenders
  drift toward a nearby ball or the nearest attacker
- the ball rides 0.3–0.5 m ahead of the player in possession until a keyframe
  marks a kick, then flies linearly in (x, y) — with a parabolic z arc when z
  is involved — to its next labeled position

## Files

- `generate.py` — Step 1 pipeline (resample + CV → frames/ + bbox/)
- `server.py` — Flask backend (bootstrap, autosave, homography, pitchlines, export)
- `homography.py` — normalized-DLT pixel→pitch solver
- `export.py` — keyframe projection + gap extrapolation → tracking-schema.json
- `static/` — vanilla-JS frontend

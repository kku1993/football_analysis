# EDITOR_PLAN.md — Implementation Plan for the Tracking Editor

This is the implementation plan for the interactive tracking editor described in
`EDITOR.md`. It is written to be executed top-to-bottom by a coding agent with no
additional context. Read the whole plan before writing code.

## 1. Goal

Build a local, single-user web app that lets a human review and correct the
bounding boxes produced by the existing tracking pipeline (`main.py`), one video
frame at a time, and then export the corrected data as a JSON file that validates
against `tracking-schema.json`.

Core requirements from `EDITOR.md`:

1. The pipeline generates tracking data plus one image per frame, with bounding
   boxes visible for the ball and all players.
2. The human can step through frames and **add / remove / edit** bounding boxes:
   create a box by dragging on the image, and enter/edit the metadata attached to
   each box.
3. While editing frame N, the frames N-1 and N+1 are shown alongside it, with
   their bounding boxes, as read-only context.

Per-box metadata (from `EDITOR.md` + `tracking-schema.json`):

- **Player box**: pixel coordinates, team (`Offence` | `Defence`), unique tracking id.
- **Ball box**: pixel coordinates, state (`Active` | `Goal` | `OutOfBounds`), and an
  optional kick event: either *kicked by* a player id or *received by* a player id.

The editor must work for **any video** processed by `main.py`, not just the sample
clip currently in the repo. Never hardcode the video path, resolution, or frame
count: the generate step takes the video as a parameter and records its properties,
and everything downstream (server, frontend, export) reads them from that metadata.

## 2. Facts about the existing codebase (verified — do not rediscover)

- **Sample input video** (an example, not a constant): `input_videos/arg-egy-14_57-goal.mp4`
  — 1920×1080, 191 frames, ~29.97 fps. Use it for end-to-end verification, but no
  code may assume these numbers.
- **Python env**: use `venv/bin/python`. Installed: opencv-python, numpy, pandas,
  ultralytics, supervision, torch. **Flask is NOT installed** — the plan adds it.
- **Tracks data structure** (produced by `trackers/tracker.py::Tracker.get_object_tracks`,
  cached in `stubs/track_stubs.pkl`):
  ```python
  tracks = {
      "players":  [ {track_id(int): {"bbox": [x1,y1,x2,y2 floats, pixel coords], ...}} ] * n_frames,
      "referees": [ same shape ],
      "ball":     [ {1: {"bbox": [...]}} ],   # ball track_id is ALWAYS the literal key 1
  }
  ```
  After the `main.py` pipeline runs, player entries also gain `team` (int **1 or 2**),
  `team_color`, `has_ball`, `position`, `position_adjusted`, `position_transformed`.
- **Team int → schema enum mapping**: `{1: "Defence", 2: "Offence"}` (this mapping was
  established in the deleted `tracking_output.py`; recover it with
  `git show 5ad880f:tracking_output.py` — it also contains working, review-worthy code
  for metric conversion and gap interpolation that Phase 4 should reuse).
- **Metric conversion** (needed only at export): `view_transformer/view_transformer.py::ViewTransformer`
  maps pixel points → meters. Beware: the transform-matrix attribute is misspelled
  `persepctive_trasnformer`. `transform_point` returns `None` for points outside the
  pitch polygon. **Its `pixel_vertices` are a hardcoded calibration for the sample
  video's camera view** — for a different video they will be wrong. This is a
  pre-existing limitation of the pipeline, not something this project must solve;
  see the generalization notes in Phase 4. Schema wants coordinates in **meters from pitch center**;
  `target_vertices` are top-left-origin, so subtract the center offset
  (see `_center_offset`/`_to_center_coords` in the recovered `tracking_output.py`).
- **Camera movement**: `camera_movement_estimator` produces per-frame (dx, dy) offsets
  (cached in `stubs/camera_movement_stub.pkl`); `position_adjusted = position - offset`.
  The exporter must apply the same adjustment to human-edited boxes.
- Player "position" for metric purposes is the **foot position**:
  `utils/bbox_utils.py::get_foot_position` = (bbox center x, y2). Ball uses
  `get_center_of_bbox`.
- Stubs exist, so the pipeline runs without GPU inference
  (`read_from_stub=True` is already set in `main.py`).

## 3. Architecture and key design decisions

### 3.1 Stack

- **Backend**: Flask (add `flask` to `requirements.txt`, pip-install into `venv`).
  Single file `editor/server.py`, serves on `http://127.0.0.1:5000`.
- **Frontend**: one HTML page + vanilla JS + `<canvas>`. No build step, no npm, no
  external CDNs. Files: `editor/static/index.html`, `editor/static/editor.js`,
  `editor/static/editor.css`.
- **Why a web app**: dragging boxes, resize handles, and metadata forms are far more
  robust in HTML/canvas than in OpenCV or Tkinter windows, and it keeps the Python
  side trivially simple (serve images, load/save JSON).

### 3.2 Boxes are overlays, not baked pixels

`EDITOR.md` says main.py "creates one image per frame, with bounding boxes drawn".
Baking boxes into the image pixels would make them uneditable. **Deliberate
interpretation**: export *clean* frames as JPEGs, and have the editor draw the boxes
as canvas overlays from JSON. The user still *sees* every frame with its boxes (the
requirement's intent), and the boxes stay editable. Draw rectangles (not the
ellipse/triangle style used in the output video) since rectangles are what is edited.

### 3.3 Editor data format: pixel space, not metric

The human edits boxes in image pixel coordinates; the schema stores meters from
pitch center. Converting on every edit would be lossy and confusing. So there are
**two artifacts**:

1. `editor_data/annotations.json` — the editor's working file, pixel-space,
   the single source of truth while editing:
   ```json
   {
     "video": "input_videos/<whatever-was-processed>.mp4",
     "width": 1920,
     "height": 1080,
     "fps": 29.97,
     "frame_count": 191,
     "frames": [
       {
         "players": [
           {"track_id": "12", "team": "Offence", "bbox": [x1, y1, x2, y2]}
         ],
         "ball": {
           "bbox": [x1, y1, x2, y2],
           "state": "Active",
           "kick": null
         }
       }
     ]
   }
   ```
   - `width`/`height`/`fps`/`frame_count` are read from the video at generate time
     (`cv2.VideoCapture` properties / `len(video_frames)`) — never hardcoded.
   - `bbox` values are floats in the source video's pixel coordinates, `x1 < x2`, `y1 < y2`.
   - `ball` is `null` on frames where the ball has no box.
   - `kick` is `null`, `{"byPlayerId": "12"}`, or `{"toPlayerId": "34"}`.
   - `track_id` is a string (schema uses string ids). Seed from the int track ids.
2. `output_videos/tracking_data.json` — the schema-conformant export, produced by a
   separate export step (Phase 4) from `annotations.json`.

### 3.4 Components

```
editor/
  __init__.py
  generate.py       # Phase 1: pipeline -> frame JPEGs + seed annotations.json
  server.py         # Phase 2: Flask app (serve frames, GET/PUT annotations)
  export.py         # Phase 4: annotations.json -> schema-conformant JSON
  static/
    index.html
    editor.js
    editor.css
editor_data/        # generated, gitignored
  frames/frame_00000.jpg ... frame_00190.jpg
  annotations.json
```

Add `editor_data/` to `.gitignore`.

## 4. Implementation phases

Execute phases in order; each has a "verify" step — run it before moving on.

### Phase 0 — Setup

1. `venv/bin/pip install flask` and add `flask` to `requirements.txt`
   (unpinned or pinned to the installed version — match the file's existing pinned style).
2. Add `editor_data/` to `.gitignore`.

### Phase 1 — `editor/generate.py`: produce frames + seed annotations

A script run as `venv/bin/python -m editor.generate --video <path>` (default: the
video currently referenced in `main.py`, so the sample workflow needs no arguments).

1. Refactor, don't duplicate: extract the pipeline portion of `main.py` (read video →
   tracks → positions → camera movement → view transform → ball interpolation → team
   assignment) into a function `run_pipeline(video_path)` in a new module
   `pipeline.py` that returns `(video_frames, tracks, camera_movement_per_frame)`.
   Update `main.py` to call it so `main.py` behavior is unchanged (its ball-control +
   video-rendering part stays in `main.py`).
   - **Stub caching must be keyed to the video**: the current fixed
     `stubs/track_stubs.pkl` / `stubs/camera_movement_stub.pkl` would silently serve
     stale data for a different video. Derive stub paths from the video filename
     (e.g. `stubs/{video_stem}_tracks.pkl`, `stubs/{video_stem}_camera.pkl`). Keep
     the existing stub files usable for the sample video (rename them or special-case
     the mapping) so verification doesn't require GPU inference.
2. `generate.py` calls `run_pipeline(video_path)`, then:
   - Writes each clean frame to `editor_data/frames/frame_{i:05d}.jpg`
     (`cv2.imwrite`, JPEG quality ~85). The zero-padding width of 5 supports videos
     up to 99,999 frames; index frames by integer, never by parsing filenames.
   - Builds the seed `annotations.json` from `tracks`:
     - players: every entry in `tracks["players"][i]` → `track_id=str(tid)`,
       `team = {1: "Defence", 2: "Offence"}[track['team']]` (default `"Defence"` if
       `team` missing), bbox as-is.
     - ball: `tracks["ball"][i].get(1)` → bbox as-is, `state="Active"`, `kick=null`
       (the pipeline cannot infer state/kick; those are exactly what the human edits).
     - referees are **not** included (schema has no referee concept).
   - If `annotations.json` already exists, refuse to overwrite unless `--force` is
     passed (protects human edits).

**Verify** (with the sample video): run it; assert the number of JPEGs equals
`frame_count` in `annotations.json` (191 for the sample), that `width`/`height`
match the video (1920×1080 for the sample), spot-check a few frames have players
with plausible bboxes (0 ≤ x ≤ width, 0 ≤ y ≤ height), and that teams are only
`Offence`/`Defence`.

### Phase 2 — `editor/server.py`: Flask backend

Minimal JSON API; the frontend owns all editing logic.

- `GET /` → `static/index.html`.
- `GET /api/meta` → `{frame_count, width, height, video}` — all read from
  `annotations.json`, nothing hardcoded.
- `GET /frames/<int:i>` → the JPEG (use `send_from_directory`; validate `0 <= i < frame_count`).
- `GET /api/frame/<int:i>` → that frame's annotation object.
- `PUT /api/frame/<int:i>` → replace that frame's annotation object.
  - **Validate server-side** before accepting: bbox shape/ordering, team enum,
    ball state enum, kick shape (exactly one of byPlayerId/toPlayerId or null),
    non-empty track_id. Reject with 400 + message on violation.
  - On success, persist the whole `annotations.json` **atomically**
    (write to `annotations.json.tmp`, then `os.replace`). Per-frame PUTs are the
    autosave mechanism — there is no separate "save" concept.
- Run with `venv/bin/python -m editor.server` → serves on `127.0.0.1:5000`,
  loads `annotations.json` into memory at startup (error clearly if Phase 1 hasn't run).
- Single user, single process: a plain module-level dict + a `threading.Lock` around
  read-modify-write of the annotations is sufficient. No database.

**Verify**: with the server running, `curl` the meta endpoint, GET frame 0's image
and annotations, PUT a modified annotation, restart the server, GET it again and
confirm the edit persisted. Also confirm a malformed PUT (e.g. `team: "Red"`) gets a 400.

### Phase 3 — Frontend: `index.html`, `editor.js`, `editor.css`

This is the largest phase. Layout:

```
+------------------------------------------------------------------+
| [◀ Prev]  Frame 42 / 190  [Next ▶]     jump-to-frame input       |
+----------------+---------------------------------+---------------+
|  prev frame    |        CURRENT FRAME            |  sidebar      |
|  (small,       |   (large canvas, editable)      |  - box list   |
|   read-only)   |                                 |  - metadata   |
|  next frame    |                                 |    form for   |
|  (small,       |                                 |    selected   |
|   read-only)   |                                 |    box        |
+----------------+---------------------------------+---------------+
```

#### 3a. Rendering

- On load, fetch `/api/meta` first; every dimension-dependent calculation uses
  `meta.width`/`meta.height`. No pixel dimension may appear as a literal in the JS.
- Current frame: one `<canvas>` sized to fit the viewport (e.g. max-width ~1100px),
  drawing the frame image scaled down from its native size. Keep a single
  `scale = canvasWidth / meta.width` factor; **store all bboxes in original image
  coordinates** and convert only at draw/hit-test time. This is the most
  bug-prone area — centralize `imageToCanvas`/`canvasToImage` helpers and use them
  everywhere.
- Boxes: players in a per-team color (e.g. blue = Defence, red = Offence), ball in
  green. Label each player box with its `track_id` at the top-left corner. Selected
  box gets a highlighted stroke + 8 square resize handles (corners + edge midpoints).
- Prev/next panels: two smaller `<canvas>` elements (read-only) showing frames
  i-1 and i+1 with their boxes drawn the same way (no interaction). At the first/last
  frame, show an empty "no frame" placeholder.
- Preload neighbor images (`new Image()` on frames i-2..i+2) so navigation feels instant.

#### 3b. Interaction model (mouse on current canvas)

State machine with modes: `idle`, `drawing`, `moving`, `resizing`.

- **Select**: click inside a box selects it (topmost/smallest wins when overlapping);
  click on empty space deselects.
- **Create**: mouse-down on empty space + drag ≥ ~5px draws a new box (rubber-band
  preview); on mouse-up, create a **player** box by default with a fresh unique
  `track_id` (max numeric id across ALL frames + 1), team defaulted from... nothing
  reliable — default `"Defence"` and immediately focus the metadata form so the user
  fixes it. The form (3c) lets the user flip the box type to Ball.
- **Move**: mouse-down inside the selected box + drag moves it.
- **Resize**: mouse-down on a handle + drag resizes; normalize so x1<x2, y1<y2 on release.
- **Delete**: `Delete`/`Backspace` deletes the selected box (with the form not focused).
- Clamp boxes to image bounds [0, meta.width]×[0, meta.height].

#### 3c. Metadata form (sidebar)

Shown for the selected box:

- Common: box type selector (`Player` / `Ball`). Changing type converts the box:
  player→ball only allowed if the frame has no ball box yet (one ball per frame —
  disable the option otherwise); ball→player assigns a fresh track_id.
- Player: `track_id` text input (validate non-empty; warn — non-blocking — if it
  duplicates another player id **in the same frame**), team radio (`Offence`/`Defence`).
- Ball: state select (`Active`/`Goal`/`OutOfBounds`); kick editor: radio
  `None` / `Kicked by` / `Received by`, plus a player-id input enabled for the latter
  two (offer a datalist of track_ids present in the current frame).
- Sidebar also lists all boxes in the frame (id + team, or "Ball"); clicking a list
  item selects that box on the canvas.

#### 3d. Navigation & persistence

- `←`/`→` arrow keys and Prev/Next buttons; a numeric input to jump to a frame.
- **Autosave**: PUT the current frame's annotations (debounced ~400ms) after every
  mutation (box created/moved/resized/deleted, metadata changed), and flush any
  pending save before navigating away from a frame. Show a small "saved / saving /
  error" status indicator. On PUT failure, keep the dirty state and show the error —
  do not silently drop edits.
- Navigation loads the target frame's annotations fresh from `GET /api/frame/<i>`.

**Verify** (agent-driven, before handing to the human): use the project's browser
tooling if available (`/run` skill), else Playwright via
`venv/bin/pip install playwright` + `venv/bin/playwright install chromium`, to:
load the page, assert frame 0 renders with boxes; simulate a drag on empty canvas
and assert a new box + form appears; edit team and reload the page to confirm
persistence; navigate with arrow keys and confirm prev/next panels update.
At minimum, if browser automation is impossible, verify every API interaction with
curl and do a manual smoke checklist in the final report.

### Phase 4 — `editor/export.py`: schema-conformant export

Run as `venv/bin/python -m editor.export [-o output_videos/tracking_data.json]`.

Converts pixel-space `annotations.json` → the metric JSON of `tracking-schema.json`.
**Start from the deleted implementation**: `git show 5ad880f:tracking_output.py` —
it already solved metric conversion, center-origin offset, and gap interpolation.
Adapt it to read from `annotations.json` instead of `tracks`:

1. Recompute per-frame camera movement for the video named in `annotations.json`
   (via `CameraMovementEstimator` with `read_from_stub=True` and the same per-video
   stub path convention as Phase 1; it needs the first video frame for
   construction — read just frame 0, not the whole video, if possible).
2. Per frame, per player box: foot position (center-x, y2) → subtract camera
   movement offset → `ViewTransformer` → center-origin meters. For the ball use
   bbox center. Use the raw (unfiltered) perspective transform for the ball
   (`_raw_transform` in the recovered file) so an out-of-bounds ball still gets
   coordinates; keep the polygon filter for players and interpolate interior gaps
   (`_interpolate_gaps`). A player with no valid position in a frame is omitted
   from that frame's `players` array.
3. Top-level `players` list: unique ids across frames, `role: "Outfield"` for all
   (the tracker converts goalkeepers to players, so `GoalKeeper` cannot be inferred;
   note this limitation in the export's docstring).
4. Ball `state` and `kick` pass through from annotations.
5. **Generalization caveat**: the metric conversion is only as good as
   `ViewTransformer.pixel_vertices`, which are calibrated for the sample video's
   camera view. The pixel-space editor (Phases 1–3) is fully video-agnostic; only
   this export step inherits the pipeline's per-video calibration. Do not try to
   auto-calibrate — just document in the export's docstring/README that
   `ViewTransformer` must be recalibrated per camera setup, exactly as `main.py`
   already requires.

**Verify**: `venv/bin/pip install jsonschema`, then validate the export against
`tracking-schema.json` programmatically. Also sanity-check (sample video): the
export's frame count equals `annotations.json`'s `frame_count`, and on-pitch player
coordinates fall within |x| ≤ ~15 m, |y| ≤ ~40 m (pitch is 23.32 m × 68 m in this
transform).

### Phase 5 — Docs & wrap-up

1. Update `EDITOR.md` (or add a Usage section to `README.md`) with the three
   commands: generate → serve/edit → export.
2. Run the full loop once end-to-end: generate, start server, exercise the API,
   export, validate. Report results honestly, including anything unverified.
3. Do NOT commit unless asked. Note: `stubs/*.pkl` show as modified in git status —
   that predates this work; leave them alone.

## 5. Pitfalls checklist (read again before Phase 3)

- **No hardcoded video specifics anywhere**: path, width, height, fps, and frame
  count all flow from `generate.py --video` into `annotations.json` and are read
  from there by the server, frontend, and exporter. Grep the finished code for
  `1920`, `1080`, `191`, and the sample filename — the only acceptable hits are
  defaults/docs pointing at the sample video and the untouched pipeline modules.

- Ball track key in `tracks["ball"]` is the literal int `1`, and interpolation means
  a ball bbox exists on ~every frame — but it may be garbage on some; that's what
  the editor is for.
- Team ints: `1 = Defence`, `2 = Offence`. Don't guess it the other way — it came
  from `tracking_output.py`.
- `ViewTransformer` attribute is misspelled `persepctive_trasnformer`; don't "fix"
  the spelling in place unless you update all callers.
- Coordinate scaling between canvas and image space: one shared helper pair, used
  by draw, hit-test, drag, and resize alike.
- Atomic JSON writes (`.tmp` + `os.replace`) — a crash mid-write must not destroy
  hours of human labeling.
- JSON keys from `PUT` bodies arrive as strings; frame indices in URLs as ints —
  be consistent (annotations store per-frame arrays, so no dict-key issues if the
  format in §3.3 is followed exactly).
- `main.py` must still work after the Phase 1 refactor — run it (stubs make it fast)
  and confirm `output_videos/output_video.avi` is regenerated.

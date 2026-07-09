# Editor

The editor is an interactive app that lets a human user correct
the the inferred tracking data by updating bounding boxes on images.

## Definition

See tracking-schema.json for information we need

- Player bounding box - identifies a player on the pitch. The box should include information about the coordinates of the player, the team of the player (offence or defence), and a unique tracking id
- Ball bounding box - identifies the ball on the pitch. the box should include information about the coordinates of the ball, its state: Active, Goal, or OutOfBounds, and whether it's being kicked or received by a player.

## Workflow

1. main.py generates tracking data and creates one image per frame, with
   bounding boxes drawn around ball and all players

2. Allow human user to go through each frame and add/remove/edit bounding boxes.
   Human should be able to create bounding box by dragging it on the image, and
   human should be allowed to enter/edit the metadata associated with each box.

3. To assist with editing, show the frame immediately before and after the current
   frame being edited, along with all the bounding boxes on the before/after frames.

## Usage

The editor is a three-step workflow. All commands run from the repo root using
the project virtualenv (`venv/bin/python`). Nothing about the video (path,
resolution, fps, frame count) is hardcoded: it flows from `--video` into
`editor_data/annotations.json` and is read from there by every later step.

### 1. Generate — frames + seed annotations

```
venv/bin/python -m editor.generate [--video input_videos/<your-video>.mp4] [--force]
```

Runs the tracking pipeline (`pipeline.run_pipeline`, shared with `main.py`),
writes one clean JPEG per frame to `editor_data/frames/`, and seeds
`editor_data/annotations.json` (pixel-space bounding boxes + metadata). `--video`
defaults to the sample clip, so the sample workflow needs no arguments. Refuses
to overwrite an existing `annotations.json` unless `--force` is given (protects
human edits). Stub caches are keyed to the video filename
(`stubs/<stem>_tracks.pkl`, `stubs/<stem>_camera.pkl`); the sample video's stubs
ship in the repo so this runs without GPU inference.

### 2. Serve / edit — the interactive editor

```
venv/bin/python -m editor.server
```

Serves the editor on http://127.0.0.1:5000. Open it in a browser and step
through frames with the Prev/Next buttons, the ← / → arrow keys, or the
jump-to-frame input. The frame being edited is shown large and editable; the
frames immediately before and after are shown as read-only context with their
boxes. For each frame you can:

- **Create** a box by dragging on empty image space (creates a player box with a
  fresh unique track id; edit its metadata in the sidebar).
- **Select** a box by clicking it (or a row in the sidebar box list).
- **Move** a selected box by dragging it; **resize** via its 8 handles.
- **Delete** the selected box with the Delete/Backspace key.
- Edit metadata in the sidebar: player track id + team (Offence/Defence),
  goalkeeper flag, or ball state (Active/Goal/OutOfBounds) and kick (none /
  kicked-by / received-by a player id). Switch a box between Player and Ball with
  the type selector.

**Attribute edits persist forward.** Editing a *persistent* box attribute —
player team, goalkeeper, track id (a rename), or ball state — applies it to the
current frame **and every subsequent frame** (players are matched by track id;
ball state to each later frame that has a ball), so you correct it once rather
than on every frame. Bounding-box position/size are per-frame by nature, and the
ball **kick** is an instantaneous event, so those are *not* propagated. The
top-bar **⇄ Invert teams** button swaps Offence/Defence for every player in every
frame at once.

Edits autosave per frame (a debounced `PUT`, written atomically to
`annotations.json`); the header shows a saved / saving / error indicator.

### 3. Export — schema-conformant tracking data

```
venv/bin/python -m editor.export [-o output_videos/tracking_data.json]
```

Converts the pixel-space `annotations.json` into the metric JSON described by
`tracking-schema.json`. Player foot positions and the ball center are
camera-motion-adjusted and run through `ViewTransformer`; interior gaps are
interpolated; ball state and kick pass through from your edits.

**Coordinate system:** meters from the **center of a standard 105 m × 68 m
pitch**. `(0, 0)` is the pitch center; `-x` toward the left goal, `+x` toward the
right goal; `-y` toward the top touchline, `+y` toward the bottom touchline. The
`ViewTransformer` homography yields real meters with its origin at the near
goal-line/touchline pitch corner, so the exporter subtracts half a pitch
(`PITCH_LENGTH/2`, `PITCH_WIDTH/2`) to recenter to the pitch center.

**Calibration caveat:** the metric conversion depends on
`ViewTransformer.pixel_vertices`, which is calibrated for the sample video's
camera view (a trapezoid from the left goal line spanning ~23 m of length × the
full 68 m width). The recentering assumes that trapezoid's near corner is a true
pitch corner. The editor itself (steps 1–2) is fully video-agnostic, but this
export step must be recalibrated per camera setup — exactly as `main.py` already
requires. Players whose foot position falls outside the calibrated trapezoid have
no valid metric position and are omitted from a frame (and, if never on it, from
the top-level `players` list).

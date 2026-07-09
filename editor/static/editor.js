"use strict";

// ---------------------------------------------------------------------------
// Tracking Editor frontend. All bboxes are stored in ORIGINAL image pixel
// coordinates; conversion to/from canvas space happens only at draw / hit-test
// time via imageToCanvas / canvasToImage. No image dimension is hardcoded --
// everything derives from /api/meta.
// ---------------------------------------------------------------------------

const COLORS = {
  Defence: "#2d6cdf",
  Offence: "#e5484d",
  ball: "#30a46c",
  selected: "#ffd60a",
  exclusion: "#e5484d",
};
const NEIGHBOR_H = 150;      // px, drawing height of the prev/next frames
const HANDLE_SIZE = 8;       // px, canvas space
const HANDLE_HIT = 7;        // px hit radius, canvas space
const MIN_DRAG = 5;          // px, canvas space, to count as a new box
const SAVE_DEBOUNCE_MS = 400;

// 8 resize handles: fractional position within the box. Center excluded.
const HANDLES = [
  { fx: 0, fy: 0 }, { fx: 0.5, fy: 0 }, { fx: 1, fy: 0 },
  { fx: 0, fy: 0.5 },                    { fx: 1, fy: 0.5 },
  { fx: 0, fy: 1 }, { fx: 0.5, fy: 1 }, { fx: 1, fy: 1 },
];

const state = {
  meta: null,
  scale: 1,
  idx: 0,
  frame: null,       // current annotation { players:[], ball:null|{} }
  prevFrame: null,
  nextFrame: null,
  selected: null,    // { type:'player', index:int } | { type:'ball' } | { type:'area', index:int } | null
  selectedTrackId: null,  // committed track_id of the selected player (for rename propagation)
  mode: "idle",      // idle | drawing | moving | resizing
  drag: null,        // per-mode scratch data (image coords); drawing.drag.kind in {'box'|'area'}
  images: new Map(),  // idx -> Image
  saveTimer: null,
  dirty: false,
  saving: false,
  exclusionAreas: [],   // [{id:int, bbox:[x1,y1,x2,y2]}] -- global, applies to every frame
  areaDrawMode: false,  // when true, drag on the canvas draws a new exclusion area
  calibration: { pitch_length: 105, pitch_width: 68, points: [] },  // global per-video; loaded from /api/calibration
  calibMode: false,     // when true, canvas clicks add calibration points and the sidebar shows the calibration panel
  calibDirty: false,
  ballTrajectories: [], // [{id, start_frame, end_frame, max_height}] -- global per-video
  arcMode: false,       // when true, the sidebar shows the ball-trajectory panel
  coreIds: [],           // [track_id, ...] -- track_ids expected in every frame, global per-video
  coreMode: false,        // when true, the sidebar shows the core-players panel
  boxFilter: "",          // lowercased search text filtering "Boxes in this frame"
};

const el = {};       // cached DOM nodes

// ---- coordinate helpers ---------------------------------------------------
const imgToCan = (v) => v * state.scale;
const canToImg = (v) => v / state.scale;

function clampX(x) { return Math.max(0, Math.min(state.meta.width, x)); }
function clampY(y) { return Math.max(0, Math.min(state.meta.height, y)); }

function normalizeBox(b) {
  let [x1, y1, x2, y2] = b;
  if (x1 > x2) [x1, x2] = [x2, x1];
  if (y1 > y2) [y1, y2] = [y2, y1];
  return [clampX(x1), clampY(y1), clampX(x2), clampY(y2)];
}

// ---- image loading / preload ---------------------------------------------
function getImage(idx) {
  if (idx < 0 || idx >= state.meta.frame_count) return null;
  if (state.images.has(idx)) return state.images.get(idx);
  const img = new Image();
  img.src = `/frames/${idx}`;
  state.images.set(idx, img);
  return img;
}

function preloadNeighbors() {
  for (let d = -2; d <= 2; d++) getImage(state.idx + d);
}

// ---- drawing --------------------------------------------------------------
function drawAreaOn(ctx, scale, bbox, opts = {}) {
  const [x1, y1, x2, y2] = bbox;
  const w = (x2 - x1) * scale, h = (y2 - y1) * scale;
  ctx.save();
  ctx.fillStyle = "rgba(229, 72, 77, 0.18)";
  ctx.fillRect(x1 * scale, y1 * scale, w, h);
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = opts.selected ? 3 : 2;
  ctx.strokeStyle = opts.selected ? COLORS.selected : COLORS.exclusion;
  ctx.strokeRect(x1 * scale, y1 * scale, w, h);
  ctx.setLineDash([]);
  ctx.font = "12px sans-serif";
  const label = "Exclusion";
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = COLORS.exclusion;
  ctx.fillRect(x1 * scale, y1 * scale - 15, tw + 8, 15);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x1 * scale + 4, y1 * scale - 4);
  if (opts.selected) {
    ctx.fillStyle = COLORS.selected;
    for (const h of HANDLES) {
      const hx = (x1 + h.fx * (x2 - x1)) * scale;
      const hy = (y1 + h.fy * (y2 - y1)) * scale;
      ctx.fillRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
    }
  }
  ctx.restore();
}

function drawBoxOn(ctx, scale, bbox, color, opts = {}) {
  const [x1, y1, x2, y2] = bbox;
  ctx.lineWidth = opts.selected ? 3 : 2;
  ctx.strokeStyle = opts.selected ? COLORS.selected : color;
  ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
  if (opts.label != null) {
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(opts.label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1 * scale, y1 * scale - 15, tw + 8, 15);
    ctx.fillStyle = "#fff";
    ctx.fillText(opts.label, x1 * scale + 4, y1 * scale - 4);
  }
  if (opts.selected) {
    ctx.fillStyle = COLORS.selected;
    for (const h of HANDLES) {
      const hx = (x1 + h.fx * (x2 - x1)) * scale;
      const hy = (y1 + h.fy * (y2 - y1)) * scale;
      ctx.fillRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
    }
  }
}

function renderFrameTo(canvas, idx, annotation, opts = {}) {
  const ctx = canvas.getContext("2d");
  const scale = canvas.width / state.meta.width;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (idx < 0 || idx >= state.meta.frame_count || !annotation) {
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "14px sans-serif";
    ctx.fillText("no frame", 10, 22);
    return;
  }

  const img = getImage(idx);
  const paint = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img && img.complete && img.naturalWidth) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
}
    // Exclusion areas live below boxes and are global (shown on every frame).
    for (let i = 0; i < state.exclusionAreas.length; i++) {
      const a = state.exclusionAreas[i];
      const sel = opts.selectable && state.selected &&
                  state.selected.type === "area" && state.selected.index === i;
      drawAreaOn(ctx, scale, a.bbox, { selected: sel });
    }
    for (let i = 0; i < annotation.players.length; i++) {
      const p = annotation.players[i];
      const sel = opts.selectable && state.selected &&
                  state.selected.type === "player" && state.selected.index === i;
      const label = p.role === "GoalKeeper" ? `${p.track_id} GK` : p.track_id;
      drawBoxOn(ctx, scale, p.bbox, COLORS[p.team] || COLORS.Defence,
                 { label, selected: sel });
    }
    if (annotation.ball) {
      const sel = opts.selectable && state.selected && state.selected.type === "ball";
      drawBoxOn(ctx, scale, annotation.ball.bbox, COLORS.ball,
                 { label: "ball", selected: sel });
    }
    // Ball-in-air badge (current z derived from trajectories), only on the
    // main canvas and only when the trajectory panel is open.
    if (opts.selectable && state.arcMode && annotation === state.frame) {
      drawArcBadge(ctx, scale);
    }
    // rubber-band preview for a box (or exclusion area) being drawn
    if (opts.selectable && state.mode === "drawing" && state.drag) {
      const b = state.drag.box;
      const isArea = state.drag.kind === "area";
      if (isArea) {
        drawAreaOn(ctx, scale, b, {});
      } else {
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.strokeRect(b[0] * scale, b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale);
        ctx.setLineDash([]);
      }
    }
    // Calibration points: numbered markers, shown only on the main canvas
    // (annotation === state.frame) and only in calibration mode. They are
    // referenced to frame 0's pixel space (camera movement is zero there).
    if (state.calibMode && annotation === state.frame) {
      ctx.save();
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      state.calibration.points.forEach((p, i) => {
        const cx = p.pixel[0] * scale, cy = p.pixel[1] * scale;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffd60a";
        ctx.fill();
        ctx.strokeStyle = "#1f2733";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "#1f2733";
        ctx.fillText(String(i + 1), cx, cy + 0.5);
      });
      ctx.restore();
    }
  };

  if (img && !img.complete) img.onload = paint;
  paint();
}

function renderAll() {
  renderFrameTo(el.mainCanvas, state.idx, state.frame, { selectable: true });
  renderFrameTo(el.prevCanvas, state.idx - 1, state.prevFrame);
  renderFrameTo(el.nextCanvas, state.idx + 1, state.nextFrame);
}

// ---- hit testing ----------------------------------------------------------
function boxAtPoint(px, py) {
  // Returns selection descriptor for the smallest box containing (px,py) in
  // image coords; players and ball considered. Smallest area wins on overlap.
  let best = null, bestArea = Infinity;
  const consider = (bbox, sel) => {
    const [x1, y1, x2, y2] = bbox;
    if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < bestArea) { bestArea = area; best = sel; }
    }
  };
  state.frame.players.forEach((p, i) => consider(p.bbox, { type: "player", index: i }));
  if (state.frame.ball) consider(state.frame.ball.bbox, { type: "ball" });
  return best;
}

// Areas are a fallback selection: preferred only when no box is hit at this
// point. Smallest containing area wins, so nested areas select the innermost.
function areaAtPoint(px, py) {
  let best = null, bestArea = Infinity;
  for (let i = 0; i < state.exclusionAreas.length; i++) {
    const [x1, y1, x2, y2] = state.exclusionAreas[i].bbox;
    if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < bestArea) { bestArea = area; best = { type: "area", index: i }; }
    }
  }
  return best;
}

// True when ``bbox`` is fully contained inside ANY exclusion area. Used to
// enforce the invariant at edit time (before the server-side purge backstop).
function fullyContainedInAny(bbox) {
  if (!state.exclusionAreas.length) return false;
  const [bx1, by1, bx2, by2] = bbox;
  for (const a of state.exclusionAreas) {
    const [ax1, ay1, ax2, ay2] = a.bbox;
    if (ax1 <= bx1 && ay1 <= by1 && ax2 >= bx2 && ay2 >= by2) return true;
  }
  return false;
}

function selectedBox() {
  if (!state.selected) return null;
  if (state.selected.type === "ball") return state.frame.ball;
  if (state.selected.type === "area") return state.exclusionAreas[state.selected.index];
  return state.frame.players[state.selected.index];
}

function handleAtPoint(px, py) {
  // px,py in image coords. Only for the currently selected box.
  const box = selectedBox();
  if (!box) return null;
  const [x1, y1, x2, y2] = box.bbox;
  for (let i = 0; i < HANDLES.length; i++) {
    const h = HANDLES[i];
    const hx = x1 + h.fx * (x2 - x1);
    const hy = y1 + h.fy * (y2 - y1);
    if (Math.abs(imgToCan(px - hx)) <= HANDLE_HIT && Math.abs(imgToCan(py - hy)) <= HANDLE_HIT) {
      return { handle: h };
    }
  }
  return null;
}

function mouseToImage(ev) {
  const rect = el.mainCanvas.getBoundingClientRect();
  const cx = (ev.clientX - rect.left) * (el.mainCanvas.width / rect.width);
  const cy = (ev.clientY - rect.top) * (el.mainCanvas.height / rect.height);
  return { x: canToImg(cx), y: canToImg(cy) };
}

// ---- mouse interaction ----------------------------------------------------
function onMouseDown(ev) {
  const { x, y } = mouseToImage(ev);

  // In calibration mode every click on the main canvas adds a calibration
  // point (pixel -> pitch correspondence). Selection / drawing / moving are
  // suspended while calibrating. Points are valid only on frame 0; the mode
  // toggle warns the human if they are elsewhere.
  if (state.calibMode) {
    addCalibPointAt(x, y);
    return;
  }

  // In area-draw mode every drag becomes a new exclusion area, regardless of
  // what is underneath (boxes/areas are not selectable in this mode). Toggle
  // the button off to edit existing selection again.
  if (state.areaDrawMode) {
    state.mode = "drawing";
    state.drag = { kind: "area", startX: x, startY: y, box: [x, y, x, y] };
    return;
  }

  // resize? (only if a box is selected and we hit one of its handles)
  const h = handleAtPoint(x, y);
  if (h) {
    state.mode = "resizing";
    state.drag = { handle: h.handle, box: selectedBox().bbox.slice() };
    return;
  }

  const hit = boxAtPoint(x, y);
  if (hit) {
    setSelection(hit);
    state.mode = "moving";
    state.drag = { startX: x, startY: y, box: selectedBox().bbox.slice() };
    renderAll();
    return;
  }

  // An exclusion area beneath the cursor is selectable when no box is.
  const ahit = areaAtPoint(x, y);
  if (ahit) {
    setSelection(ahit);
    state.mode = "moving";
    state.drag = { startX: x, startY: y, box: selectedBox().bbox.slice() };
    renderAll();
    return;
  }

  // empty space -> begin drawing a new box
  state.mode = "drawing";
  state.drag = { kind: "box", startX: x, startY: y, box: [x, y, x, y] };
}

function onMouseMove(ev) {
  if (state.mode === "idle") return;
  const { x, y } = mouseToImage(ev);

  if (state.mode === "drawing") {
    state.drag.box = [state.drag.startX, state.drag.startY, x, y];
    renderAll();
  } else if (state.mode === "moving") {
    const dx = x - state.drag.startX, dy = y - state.drag.startY;
    let [x1, y1, x2, y2] = state.drag.box;
    const w = x2 - x1, hgt = y2 - y1;
    let nx1 = clampX(x1 + dx), ny1 = clampY(y1 + dy);
    // keep width/height when clamping against far edges
    nx1 = Math.min(nx1, state.meta.width - w);
    ny1 = Math.min(ny1, state.meta.height - hgt);
    nx1 = Math.max(0, nx1); ny1 = Math.max(0, ny1);
    selectedBox().bbox = [nx1, ny1, nx1 + w, ny1 + hgt];
    renderAll();
  } else if (state.mode === "resizing") {
    const h = state.drag.handle;
    let [x1, y1, x2, y2] = selectedBox().bbox;
    if (h.fx === 0) x1 = clampX(x); else if (h.fx === 1) x2 = clampX(x);
    if (h.fy === 0) y1 = clampY(y); else if (h.fy === 1) y2 = clampY(y);
    selectedBox().bbox = [x1, y1, x2, y2];
    renderAll();
  }
}

async function onMouseUp(ev) {
  if (state.mode === "drawing") {
    const b = state.drag.box;
    const dragPx = Math.max(Math.abs(imgToCan(b[2] - b[0])), Math.abs(imgToCan(b[3] - b[1])));
    const kind = state.drag.kind;
    state.mode = "idle";
    if (dragPx >= MIN_DRAG) {
      const norm = normalizeBox(b);
      if (kind === "area") await createArea(norm);
      else await createBox(norm);
    } else {
      // treated as a click on empty space -> deselect
      setSelection(null);
      renderAll();
    }
    state.drag = null;
    return;
  }

  if (state.mode === "moving" || state.mode === "resizing") {
    const box = selectedBox();
    if (box) box.bbox = normalizeBox(box.bbox);
    state.mode = "idle";
    state.drag = null;
    renderAll();

    // An exclusion area edit is persisted via its own endpoint (and triggers a
    // server-side purge), not the per-frame PUT.
    if (state.selected && state.selected.type === "area") {
      await updateArea();
      return;
    }

    // Enforce the containment invariant locally: if a box was dragged /
    // resized so it now sits fully inside an exclusion area, drop it. The
    // server's purge on PUT is a backstop for any races.
    if (state.selected) {
      const sel = state.selected;
      if (sel.type === "player") {
        const p = state.frame.players[sel.index];
        if (p && fullyContainedInAny(p.bbox)) {
          state.frame.players.splice(sel.index, 1);
          setSelection(null);
          renderAll();
        }
      } else if (sel.type === "ball" && state.frame.ball && fullyContainedInAny(state.frame.ball.bbox)) {
        state.frame.ball = null;
        setSelection(null);
        renderAll();
      }
    }
    markDirty();
    return;
  }
  state.mode = "idle";
  state.drag = null;
}

async function createBox(bbox) {
  // Boxes drawn inside an exclusion area are never created: the invariant must
  // hold at every moment, so reject the drag silently.
  if (fullyContainedInAny(bbox)) {
    renderAll();
    return;
  }

  // Default new boxes to "ball"; only one ball allowed per frame, so fall
  // back to creating a player once a ball already exists.
  if (!state.frame.ball) {
    state.frame.ball = { bbox, state: "Active", kick: null };
    setSelection({ type: "ball" });
    renderAll();
    markDirty();
    return;
  }

  let newId = "1";
  try {
    const r = await fetch("/api/next_track_id");
    newId = (await r.json()).next_id;
  } catch (e) { /* fall back to "1"; user can edit */ }
  state.frame.players.push({ track_id: newId, team: "Defence", bbox });
  setSelection({ type: "player", index: state.frame.players.length - 1 });
  renderAll();
  markDirty();
  el.trackIdInput.focus();
  el.trackIdInput.select();
}

// ---- exclusion areas -----------------------------------------------------
function toggleAreaMode() {
  state.areaDrawMode = !state.areaDrawMode;
  el.areaBtn.classList.toggle("active", state.areaDrawMode);
}

async function createArea(bbox) {
  setSaveStatus("saving");
  try {
    const r = await fetch("/api/exclusion_areas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox }),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Create area failed: ${msg}`;
      return;
    }
    const data = await r.json();
    state.exclusionAreas.push({ id: data.id, bbox });
    setSaveStatus("saved");
    el.saveStatus.title = "";
    // Server purged every frame; reload current + neighbors so the preview
    // reflects the same state the server now holds.
    await reloadAround();
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Create area failed: ${e}`;
  }
}

async function updateArea() {
  const sel = state.selected;
  if (!sel || sel.type !== "area") return;
  const a = state.exclusionAreas[sel.index];
  setSaveStatus("saving");
  try {
    const r = await fetch(`/api/exclusion_areas/${a.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox: a.bbox.map(Number) }),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Update area failed: ${msg}`;
      return;
    }
    setSaveStatus("saved");
    el.saveStatus.title = "";
    await reloadAround();
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Update area failed: ${e}`;
  }
}

async function deleteAreaSelected() {
  const sel = state.selected;
  if (!sel || sel.type !== "area") return;
  const a = state.exclusionAreas[sel.index];
  // Optimistic removal; previously-purged boxes are intentionally not restored.
  state.exclusionAreas.splice(sel.index, 1);
  setSelection(null);
  renderAll();
  setSaveStatus("saving");
  try {
    const r = await fetch(`/api/exclusion_areas/${a.id}`, { method: "DELETE" });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Delete area failed: ${msg}`;
      return;
    }
    setSaveStatus("saved");
    el.saveStatus.title = "";
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Delete area failed: ${e}`;
  }
}

// ---- pitch calibration ---------------------------------------------------
// Global per-video pixel -> pitch correspondences. The homography is built
// from these (>=4 non-collinear). They are referenced to frame 0's pixel
// space -- camera movement is zero there and the exporter subtracts the
// per-frame camera offset from every later box before applying the transform,
// so calibrating on frame 0 covers all frames.
async function loadCalibration() {
  let cal = state.meta && state.meta.calibration;
  if (!cal) {
    try { cal = await (await fetch("/api/calibration")).json(); }
    catch (e) { cal = null; }
  }
  state.calibration = cal
    ? { pitch_length: cal.pitch_length, pitch_width: cal.pitch_width,
        points: (cal.points || []).map((p) => ({
          pixel: [Number(p.pixel[0]), Number(p.pixel[1])],
          pitch: [Number(p.pitch[0]), Number(p.pitch[1])],
        })) }
    : { pitch_length: 105, pitch_width: 68, points: [] };
  state.calibDirty = false;
}

async function toggleCalibMode() {
  state.calibMode = !state.calibMode;
  el.calibBtn.classList.toggle("active", state.calibMode);
  el.calibSection.classList.toggle("hidden", !state.calibMode);
  // Leaving the box form's radio inputs / area fields visible alongside the
  // calibration panel would be confusing; drop any selection.
  setSelection(null);
  if (state.calibMode) {
    if (state.idx !== 0) {
      if (confirm("Calibration points must be marked on frame 0 (camera movement is referenced there). Jump to frame 0 now?")) {
        await loadFrame(0);
      }
    }
    syncCalibPanel();
  }
  renderAll();
}

function markCalibDirty() {
  state.calibDirty = true;
  if (el.calibWarn) el.calibWarn.textContent = "Unsaved changes.";
}

function addCalibPointAt(x, y) {
  // Default pitch coord = pitch center; refined via the panel inputs / preset.
  const L = parseFloat(el.calibLen.value) || state.calibration.pitch_length;
  const W = parseFloat(el.calibWid.value) || state.calibration.pitch_width;
  state.calibration.points.push({
    pixel: [Math.round(x * 100) / 100, Math.round(y * 100) / 100],
    pitch: [L / 2, W / 2],
  });
  markCalibDirty();
  syncCalibPanel();
  renderAll();
}

function syncCalibPanel() {
  el.calibLen.value = state.calibration.pitch_length;
  el.calibWid.value = state.calibration.pitch_width;
  el.calibWarn.textContent = state.calibDirty ? "Unsaved changes." : "";
  el.calibList.innerHTML = "";
  state.calibration.points.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "calib-pt";
    const idx = document.createElement("span");
    idx.className = "calib-idx"; idx.textContent = String(i + 1);
    const pix = document.createElement("span");
    pix.className = "calib-pix muted";
    pix.textContent = `px (${Math.round(p.pixel[0])}, ${Math.round(p.pixel[1])})`;
    const grp = document.createElement("div");
    grp.className = "calib-coords";
    const xlab = document.createElement("label"); xlab.textContent = "x";
    const xin = document.createElement("input");
    xin.type = "number"; xin.step = "0.01"; xin.value = p.pitch[0];
    const ylab = document.createElement("label"); ylab.textContent = "y";
    const yin = document.createElement("input");
    yin.type = "number"; yin.step = "0.01"; yin.value = p.pitch[1];
    xin.oninput = () => { p.pitch[0] = parseFloat(xin.value) || 0; markCalibDirty(); };
    yin.oninput = () => { p.pitch[1] = parseFloat(yin.value) || 0; markCalibDirty(); };
    grp.appendChild(xlab); grp.appendChild(xin); grp.appendChild(ylab); grp.appendChild(yin);
    const del = document.createElement("button");
    del.type = "button"; del.textContent = "✕"; del.className = "calib-del";
    del.onclick = () => {
      state.calibration.points.splice(i, 1);
      markCalibDirty();
      syncCalibPanel();
      renderAll();
    };
    li.appendChild(idx); li.appendChild(pix); li.appendChild(grp); li.appendChild(del);
    el.calibList.appendChild(li);
  });
  const n = state.calibration.points.length;
  el.calibHint.textContent = n === 0
    ? "Click on the frame to add a point."
    : `${n} point(s). Need >= 4. Click on the frame to add more.`;
}

async function saveCalibration() {
  const L = parseFloat(el.calibLen.value);
  const W = parseFloat(el.calibWid.value);
  if (!(L > 0) || !(W > 0)) {
    el.calibWarn.textContent = "Pitch length and width must be positive.";
    return;
  }
  if (state.calibration.points.length < 4) {
    el.calibWarn.textContent = "Need at least 4 pixel -> pitch correspondences.";
    return;
  }
  state.calibration.pitch_length = L;
  state.calibration.pitch_width = W;
  setSaveStatus("saving");
  try {
    const r = await fetch("/api/calibration", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.calibration),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.calibWarn.textContent = `Save failed: ${msg}`;
      return;
    }
    state.calibDirty = false;
    el.calibWarn.textContent = "Saved.";
    setSaveStatus("saved");
  } catch (e) {
    setSaveStatus("error");
    el.calibWarn.textContent = `Save failed: ${e}`;
  }
}

function applyCalibPreset() {
  const v = el.calibPreset.value;
  el.calibPreset.value = "";
  if (!v) return;
  const pts = state.calibration.points;
  if (pts.length === 0) {
    el.calibWarn.textContent = "Add a point on the frame first, then pick a preset.";
    return;
  }
  const [px, py] = v.split(",").map(Number);
  pts[pts.length - 1].pitch = [px, py];
  markCalibDirty();
  syncCalibPanel();
  renderAll();
}

// ---- ball trajectories ---------------------------------------------------
// Global per-video list of frame ranges where the ball is airborne, with a
// max height. The exporter interpolates a parabola (z=0 at endpoints, max at
// midpoint) for each range. Non-overlapping (enforced server-side), so every
// frame has at most one z value.
function toggleArcMode() {
  state.arcMode = !state.arcMode;
  el.arcBtn.classList.toggle("active", state.arcMode);
  el.arcSection.classList.toggle("hidden", !state.arcMode);
  setSelection(null);
  if (state.arcMode) syncArcPanel();
  renderAll();
}

function loadBallTrajectories() {
  const t = (state.meta && state.meta.ball_trajectories) || [];
  state.ballTrajectories = t.map((x) => ({
    id: x.id,
    start_frame: Number(x.start_frame),
    end_frame: Number(x.end_frame),
    max_height: Number(x.max_height),
  }));
}

function syncArcPanel() {
  el.arcList.innerHTML = "";
  el.arcWarn.textContent = "";
  el.arcStart.value = state.idx;
  el.arcEnd.value = Math.min(state.idx + 5, state.meta.frame_count - 1);
  el.arcHeight.value = "";
  // Pre-fill start/end from the current frame, as a convenience.
  state.ballTrajectories.forEach((t) => {
    const li = document.createElement("li");
    li.className = "arc-pt";
    const mid = (t.start_frame + t.end_frame) / 2;
    const peak = 4 * t.max_height * 0.5 * 0.5;
    const lbl = document.createElement("span");
    lbl.textContent = `#${t.id}: frames ${t.start_frame}\u2013${t.end_frame} · h=${t.max_height} m · peak z≈${peak.toFixed(2)} m (mid=${mid.toFixed(0)})`;
    const del = document.createElement("button");
    del.type = "button"; del.textContent = "\u2715"; del.className = "calib-del";
    del.onclick = () => deleteTrajectory(t.id);
    li.appendChild(lbl); li.appendChild(del);
    el.arcList.appendChild(li);
  });
}

async function addTrajectory() {
  const s = parseInt(el.arcStart.value, 10);
  const e = parseInt(el.arcEnd.value, 10);
  const h = parseFloat(el.arcHeight.value);
  if (Number.isNaN(s) || Number.isNaN(e)) { el.arcWarn.textContent = "Start and end frame are required."; return; }
  if (Number.isNaN(h) || h <= 0) { el.arcWarn.textContent = "Max height must be a positive number."; return; }
  if (e <= s) { el.arcWarn.textContent = "End frame must be greater than start frame."; return; }
  setSaveStatus("saving");
  try {
    const r = await fetch("/api/ball_trajectories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_frame: s, end_frame: e, max_height: h }),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error"); el.arcWarn.textContent = `Add failed: ${msg}`; return;
    }
    const data = await r.json();
    state.ballTrajectories.push(data);
    setSaveStatus("saved"); el.arcWarn.textContent = "";
    syncArcPanel(); renderAll();
  } catch (err) {
    setSaveStatus("error"); el.arcWarn.textContent = `Add failed: ${err}`;
  }
}

async function deleteTrajectory(id) {
  setSaveStatus("saving");
  try {
    const r = await fetch(`/api/ball_trajectories/${id}`, { method: "DELETE" });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error"); el.arcWarn.textContent = `Delete failed: ${msg}`; return;
    }
  state.ballTrajectories = state.ballTrajectories.filter((t) => t.id !== id);
    setSaveStatus("saved"); el.arcWarn.textContent = "";
    syncArcPanel(); renderAll();
  } catch (err) {
    setSaveStatus("error"); el.arcWarn.textContent = `Delete failed: ${err}`;
  }
}

// Render the ball's current z on the main canvas, derived from the
// trajectories, so the human can see the height profile while scrubbing. Drawn
// as a small "z=h" badge next to the ball box when the panel is open.
function drawArcBadge(ctx, scale) {
  if (!state.arcMode || !state.frame || !state.frame.ball) return;
  const z = ballHeightAt(state.idx);
  if (z <= 0.001) return;
  const b = state.frame.ball.bbox;
  const bx = (b[0] + b[2]) / 2 * scale, by = b[1] * scale;
  ctx.save();
  ctx.font = "bold 12px sans-serif";
  const txt = `z=${z.toFixed(2)} m`;
  const tw = ctx.measureText(txt).width;
  ctx.fillStyle = "#30a46c";
  ctx.fillRect(bx - tw/2 - 4, by - 18, tw + 8, 15);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(txt, bx, by - 10.5);
  ctx.restore();
}

function ballHeightAt(idx) {
  for (const t of state.ballTrajectories) {
    if (t.start_frame <= idx && idx <= t.end_frame) {
      if (t.end_frame === t.start_frame) return 0;
      const s = (idx - t.start_frame) / (t.end_frame - t.start_frame);
      return 4 * t.max_height * s * (1 - s);
    }
  }
  return 0;
}

// ---- core players ----------------------------------------------------------
// Global per-video list of track_ids expected in every frame. Missing/extra
// status for the current frame is computed client-side (the frame is already
// loaded); jumping to the nearest frame with an issue requires a server round
// trip since only the current frame + 2 neighbors are loaded in the browser.
function loadCoreIds() {
  state.coreIds = (state.meta.core_player_ids || []).slice();
}

function toggleCoreMode() {
  state.coreMode = !state.coreMode;
  el.coreBtn.classList.toggle("active", state.coreMode);
  el.coreSection.classList.toggle("hidden", !state.coreMode);
  if (state.coreMode) syncCorePanel();
  renderAll();
}

function syncCorePanel() {
  el.coreIdsInput.value = state.coreIds.join(", ");
  el.coreWarn.textContent = "";
  updateCoreStatus();
}

function parseCoreIdsInput() {
  return el.coreIdsInput.value.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
}

async function saveCoreIds() {
  const ids = parseCoreIdsInput();
  const seen = new Set();
  for (const id of ids) {
    if (seen.has(id)) { el.coreWarn.textContent = `Duplicate id: ${id}`; return; }
    seen.add(id);
  }
  setSaveStatus("saving");
  try {
    const r = await fetch("/api/core_player_ids", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error"); el.coreWarn.textContent = `Save failed: ${msg}`; return;
    }
    state.coreIds = ids;
    setSaveStatus("saved"); el.coreWarn.textContent = "Saved.";
    updateCoreStatus();
  } catch (e) {
    setSaveStatus("error"); el.coreWarn.textContent = `Save failed: ${e}`;
  }
}

// Recomputes the "This frame" status line in the core-players panel from the
// currently-loaded frame. Cheap, so called on every selection/box-list sync
// rather than only while the panel is open.
function updateCoreStatus() {
  if (!el.coreStatus || !state.frame) return;
  if (!state.coreIds.length) {
    el.coreStatus.textContent = "No core ids configured.";
    el.coreStatus.classList.remove("warn-text");
    return;
  }
  const present = new Set(state.frame.players.map((p) => p.track_id));
  const missing = state.coreIds.filter((id) => !present.has(id));
  const coreSet = new Set(state.coreIds);
  const extra = state.frame.players.map((p) => p.track_id).filter((id) => !coreSet.has(id));
  const parts = [];
  if (missing.length) parts.push(`Missing: ${missing.join(", ")}`);
  if (extra.length) parts.push(`Extra: ${extra.join(", ")}`);
  el.coreStatus.textContent = parts.length
    ? parts.join(" · ")
    : "OK — all core players present, no extras.";
  el.coreStatus.classList.toggle("warn-text", parts.length > 0);
}

async function jumpToIssue(dir) {
  if (!state.coreIds.length) {
    el.coreWarn.textContent = "Configure core ids first.";
    if (!state.coreMode) toggleCoreMode();
    return;
  }
  await flushSave();
  try {
    const r = await fetch(`/api/frame_issues/nearest?from=${state.idx}&dir=${dir}`);
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      el.coreWarn.textContent = `Lookup failed: ${msg}`;
      return;
    }
    const data = await r.json();
    if (data.frame == null) {
      el.coreWarn.textContent = dir === "next"
        ? "No more issue frames after this one."
        : "No issue frames before this one.";
      return;
    }
    el.coreWarn.textContent = "";
    await loadFrame(data.frame);
    if (state.coreMode) syncCorePanel();
  } catch (e) {
    el.coreWarn.textContent = `Lookup failed: ${e}`;
  }
}

// Refresh current frame + neighbors from the server, used after any change to
// exclusion areas (which may have purged boxes across all frames).
async function reloadAround() {
  const idx = state.idx;
  const [cur, prev, next] = await Promise.all([
    fetchFrame(idx),
    idx - 1 >= 0 ? fetchFrame(idx - 1) : Promise.resolve(null),
    idx + 1 < state.meta.frame_count ? fetchFrame(idx + 1) : Promise.resolve(null),
  ]);
  state.frame = cur;
  state.prevFrame = prev;
  state.nextFrame = next;
  renderAll();
}

// ---- selection + form -----------------------------------------------------
function setSelection(sel) {
  state.selected = sel;
  syncForm();
  syncBoxList();
}

function syncBoxList() {
  el.boxList.innerHTML = "";
  const filter = state.boxFilter;
  const addItem = (label, color, sel) => {
    if (filter && !label.toLowerCase().includes(filter)) return;
    const li = document.createElement("li");
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = color;
    li.appendChild(sw);
    li.appendChild(document.createTextNode(label));
    const isSel = state.selected &&
      state.selected.type === sel.type &&
      (sel.type === "ball" || state.selected.index === sel.index);
    if (isSel) li.classList.add("selected");
    li.onclick = () => { setSelection(sel); renderAll(); };
    el.boxList.appendChild(li);
  };
  state.frame.players.forEach((p, i) => {
    const gk = p.role === "GoalKeeper" ? " · GK" : "";
    addItem(`#${p.track_id} · ${p.team}${gk}`, COLORS[p.team] || COLORS.Defence, { type: "player", index: i });
  });
  if (state.frame.ball) addItem("Ball", COLORS.ball, { type: "ball" });
  // Exclusion areas listed after boxes; index is into state.exclusionAreas.
  state.exclusionAreas.forEach((a, i) => {
    addItem("Exclusion area", COLORS.exclusion, { type: "area", index: i });
  });
  updateCoreStatus();
}

function syncForm() {
  const sel = state.selected;
  el.areaFields.classList.add("hidden");
  if (!sel) {
    el.metaForm.classList.add("hidden");
    el.noSelection.classList.remove("hidden");
    return;
  }
  if (sel.type === "area") {
    // Areas have no per-instance attributes beyond geometry; the meta form is
    // irrelevant. Show the dedicated area panel instead.
    el.metaForm.classList.add("hidden");
    el.noSelection.classList.add("hidden");
    el.areaFields.classList.remove("hidden");
    return;
  }
  el.metaForm.classList.remove("hidden");
  el.noSelection.classList.add("hidden");

  // datalist of ids present in this frame (for kick target)
  el.frameIds.innerHTML = "";
  state.frame.players.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.track_id;
    el.frameIds.appendChild(o);
  });

  const isBall = sel.type === "ball";
  el.metaForm.boxType.value = isBall ? "ball" : "player";
  // "Ball" type only allowed if there is no other ball (one ball per frame).
  el.ballTypeRadio.disabled = !isBall && !!state.frame.ball;

  el.playerFields.classList.toggle("hidden", isBall);
  el.ballFields.classList.toggle("hidden", !isBall);

  if (isBall) {
    const b = state.frame.ball;
    el.ballState.value = b.state;
    let kick = "none", pid = "";
    if (b.kick && b.kick.byPlayerId != null) { kick = "by"; pid = b.kick.byPlayerId; }
    else if (b.kick && b.kick.toPlayerId != null) { kick = "to"; pid = b.kick.toPlayerId; }
    el.metaForm.kick.value = kick;
    el.kickPlayerId.value = pid;
    el.kickPlayerId.disabled = (kick === "none");
  } else {
    const p = state.frame.players[sel.index];
    state.selectedTrackId = p.track_id;   // committed id, for rename propagation
    el.trackIdInput.value = p.track_id;
    el.metaForm.team.value = p.team;
    el.gkCheck.checked = (p.role === "GoalKeeper");
    checkDuplicateId();
  }
}

function checkDuplicateId() {
  if (!state.selected || state.selected.type !== "player") return;
  const idx = state.selected.index;
  const id = el.trackIdInput.value.trim();
  const dup = state.frame.players.some((p, i) => i !== idx && p.track_id === id);
  el.dupWarn.textContent = dup ? `Warning: track id "${id}" is used by another box in this frame.` : "";
  el.dupWarn.classList.toggle("hidden", !dup);
}

// ---- form change handlers -------------------------------------------------
function onBoxTypeChange() {
  const sel = state.selected;
  if (!sel) return;
  const wantBall = el.metaForm.boxType.value === "ball";

  if (wantBall && sel.type === "player") {
    if (state.frame.ball) { syncForm(); return; }   // guard: already a ball
    const p = state.frame.players[sel.index];
    state.frame.players.splice(sel.index, 1);
    state.frame.ball = { bbox: p.bbox, state: "Active", kick: null };
    setSelection({ type: "ball" });
  } else if (!wantBall && sel.type === "ball") {
    const b = state.frame.ball;
    state.frame.ball = null;
    const newId = String(maxIdInFrame() + 1);
    state.frame.players.push({ track_id: newId, team: "Defence", bbox: b.bbox });
    setSelection({ type: "player", index: state.frame.players.length - 1 });
  }
  renderAll();
  markDirty();
}

function maxIdInFrame() {
  let mx = 0;
  for (const p of state.frame.players) {
    const n = parseInt(p.track_id, 10);
    if (!Number.isNaN(n)) mx = Math.max(mx, n);
  }
  return mx;
}

// Live text typing in the form: updates the CURRENT frame only (no propagation).
function onFormInputLive(e) {
  const sel = state.selected;
  if (!sel) return;
  const t = e.target;
  if (t && t.name === "boxType") return;            // handled on 'change'
  if (sel.type === "player") {
    if (t === el.trackIdInput) checkDuplicateId();  // rename commits on 'change'
  } else if (state.frame.ball) {
    if (t === el.kickPlayerId) applyKickFromForm(); // per-frame event
  }
}

// Discrete attribute commits. Persistent attributes (player team / role /
// track_id, ball state) are applied to the current frame AND propagated to
// every subsequent frame. Per-frame edits (ball kick event, box-type
// conversion) apply to the current frame only.
async function onFormChange(e) {
  const sel = state.selected;
  if (!sel) return;
  const t = e.target;
  if (t.name === "boxType") { onBoxTypeChange(); return; }

  if (sel.type === "player") {
    const p = state.frame.players[sel.index];
    if (t.name === "team") {
      p.team = el.metaForm.team.value;
      renderAll();
      await applyPlayerAttrForward(p.track_id, { team: p.team });
    } else if (t === el.gkCheck) {
      p.role = el.gkCheck.checked ? "GoalKeeper" : "Outfield";
      renderAll();
      await applyPlayerAttrForward(p.track_id, { role: p.role });
    } else if (t === el.trackIdInput) {
      const oldId = state.selectedTrackId;
      const newId = el.trackIdInput.value.trim();
      if (!newId || newId === oldId) { el.trackIdInput.value = oldId; return; }
      p.track_id = newId;
      state.selectedTrackId = newId;
      renderAll();
      // match the OLD id in later frames and rename it to the new id
      await applyPlayerAttrForward(oldId, { track_id: newId });
    }
  } else if (state.frame.ball) {
    const b = state.frame.ball;
    if (t === el.ballState) {
      b.state = el.ballState.value;
      renderAll();
      await applyBallAttrForward({ state: b.state });
    } else if (t.name === "kick") {
      applyKickFromForm();
    }
  }
}

// Ball kick is an instantaneous event -> current frame only, not propagated.
function applyKickFromForm() {
  const b = state.frame.ball;
  if (!b) return;
  const kick = el.metaForm.kick.value;
  el.kickPlayerId.disabled = (kick === "none");
  if (kick === "none") b.kick = null;
  else if (kick === "by") b.kick = { byPlayerId: el.kickPlayerId.value.trim() };
  else b.kick = { toPlayerId: el.kickPlayerId.value.trim() };
  renderAll();
  markDirty();
}

// The current frame's box was already updated locally; persist it and push the
// same attribute(s) to all subsequent frames.
async function applyPlayerAttrForward(matchId, attrs) {
  markDirty();
  if (state.idx + 1 < state.meta.frame_count) {
    await flushSave();
    await postJSON("/api/propagate", {
      target: "player", track_id: matchId, attrs, from_frame: state.idx + 1,
    });
    state.nextFrame = await fetchFrame(state.idx + 1);
    renderAll();
  }
}

async function applyBallAttrForward(attrs) {
  markDirty();
  if (state.idx + 1 < state.meta.frame_count) {
    await flushSave();
    await postJSON("/api/propagate", {
      target: "ball", attrs, from_frame: state.idx + 1,
    });
    state.nextFrame = await fetchFrame(state.idx + 1);
    renderAll();
  }
}

async function postJSON(url, body) {
  setSaveStatus("saving");
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Update failed: ${msg}`;
      return false;
    }
    setSaveStatus("saved");
    el.saveStatus.title = "";
    return true;
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Update failed: ${e}`;
    return false;
  }
}

async function deleteSelected() {
  const sel = state.selected;
  if (!sel) return;

  if (sel.type === "area") {
    return deleteAreaSelected();
  }

  if (sel.type === "ball") {
    state.frame.ball = null;
    setSelection(null);
    renderAll();
    markDirty();
    return;
  }

  // Deleting a player box removes this box from the current frame AND purges
  // that player (by track_id) from every subsequent frame.
  const tid = state.frame.players[sel.index].track_id;
  state.frame.players.splice(sel.index, 1);
  setSelection(null);
  renderAll();
  markDirty();

  if (state.idx + 1 < state.meta.frame_count) {
    await flushSave();                         // persist this frame first
    await deletePlayerFrom(tid, state.idx + 1);  // purge from later frames
    // the next-frame preview may have changed; refresh it
    state.nextFrame = await fetchFrame(state.idx + 1);
    renderAll();
  }
}

async function deletePlayerFrom(tid, fromFrame) {
  setSaveStatus("saving");
  try {
    const r = await fetch("/api/delete_player", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: tid, from_frame: fromFrame }),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Delete failed: ${msg}`;
      return;
    }
    setSaveStatus("saved");
    el.saveStatus.title = "";
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Delete failed: ${e}`;
  }
}

async function invertTeams() {
  el.invertBtn.disabled = true;
  setSaveStatus("saving");
  try {
    await flushSave();                       // persist current frame before the global swap
    const r = await fetch("/api/invert_teams", { method: "POST" });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Invert failed: ${msg}`;
      return;
    }
    setSaveStatus("saved");
    el.saveStatus.title = "";
    await loadFrame(state.idx);              // reload current + neighbors with swapped teams
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Invert failed: ${e}`;
  } finally {
    el.invertBtn.disabled = false;
  }
}

// ---- persistence ----------------------------------------------------------
function setSaveStatus(s) {
  el.saveStatus.textContent = s;
  el.saveStatus.className = s;
}

function markDirty() {
  state.dirty = true;
  setSaveStatus("saving");
  if (state.saveTimer) clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => { flushSave(); }, SAVE_DEBOUNCE_MS);
}

function currentPayload() {
  const ball = state.frame.ball
    ? { bbox: state.frame.ball.bbox.map(Number), state: state.frame.ball.state, kick: state.frame.ball.kick || null }
    : null;
  // Drop kick objects with an empty id (incomplete edits) to keep server-valid.
  if (ball && ball.kick) {
    const v = ball.kick.byPlayerId ?? ball.kick.toPlayerId;
    if (!v || !String(v).trim()) ball.kick = null;
  }
  return {
    players: state.frame.players.map((p) => ({
      track_id: p.track_id, team: p.team, role: p.role || "Outfield", bbox: p.bbox.map(Number),
    })),
    ball,
  };
}

async function flushSave() {
  if (!state.dirty || state.saving) return;
  if (state.saveTimer) { clearTimeout(state.saveTimer); state.saveTimer = null; }
  state.saving = true;
  const idx = state.idx;
  try {
    const r = await fetch(`/api/frame/${idx}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentPayload()),
    });
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).error || r.status;
      setSaveStatus("error");
      el.saveStatus.title = `Save failed: ${msg}`;
      state.saving = false;
      return;   // keep dirty
    }
    state.dirty = false;
    setSaveStatus("saved");
    el.saveStatus.title = "";
  } catch (e) {
    setSaveStatus("error");
    el.saveStatus.title = `Save failed: ${e}`;
  } finally {
    state.saving = false;
  }
}

// ---- navigation -----------------------------------------------------------
async function loadFrame(idx) {
  idx = Math.max(0, Math.min(state.meta.frame_count - 1, idx));
  await flushSave();  // ensure current edits are persisted before leaving
  state.idx = idx;
  state.selected = null;
  state.mode = "idle";
  state.drag = null;

  const [cur, prev, next] = await Promise.all([
    fetchFrame(idx),
    idx - 1 >= 0 ? fetchFrame(idx - 1) : Promise.resolve(null),
    idx + 1 < state.meta.frame_count ? fetchFrame(idx + 1) : Promise.resolve(null),
  ]);
  state.frame = cur;
  state.prevFrame = prev;
  state.nextFrame = next;

  el.frameLabel.textContent = `Frame ${idx} / ${state.meta.frame_count - 1}`;
  el.jumpInput.value = idx;
  preloadNeighbors();
  setSelection(null);
  renderAll();
}

async function fetchFrame(idx) {
  const r = await fetch(`/api/frame/${idx}`);
  return await r.json();
}

// ---- keyboard -------------------------------------------------------------
function isEditingField() {
  const a = document.activeElement;
  return a && (a.tagName === "INPUT" || a.tagName === "SELECT" || a.tagName === "TEXTAREA");
}

function onKeyDown(ev) {
  if (isEditingField()) return;
  if (ev.key === "ArrowLeft" && ev.shiftKey) { ev.preventDefault(); jumpToIssue("prev"); }
  else if (ev.key === "ArrowRight" && ev.shiftKey) { ev.preventDefault(); jumpToIssue("next"); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); loadFrame(state.idx - 1); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); loadFrame(state.idx + 1); }
  else if (ev.key === "Delete" || ev.key === "Backspace") { ev.preventDefault(); deleteSelected(); }
}

// ---- canvas sizing --------------------------------------------------------
function sizeCanvases() {
  const meta = state.meta;

  // Current frame: fit as large as possible within its wrapper, preserving
  // aspect ratio (limited by whichever of width/height runs out first).
  const wrap = el.canvasWrap;
  const availW = wrap.clientWidth || Math.round(window.innerWidth * 0.6);
  const availH = wrap.clientHeight || Math.round(window.innerHeight * 0.6);
  const scale = Math.min(availW / meta.width, availH / meta.height);
  el.mainCanvas.width = Math.max(1, Math.floor(meta.width * scale));
  el.mainCanvas.height = Math.max(1, Math.floor(meta.height * scale));
  state.scale = el.mainCanvas.width / meta.width;

  // Neighbor frames: small, sized by a fixed drawing height.
  const nscale = NEIGHBOR_H / meta.height;
  const nw = Math.round(meta.width * nscale);
  for (const c of [el.prevCanvas, el.nextCanvas]) { c.width = nw; c.height = NEIGHBOR_H; }
}

// ---- init -----------------------------------------------------------------
async function init() {
  const ids = ["prevBtn", "nextBtn", "frameLabel", "jumpInput", "videoName", "saveStatus",
    "prevCanvas", "nextCanvas", "mainCanvas", "canvasWrap", "boxList", "boxSearchInput", "metaForm", "noSelection",
    "playerFields", "ballFields", "trackIdInput", "ballState", "kickPlayerId",
    "frameIds", "dupWarn", "ballTypeRadio", "deleteBtn", "invertBtn", "gkCheck",
    "areaBtn", "areaFields", "deleteAreaBtn",
    "calibBtn", "calibSection", "calibLen", "calibWid", "calibList", "calibHint",
    "calibPreset", "calibSaveBtn", "calibWarn",
    "arcBtn", "arcSection", "arcList", "arcStart", "arcEnd", "arcHeight",
    "arcAddBtn", "arcWarn",
    "coreBtn", "coreSection", "coreIdsInput", "coreSaveBtn", "coreWarn",
    "coreStatus", "corePrevBtn", "coreNextBtn"];
  for (const id of ids) el[id] = document.getElementById(id);

  state.meta = await (await fetch("/api/meta")).json();
  state.exclusionAreas = (state.meta.exclusion_areas || []).map((a) => ({
    id: a.id, bbox: a.bbox.map(Number),
  }));
  await loadCalibration();
  loadBallTrajectories();
  loadCoreIds();
  el.videoName.textContent = state.meta.video;
  sizeCanvases();

  el.mainCanvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  window.addEventListener("keydown", onKeyDown);

  el.prevBtn.onclick = () => loadFrame(state.idx - 1);
  el.nextBtn.onclick = () => loadFrame(state.idx + 1);
  el.jumpInput.onchange = () => loadFrame(parseInt(el.jumpInput.value, 10) || 0);
  el.boxSearchInput.oninput = () => {
    state.boxFilter = el.boxSearchInput.value.trim().toLowerCase();
    syncBoxList();
  };
  el.invertBtn.onclick = invertTeams;
  el.areaBtn.onclick = toggleAreaMode;
  el.deleteAreaBtn.onclick = deleteAreaSelected;
  el.calibBtn.onclick = toggleCalibMode;
  el.calibSaveBtn.onclick = saveCalibration;
  el.calibPreset.onchange = applyCalibPreset;
  el.arcBtn.onclick = toggleArcMode;
  el.arcAddBtn.onclick = addTrajectory;
  el.coreBtn.onclick = toggleCoreMode;
  el.coreSaveBtn.onclick = saveCoreIds;
  el.corePrevBtn.onclick = () => jumpToIssue("prev");
  el.coreNextBtn.onclick = () => jumpToIssue("next");

  el.metaForm.addEventListener("input", onFormInputLive);
  el.metaForm.addEventListener("change", onFormChange);
  el.deleteBtn.onclick = deleteSelected;

  // flush a pending save if the user closes/reloads the tab. The calibration
  // panel has its own save button (it is a separate put, not part of the
  // per-frame autosave); flush it too on unload to avoid losing edits.
  window.addEventListener("beforeunload", () => {
    if (state.dirty) flushSave();
    if (state.calibDirty) saveCalibration();
  });

  // re-fit the current frame when the window resizes
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { sizeCanvases(); renderAll(); }, 100);
  });

  await loadFrame(0);
  // re-fit once the grid has settled around the loaded content, then repaint.
  sizeCanvases();
  renderAll();
}

init();

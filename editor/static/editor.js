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
  selected: null,    // { type:'player', index:int } | { type:'ball' } | null
  mode: "idle",      // idle | drawing | moving | resizing
  drag: null,        // per-mode scratch data (image coords)
  images: new Map(),  // idx -> Image
  saveTimer: null,
  dirty: false,
  saving: false,
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
    for (let i = 0; i < annotation.players.length; i++) {
      const p = annotation.players[i];
      const sel = opts.selectable && state.selected &&
                  state.selected.type === "player" && state.selected.index === i;
      drawBoxOn(ctx, scale, p.bbox, COLORS[p.team] || COLORS.Defence,
                { label: p.track_id, selected: sel });
    }
    if (annotation.ball) {
      const sel = opts.selectable && state.selected && state.selected.type === "ball";
      drawBoxOn(ctx, scale, annotation.ball.bbox, COLORS.ball,
                { label: "ball", selected: sel });
    }
    // rubber-band preview for a box being drawn
    if (opts.selectable && state.mode === "drawing" && state.drag) {
      const b = state.drag.box;
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(b[0] * scale, b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale);
      ctx.setLineDash([]);
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

function selectedBox() {
  if (!state.selected) return null;
  if (state.selected.type === "ball") return state.frame.ball;
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

  // empty space -> begin drawing a new box
  state.mode = "drawing";
  state.drag = { startX: x, startY: y, box: [x, y, x, y] };
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
    state.mode = "idle";
    if (dragPx >= MIN_DRAG) {
      await createPlayerBox(normalizeBox(b));
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
    markDirty();
    return;
  }
  state.mode = "idle";
  state.drag = null;
}

async function createPlayerBox(bbox) {
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

// ---- selection + form -----------------------------------------------------
function setSelection(sel) {
  state.selected = sel;
  syncForm();
  syncBoxList();
}

function syncBoxList() {
  el.boxList.innerHTML = "";
  const addItem = (label, color, sel) => {
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
  state.frame.players.forEach((p, i) =>
    addItem(`#${p.track_id} · ${p.team}`, COLORS[p.team] || COLORS.Defence, { type: "player", index: i }));
  if (state.frame.ball) addItem("Ball", COLORS.ball, { type: "ball" });
}

function syncForm() {
  const sel = state.selected;
  if (!sel) {
    el.metaForm.classList.add("hidden");
    el.noSelection.classList.remove("hidden");
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
    el.trackIdInput.value = p.track_id;
    el.metaForm.team.value = p.team;
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

function onFormInput() {
  const sel = state.selected;
  if (!sel) return;
  if (sel.type === "player") {
    const p = state.frame.players[sel.index];
    p.track_id = el.trackIdInput.value.trim() || p.track_id;
    p.team = el.metaForm.team.value;
    checkDuplicateId();
  } else {
    const b = state.frame.ball;
    b.state = el.ballState.value;
    const kick = el.metaForm.kick.value;
    el.kickPlayerId.disabled = (kick === "none");
    if (kick === "none") b.kick = null;
    else if (kick === "by") b.kick = { byPlayerId: el.kickPlayerId.value.trim() };
    else b.kick = { toPlayerId: el.kickPlayerId.value.trim() };
  }
  renderAll();
  markDirty();
}

async function deleteSelected() {
  const sel = state.selected;
  if (!sel) return;

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
      track_id: p.track_id, team: p.team, bbox: p.bbox.map(Number),
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
  if (ev.key === "ArrowLeft") { ev.preventDefault(); loadFrame(state.idx - 1); }
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
    "prevCanvas", "nextCanvas", "mainCanvas", "canvasWrap", "boxList", "metaForm", "noSelection",
    "playerFields", "ballFields", "trackIdInput", "ballState", "kickPlayerId",
    "frameIds", "dupWarn", "ballTypeRadio", "deleteBtn"];
  for (const id of ids) el[id] = document.getElementById(id);

  state.meta = await (await fetch("/api/meta")).json();
  el.videoName.textContent = state.meta.video;
  sizeCanvases();

  el.mainCanvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  window.addEventListener("keydown", onKeyDown);

  el.prevBtn.onclick = () => loadFrame(state.idx - 1);
  el.nextBtn.onclick = () => loadFrame(state.idx + 1);
  el.jumpInput.onchange = () => loadFrame(parseInt(el.jumpInput.value, 10) || 0);

  el.metaForm.addEventListener("input", (e) => {
    if (e.target.name === "boxType") onBoxTypeChange();
    else onFormInput();
  });
  el.metaForm.addEventListener("change", (e) => {
    if (e.target.name === "boxType") onBoxTypeChange();
  });
  el.deleteBtn.onclick = deleteSelected;

  // flush a pending save if the user closes/reloads the tab
  window.addEventListener("beforeunload", () => { if (state.dirty) flushSave(); });

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

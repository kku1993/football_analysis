/* Football label editor.
 *
 * Two phases: (1) pick which frames to process, (2) for each to-process frame
 * add homography calibration dots, then drag/label the player & ball boxes.
 * Boxes are seeded from the CV output (bbox/) and projected to pitch meters in
 * real time via each frame's homography. State autosaves after every change.
 */

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");
const $ = (id) => document.getElementById(id);

// ---- data from the server + editable state ----
let data = { fps: 12, frames: [], bbox: {}, pitch: {} };
let state = null;

let mode = "select";        // "select" | "calib" | "boxes"
let curFrame = null;        // current frame name
let img = new Image();
let imgLoaded = false;

let view = { scale: 1, ox: 0, oy: 0 };
let saveTimer = null;

// calibration
let selectedDot = -1;
let armedIdx = -1;          // pitch landmark waiting to be placed
let pitchLines = null;
let overlayTimer = null;

// boxes
let selected = null;        // {kind:"player"|"ball", id?} of selected box
let pendingAdd = null;      // "player" | "ball" | null
let isolation = false;
let boxSearch = "";         // filter term for the box list
const homoCache = {};       // frameName -> 3x3 matrix | null
let homoTimer = null;

const DOT_HIT_PX = 12;
const HANDLE_PX = 7;
// High-contrast against the green pitch (and distinct from the yellow ball).
const TEAM_COLOR = { A: "#00e0ff", B: "#ff2ec4" };
const BALL_COLOR = "#ffd94a";

// ---------- pitch landmarks (known-point picker) ----------
const PITCH_POINTS = (() => {
  const pts = [];
  const add = (name, x, y) => pts.push({ name, x, y });
  const quad = [
    ["corner", 52.5, 34], ["9.15 m corner mark on sideline", 43.35, 34],
    ["9.15 m corner mark on goal line", 52.5, 24.85],
    ["penalty box corner on goal line", 52.5, 20.16], ["penalty box corner", 36, 20.16],
    ["goal area corner on goal line", 52.5, 9.16], ["goal area corner", 47, 9.16],
    ["penalty arc meets penalty box", 36, 7.31], ["goal post", 52.5, 3.66],
  ];
  for (const [name, x, y] of quad)
    for (const sx of [-1, 1]) for (const sy of [1, -1])
      add(`${sx < 0 ? "left" : "right"} ${sy > 0 ? "upper" : "lower"} ${name}`, sx * x, sy * y);
  for (const sx of [-1, 1]) {
    const side = sx < 0 ? "left" : "right";
    add(`${side} penalty mark`, 41.5 * sx, 0);
    add(`${side} goal area front midpoint`, 47 * sx, 0);
    add(`${side} penalty arc apex`, 32.35 * sx, 0);
  }
  add("halfway line upper end", 0, 34); add("halfway line lower end", 0, -34);
  add("center circle top", 0, 9.15); add("center circle bottom", 0, -9.15);
  add("center circle left", -9.15, 0); add("center circle right", 9.15, 0);
  add("center spot", 0, 0);
  return pts;
})();

// ---------- helpers ----------
function frameIndex(name) {
  const m = /(\d+)(?=\.[^.]+$)/.exec(name);
  return m ? parseInt(m[1], 10) : 0;
}
function labelFrames() {
  return [...(state.toProcess || [])].sort((a, b) => frameIndex(a) - frameIndex(b));
}
function currentList() { return mode === "select" ? data.frames : labelFrames(); }

function frameState(name) {
  if (!state.frames[name]) state.frames[name] = { players: [], ball: null };
  return state.frames[name];
}

/* Seed a frame's editable boxes from the CV output the first time it's used. */
function seedFrame(name) {
  if (state.frames[name] && state.frames[name]._seeded) return;
  const bb = data.bbox[name] || { players: [], ball: null };
  const fs = frameState(name);
  if (!fs.players.length) {
    fs.players = (bb.players || []).map((p) => {
      registerPlayer(p.id, p.team);
      return { id: String(p.id), x_min: p.x_min, y_min: p.y_min,
               x_max: p.x_max, y_max: p.y_max, z: 0, anchor: null };
    });
  }
  if (!fs.ball && bb.ball) {
    fs.ball = { ...bb.ball, z: 0, anchor: null, state: "Active", kick: null };
  }
  fs._seeded = true;
}

function registerPlayer(id, team) {
  id = String(id);
  if (!state.players[id]) state.players[id] = { team: team || "A", role: "Outfield" };
  const n = frameIndex("x" + id) || parseInt(id, 10);
  if (!isNaN(n) && n >= state.nextId) state.nextId = n + 1;
}
function playerMeta(id) {
  if (!state.players[id]) state.players[id] = { team: "A", role: "Outfield" };
  return state.players[id];
}

function anchorPx(box, kind) {
  const cx = (box.x_min + box.x_max) / 2;
  const a = box.anchor;
  if (a === "center" || (a == null && kind === "ball"))
    return [cx, (box.y_min + box.y_max) / 2];
  return [cx, box.y_max];
}
function applyH(H, px, py) {
  const X = H[0][0] * px + H[0][1] * py + H[0][2];
  const Y = H[1][0] * px + H[1][1] * py + H[1][2];
  const W = H[2][0] * px + H[2][1] * py + H[2][2];
  return [X / W, Y / W];
}

// ---------- init ----------
async function init() {
  data = await fetch("/api/bootstrap").then((r) => r.json());
  state = data.state;
  // migrate / default missing keys
  state.toProcess ||= [];
  state.teamMapping ||= { A: "Offence", B: "Defence" };
  state.players ||= {};
  state.calibration ||= {};
  state.frames ||= {};
  state.correctSets ||= [];
  state.nextId ||= 1;

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  curFrame = data.frames[0];
  setMode("select");
  loadFrame(curFrame);
  wireControls();
}

function loadFrame(name) {
  if (!name) return;
  curFrame = name;
  selectedDot = -1; armedIdx = -1; selected = null; pendingAdd = null;
  imgLoaded = false; pitchLines = null;
  if (mode !== "select") { seedFrame(name); ensureHomography(name); }
  img = new Image();
  img.onload = () => { imgLoaded = true; fitView(); draw(); };
  img.src = "/frames/" + name;
  renderSidebar();
}

function navigate(delta) {
  const list = currentList();
  let i = list.indexOf(curFrame);
  if (i < 0) i = 0;
  const ni = Math.max(0, Math.min(list.length - 1, i + delta));
  loadFrame(list[ni]);
}

// ---------- view ----------
function resizeCanvas() {
  const wrap = $("canvas-wrap");
  canvas.width = wrap.clientWidth; canvas.height = wrap.clientHeight;
  if (imgLoaded) { fitView(); draw(); }
}
function fitView() {
  const s = Math.min(canvas.width / img.width, canvas.height / img.height);
  view.scale = s;
  view.ox = (canvas.width - img.width * s) / 2;
  view.oy = (canvas.height - img.height * s) / 2;
}
function toImage(cx, cy) { return { x: (cx - view.ox) / view.scale, y: (cy - view.oy) / view.scale }; }
function toCanvas(ix, iy) { return { x: ix * view.scale + view.ox, y: iy * view.scale + view.oy }; }

// ---------- drawing ----------
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!imgLoaded) return;
  ctx.save();
  ctx.translate(view.ox, view.oy); ctx.scale(view.scale, view.scale);
  ctx.drawImage(img, 0, 0);
  ctx.restore();

  if (mode === "calib") { drawPitchOverlay(); drawDots(); }
  else if (mode === "boxes") drawBoxes();
}

function drawPitchOverlay() {
  if (!pitchLines || !$("overlay-toggle").checked) return;
  ctx.strokeStyle = "rgba(255,60,220,.85)"; ctx.lineWidth = 2;
  for (const line of pitchLines) {
    ctx.beginPath();
    line.forEach(([u, v], i) => { const p = toCanvas(u, v); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); });
    ctx.stroke();
  }
}
function drawDots() {
  const list = state.calibration[curFrame] || [];
  ctx.font = "bold 13px sans-serif";
  list.forEach((d, i) => {
    const p = toCanvas(d.px, d.py);
    ctx.strokeStyle = i === selectedDot ? "#fff176" : "#e8b93c"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.x - 9, p.y); ctx.lineTo(p.x + 9, p.y);
    ctx.moveTo(p.x, p.y - 9); ctx.lineTo(p.x, p.y + 9); ctx.stroke();
    ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${i + 1} (${d.x}, ${d.y})`, p.x + 10, p.y - 8);
  });
}

function eachBox(cb) {
  const fs = frameState(curFrame);
  fs.players.forEach((b) => cb({ kind: "player", id: b.id }, b));
  if (fs.ball) cb({ kind: "ball" }, fs.ball);
}
function isSelected(ref) {
  return selected && selected.kind === ref.kind &&
    (ref.kind === "ball" || selected.id === ref.id);
}
function drawBoxes() {
  ctx.font = "12px sans-serif";
  eachBox((ref, b) => {
    if (isolation && selected && !isSelected(ref)) return;
    const p1 = toCanvas(b.x_min, b.y_min), p2 = toCanvas(b.x_max, b.y_max);
    const sel = isSelected(ref);
    const color = ref.kind === "ball" ? BALL_COLOR
      : (TEAM_COLOR[playerMeta(ref.id).team] || "#aaa");
    ctx.strokeStyle = color; ctx.lineWidth = sel ? 3 : 2;
    ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    // label with a dark outline so it stays legible on any background
    const lbl = ref.kind === "ball" ? "ball"
      : `${ref.id} ${state.teamMapping[playerMeta(ref.id).team][0]}`;
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.75)";
    ctx.strokeText(lbl, p1.x, p1.y - 3);
    ctx.fillStyle = color;
    ctx.fillText(lbl, p1.x, p1.y - 3);
    // anchor marker
    const [ax, ay] = anchorPx(b, ref.kind); const ap = toCanvas(ax, ay);
    ctx.beginPath(); ctx.arc(ap.x, ap.y, 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    if (sel) {
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of boxCorners(b))
        { const c = toCanvas(hx, hy); ctx.fillRect(c.x - HANDLE_PX / 2, c.y - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX); }
    }
  });
}
function boxCorners(b) {
  return [[b.x_min, b.y_min], [b.x_max, b.y_min], [b.x_min, b.y_max], [b.x_max, b.y_max]];
}

// ---------- homography (live pixel -> pitch) ----------
function ensureHomography(name) {
  const dots = state.calibration[name] || [];
  if (dots.length < 4) { homoCache[name] = null; return; }
  clearTimeout(homoTimer);
  homoTimer = setTimeout(async () => {
    const r = await fetch("/api/homography", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dots }),
    }).then((r) => r.json());
    homoCache[name] = r.homography || null;
    if (mode === "boxes") { renderSelectedPanel(); updateHomoWarn(); }
  }, 120);
}
function boxXY(box, kind) {
  const H = homoCache[curFrame];
  if (!H) return null;
  const [px, py] = anchorPx(box, kind);
  return applyH(H, px, py);
}

// ---------- interactions ----------
let mouse = { down: false, moved: false, sx: 0, sy: 0, mode: null, grab: null };

canvas.addEventListener("mousedown", (e) => {
  const r = canvas.getBoundingClientRect();
  mouse.down = true; mouse.moved = false;
  mouse.sx = e.clientX - r.left; mouse.sy = e.clientY - r.top;
  mouse.mode = null; mouse.grab = null;
  if (mode === "calib") { mouse.grab = hitDot(mouse.sx, mouse.sy); }
  else if (mode === "boxes") {
    const h = hitBox(mouse.sx, mouse.sy);
    if (h) { selected = h.ref; mouse.mode = h.corner ? "resize" : "move"; mouse.grab = { box: h.box, corner: h.corner }; renderSidebar(); draw(); }
  }
});

canvas.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const ip = toImage(cx, cy);
  hud.textContent = `pixel (${Math.round(ip.x)}, ${Math.round(ip.y)})`;
  if (!mouse.down) {
    if (mode === "calib") canvas.style.cursor = hitDot(cx, cy) >= 0 ? "grab" : "crosshair";
    else if (mode === "boxes") canvas.style.cursor = pendingAdd ? "copy" : (hitBox(cx, cy) ? "move" : "crosshair");
    return;
  }
  if (Math.abs(cx - mouse.sx) + Math.abs(cy - mouse.sy) > 3) mouse.moved = true;

  if (mode === "calib" && mouse.grab >= 0) {
    const d = (state.calibration[curFrame] || [])[mouse.grab];
    d.px = Math.round(ip.x * 10) / 10; d.py = Math.round(ip.y * 10) / 10;
    selectedDot = mouse.grab; draw(); scheduleSave(); refreshOverlay();
  } else if (mode === "boxes" && mouse.mode === "move") {
    const b = mouse.grab.box;
    const dx = (cx - mouse.sx) / view.scale, dy = (cy - mouse.sy) / view.scale;
    b.x_min += dx; b.x_max += dx; b.y_min += dy; b.y_max += dy;
    mouse.sx = cx; mouse.sy = cy;
    draw(); renderSelectedPanel(); scheduleSave();
  } else if (mode === "boxes" && mouse.mode === "resize") {
    const b = mouse.grab.box, c = mouse.grab.corner;
    if (c.includes("min-x")) b.x_min = ip.x; if (c.includes("max-x")) b.x_max = ip.x;
    if (c.includes("min-y")) b.y_min = ip.y; if (c.includes("max-y")) b.y_max = ip.y;
    draw(); renderSelectedPanel(); scheduleSave();
  } else if (mouse.moved && mouse.mode == null) {
    // pan the frame (calib: not dragging a dot; boxes: not on a box). The
    // pitch overlay / dots redraw in the new view automatically.
    view.ox += cx - mouse.sx; view.oy += cy - mouse.sy;
    mouse.sx = cx; mouse.sy = cy; draw();
  }
});

window.addEventListener("mouseup", () => {
  if (!mouse.down) return;
  const wasDrag = mouse.moved;
  mouse.down = false; canvas.style.cursor = "crosshair";
  if (mode === "boxes" && mouse.mode === "resize") normalizeBox(mouse.grab.box);
  if (wasDrag) { mouse.mode = null; mouse.grab = null; return; }

  // plain click
  const ip = toImage(mouse.sx, mouse.sy);
  const inImg = ip.x >= 0 && ip.y >= 0 && ip.x <= img.width && ip.y <= img.height;
  if (mode === "calib") {
    if (mouse.grab >= 0) { selectDot(mouse.grab); }
    else if (inImg) {
      const known = PITCH_POINTS[armedIdx]; armedIdx = -1;
      (state.calibration[curFrame] ||= []).push({
        px: Math.round(ip.x * 10) / 10, py: Math.round(ip.y * 10) / 10,
        x: known ? known.x : 0, y: known ? known.y : 0,
      });
      selectDot((state.calibration[curFrame].length - 1), !known);
      scheduleSave(); refreshOverlay(); ensureHomography(curFrame);
    }
  } else if (mode === "boxes") {
    if (pendingAdd && inImg) { addBoxAt(pendingAdd, ip.x, ip.y); pendingAdd = null; }
    else { const h = hitBox(mouse.sx, mouse.sy); selected = h ? h.ref : null; renderSidebar(); draw(); }
  }
  mouse.mode = null; mouse.grab = null;
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const ns = Math.min(Math.max(view.scale * factor, 0.05), 20);
  view.ox = cx - (cx - view.ox) * (ns / view.scale);
  view.oy = cy - (cy - view.oy) * (ns / view.scale);
  view.scale = ns; draw();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") navigate(-1);
  else if (e.key === "ArrowRight") navigate(1);
  else if (e.key === " " && mode === "select") { e.preventDefault(); toggleToProcess(); }
  else if (e.key === "Escape") { armedIdx = -1; pendingAdd = null; renderSidebar(); draw(); }
  else if (e.key === "Delete" || e.key === "Backspace") {
    if (mode === "calib" && selectedDot >= 0) {
      state.calibration[curFrame].splice(selectedDot, 1); selectedDot = -1;
      draw(); renderSidebar(); scheduleSave(); ensureHomography(curFrame);
    } else if (mode === "boxes" && selected) deleteSelected();
  }
});

// ---------- calibration bits ----------
function hitDot(cx, cy) {
  const list = state.calibration[curFrame] || [];
  for (let i = list.length - 1; i >= 0; i--) {
    const p = toCanvas(list[i].px, list[i].py);
    if (Math.hypot(p.x - cx, p.y - cy) <= DOT_HIT_PX) return i;
  }
  return -1;
}
function selectDot(i, focusInput) {
  selectedDot = i; draw(); renderSidebar();
  if (focusInput) {
    const row = document.querySelectorAll(".dot-row")[i];
    if (row) { const inp = row.querySelector("input"); inp.focus(); inp.select(); }
  }
}
function refreshOverlay() {
  clearTimeout(overlayTimer);
  if (mode !== "calib" || !$("overlay-toggle").checked) return;
  overlayTimer = setTimeout(async () => {
    const resp = await fetch("/api/pitchlines", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dots: state.calibration[curFrame] || [] }),
    });
    pitchLines = resp.ok ? (await resp.json()).lines : null; draw();
  }, 250);
}

// ---------- box editing ----------
function hitBox(cx, cy) {
  // corners of selected box first
  const fs = frameState(curFrame);
  const all = [];
  fs.players.forEach((b) => all.push({ ref: { kind: "player", id: b.id }, box: b }));
  if (fs.ball) all.push({ ref: { kind: "ball" }, box: fs.ball });
  if (selected) {
    const s = all.find((o) => isSelected(o.ref));
    if (s) {
      const corners = { "min-x min-y": [s.box.x_min, s.box.y_min], "max-x min-y": [s.box.x_max, s.box.y_min],
                        "min-x max-y": [s.box.x_min, s.box.y_max], "max-x max-y": [s.box.x_max, s.box.y_max] };
      for (const [name, [ix, iy]] of Object.entries(corners)) {
        const p = toCanvas(ix, iy);
        if (Math.abs(p.x - cx) <= HANDLE_PX && Math.abs(p.y - cy) <= HANDLE_PX)
          return { ...s, corner: name };
      }
    }
  }
  for (let i = all.length - 1; i >= 0; i--) {
    if (isolation && selected && !isSelected(all[i].ref)) continue;
    const b = all[i].box;
    const p1 = toCanvas(b.x_min, b.y_min), p2 = toCanvas(b.x_max, b.y_max);
    if (cx >= Math.min(p1.x, p2.x) && cx <= Math.max(p1.x, p2.x) &&
        cy >= Math.min(p1.y, p2.y) && cy <= Math.max(p1.y, p2.y))
      return all[i];
  }
  return null;
}
function normalizeBox(b) {
  if (b.x_min > b.x_max) [b.x_min, b.x_max] = [b.x_max, b.x_min];
  if (b.y_min > b.y_max) [b.y_min, b.y_max] = [b.y_max, b.y_min];
}
function addBoxAt(kind, x, y) {
  const fs = frameState(curFrame);
  if (kind === "ball") {
    fs.ball = { x_min: x - 7, y_min: y - 7, x_max: x + 7, y_max: y + 7,
                z: 0, anchor: null, state: "Active", kick: null };
    selected = { kind: "ball" };
  } else {
    const id = String(state.nextId++);
    state.players[id] = { team: "A", role: "Outfield" };
    fs.players.push({ id, x_min: x - 18, y_min: y - 45, x_max: x + 18, y_max: y + 45,
                      z: 0, anchor: null });
    selected = { kind: "player", id };
  }
  draw(); renderSidebar(); scheduleSave();
}
function deleteSelected() {
  if (!selected) return;
  if (selected.kind === "ball") { frameState(curFrame).ball = null; }
  else {
    // remove this id from the current frame and all subsequent frames
    const fromIdx = frameIndex(curFrame);
    for (const name of Object.keys(state.frames)) {
      if (frameIndex(name) >= fromIdx)
        state.frames[name].players = (state.frames[name].players || []).filter((p) => p.id !== selected.id);
    }
    // also drop from not-yet-seeded to-process frames by seeding then filtering
    for (const name of labelFrames()) {
      if (frameIndex(name) >= fromIdx) { seedFrame(name);
        state.frames[name].players = state.frames[name].players.filter((p) => p.id !== selected.id); }
    }
  }
  selected = null; draw(); renderSidebar(); scheduleSave();
}

// ---------- correct rosters / flagging ----------
function applicableSet(name) {
  const idx = frameIndex(name);
  let best = null, bestStart = -Infinity;
  for (const s of state.correctSets) {
    const si = frameIndex(s.start), ei = s.end ? frameIndex(s.end) : Infinity;
    if (idx >= si && idx <= ei && si > bestStart) { best = s; bestStart = si; }
  }
  return best;
}
function frameFlags(name) {
  const set = applicableSet(name);
  if (!set) return null;
  seedFrame(name);
  const present = new Set((state.frames[name].players || []).map((p) => p.id));
  const expected = new Set([...(set.offence || []), ...(set.defence || [])].map(String));
  const missing = [...expected].filter((id) => !present.has(id));
  const extra = [...present].filter((id) => !expected.has(id));
  return (missing.length || extra.length) ? { missing, extra } : null;
}

// ---------- to-process selection ----------
function toggleToProcess() {
  const i = state.toProcess.indexOf(curFrame);
  if (i >= 0) state.toProcess.splice(i, 1); else state.toProcess.push(curFrame);
  renderSidebar(); scheduleSave();
}

// ---------- mode switching ----------
function setMode(m) {
  if (m !== "select" && !state.toProcess.length) {
    $("export-status").textContent = "Select at least one frame to process first.";
    return;
  }
  mode = m;
  $("select-panel").hidden = m !== "select";
  $("label-panel").hidden = m === "select";
  if (m !== "select") {
    $("calib-sub").hidden = m !== "calib";
    $("boxes-sub").hidden = m !== "boxes";
    $("mode-calib").classList.toggle("active", m === "calib");
    $("mode-boxes").classList.toggle("active", m === "boxes");
    // ensure current frame is a to-process frame
    if (!state.toProcess.includes(curFrame)) curFrame = labelFrames()[0];
    seedFrame(curFrame); ensureHomography(curFrame);
  }
  $("mode-badge").textContent =
    m === "select" ? "SELECT FRAMES" : (m === "calib" ? "CALIBRATE" : "LABEL BOXES");
  loadFrame(curFrame);
}

// ---------- sidebar rendering ----------
function renderSidebar() {
  const list = currentList();
  const pos = list.indexOf(curFrame);
  $("frame-label").textContent = `${curFrame || ""}  (${pos + 1}/${list.length})`;
  $("prev").disabled = pos <= 0;
  $("next").disabled = pos >= list.length - 1;
  $("status-line").textContent =
    mode === "select" ? `${state.toProcess.length} frame(s) marked to process`
                      : `${labelFrames().length} to-process frame(s)`;

  if (mode === "select") renderSelectPanel();
  else if (mode === "calib") renderCalibPanel();
  else renderBoxesPanel();
}

function renderSelectPanel() {
  $("toprocess-toggle").checked = state.toProcess.includes(curFrame);
  $("toprocess-summary").textContent =
    state.toProcess.length ? "Selected: " + labelFrames().map(frameIndex).join(", ") : "None selected yet.";
  const strip = $("toprocess-strip"); strip.innerHTML = "";
  data.frames.forEach((n) => {
    const c = document.createElement("span");
    c.className = "chip" + (state.toProcess.includes(n) ? " on" : "") + (n === curFrame ? " cur" : "");
    c.textContent = frameIndex(n);
    c.onclick = () => loadFrame(n);
    strip.appendChild(c);
  });
}

function renderCalibPanel() {
  renderPitchMap();
  const dots = state.calibration[curFrame] || [];
  const done = labelFrames().filter((n) => (state.calibration[n] || []).length >= 4).length;
  $("calib-status").textContent =
    `This frame: ${dots.length} dot${dots.length === 1 ? "" : "s"}` +
    (dots.length >= 4 ? " ✓" : ` (need ${4 - dots.length} more)`) +
    ` • ${done}/${labelFrames().length} to-process frames calibrated`;
  const listEl = $("dots-list"); listEl.innerHTML = "";
  dots.forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "dot-row" + (i === selectedDot ? " selected" : "");
    row.innerHTML = `<span class="idx">${i + 1}</span>x <input type="number" step="0.01" value="${d.x}">` +
      `y <input type="number" step="0.01" value="${d.y}"><button class="del">×</button>`;
    const [xi, yi] = row.querySelectorAll("input");
    xi.onchange = () => { d.x = parseFloat(xi.value) || 0; draw(); scheduleSave(); ensureHomography(curFrame); refreshOverlay(); };
    yi.onchange = () => { d.y = parseFloat(yi.value) || 0; draw(); scheduleSave(); ensureHomography(curFrame); refreshOverlay(); };
    row.querySelector(".del").onclick = () => {
      dots.splice(i, 1); if (selectedDot === i) selectedDot = -1;
      draw(); renderSidebar(); scheduleSave(); ensureHomography(curFrame); refreshOverlay();
    };
    row.onclick = (e) => { if (e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON") selectDot(i); };
    listEl.appendChild(row);
  });
}

function renderPitchMap() {
  const svg = [];
  svg.push(`<rect x="-52.5" y="-34" width="105" height="68" class="pl"/>`);
  svg.push(`<line x1="0" y1="-34" x2="0" y2="34" class="pl"/>`);
  svg.push(`<circle cx="0" cy="0" r="9.15" class="pl"/>`);
  for (const s of [-1, 1]) {
    const gx = 52.5 * s, bx = 36 * s, ax = 47 * s;
    svg.push(`<polyline points="${gx},-20.16 ${bx},-20.16 ${bx},20.16 ${gx},20.16" class="pl"/>`);
    svg.push(`<polyline points="${gx},-9.16 ${ax},-9.16 ${ax},9.16 ${gx},9.16" class="pl"/>`);
    svg.push(`<path d="M ${bx} -7.31 A 9.15 9.15 0 0 ${s < 0 ? 1 : 0} ${bx} 7.31" class="pl"/>`);
  }
  const dotsHere = state.calibration[curFrame] || [];
  PITCH_POINTS.forEach((p, i) => {
    const placed = dotsHere.some((d) => Math.abs(d.x - p.x) < 0.01 && Math.abs(d.y - p.y) < 0.01);
    const cls = "lm" + (i === armedIdx ? " armed" : "") + (placed ? " placed" : "");
    svg.push(`<circle cx="${p.x}" cy="${-p.y}" r="1.7" class="${cls}" data-i="${i}"><title>${p.name} (${p.x}, ${p.y})</title></circle>`);
  });
  const wrap = $("pitch-map");
  wrap.innerHTML = `<svg viewBox="-56 -37.5 112 75">${svg.join("")}</svg>`;
  wrap.querySelectorAll(".lm").forEach((el) => el.onclick = () => {
    const i = +el.dataset.i; armedIdx = armedIdx === i ? -1 : i; renderPitchMap();
    const p = PITCH_POINTS[armedIdx];
    $("pick-status").textContent = p ? `placing: ${p.name} (${p.x}, ${p.y}) — click its location on the frame` : "";
  });
}

function renderBoxesPanel() {
  updateHomoWarn();
  $("teamA-toggle").textContent = state.teamMapping.A;
  $("teamB-note").textContent = `Team B = ${state.teamMapping.B}`;
  renderSelectedPanel();
  renderBoxList();
  renderCorrectSets();
}

// ---------- box list + search ----------
function boxItems() {
  const fs = frameState(curFrame);
  const items = fs.players.map((b) => ({
    ref: { kind: "player", id: b.id }, key: b.id,
    label: `${b.id} ${state.teamMapping[playerMeta(b.id).team][0]}`,
    color: TEAM_COLOR[playerMeta(b.id).team] || "#aaa",
  }));
  items.sort((a, b) => (parseInt(a.key, 10) || 0) - (parseInt(b.key, 10) || 0));
  if (fs.ball) items.push({ ref: { kind: "ball" }, key: "ball", label: "ball", color: BALL_COLOR });
  return items;
}
function selectBox(ref) { selected = ref; renderSidebar(); draw(); }
function renderBoxList() {
  const term = boxSearch.trim().toLowerCase();
  let items = boxItems();
  if (term) items = items.filter((it) => it.label.toLowerCase().includes(term));
  // narrowing to a single box auto-selects it
  if (term && items.length === 1 && !isSelected(items[0].ref)) selected = items[0].ref;

  const list = $("box-list"); list.innerHTML = "";
  if (!items.length) { list.innerHTML = '<div class="muted">no match</div>'; return; }
  items.forEach((it) => {
    const row = document.createElement("div");
    row.className = "box-row" + (isSelected(it.ref) ? " selected" : "");
    row.innerHTML = `<span class="sw" style="background:${it.color}"></span>${it.label}`;
    row.onclick = () => selectBox(it.ref);
    list.appendChild(row);
  });
  renderSelectedPanel();
}
function updateHomoWarn() {
  $("homography-warn").textContent = homoCache[curFrame]
    ? "" : "⚠ This frame needs 4+ calibration points for pitch coordinates.";
}

function renderSelectedPanel() {
  const panel = $("selected-panel");
  if (!selected) { panel.className = "muted"; panel.textContent = "No box selected. Click a box, or use + Player / + Ball."; return; }
  panel.className = "active";
  const fs = frameState(curFrame);
  if (selected.kind === "ball") {
    if (!fs.ball) { selected = null; return renderSelectedPanel(); }
    const b = fs.ball; const xy = boxXY(b, "ball");
    panel.innerHTML = `<b>Ball</b>
      <div class="field"><label>state</label>
        <select id="b-state">${["Active", "Goal", "OutOfBounds"].map((s) => `<option ${b.state === s ? "selected" : ""}>${s}</option>`).join("")}</select></div>
      <div class="field"><label>kick</label>
        <select id="b-kick">${[["none", "none"], ["by", "kicked by"], ["to", "received by"]].map(([v, t]) => `<option value="${v}" ${kickType(b) === v ? "selected" : ""}>${t}</option>`).join("")}</select>
        <input id="b-kickid" placeholder="player id" value="${kickId(b)}" ${kickType(b) === "none" ? "disabled" : ""}></div>
      <div class="field"><label>z (m)</label><input id="b-z" type="number" step="0.1" value="${b.z ?? 0}"></div>
      <div class="field"><label>anchor</label>${anchorSelect(b)}</div>
      <div class="field"><label>pitch x,y</label><span class="xy">${fmtXY(xy)}</span></div>
      <button class="del" id="b-del">Delete ball</button>`;
    $("b-state").onchange = (e) => { b.state = e.target.value; scheduleSave(); };
    $("b-kick").onchange = (e) => { setKick(b, e.target.value, $("b-kickid").value); renderSelectedPanel(); scheduleSave(); };
    $("b-kickid").onchange = (e) => { setKick(b, kickType(b), e.target.value); scheduleSave(); };
    $("b-z").onchange = (e) => { b.z = parseFloat(e.target.value) || 0; scheduleSave(); };
    wireAnchor(b, "ball");
    $("b-del").onclick = deleteSelected;
  } else {
    const b = fs.players.find((p) => p.id === selected.id);
    if (!b) { selected = null; return renderSelectedPanel(); }
    const meta = playerMeta(b.id); const xy = boxXY(b, "player");
    panel.innerHTML = `<b>Player</b>
      <div class="field"><label>id</label><input id="p-id" class="wide" value="${b.id}"></div>
      <div class="field"><label>team</label>
        <select id="p-team">${["A", "B"].map((t) => `<option ${meta.team === t ? "selected" : ""}>${t}</option>`).join("")}</select>
        <span class="muted">= ${state.teamMapping[meta.team]}</span></div>
      <div class="field"><label>role</label>
        <select id="p-role">${["Outfield", "GoalKeeper"].map((r) => `<option ${meta.role === r ? "selected" : ""}>${r}</option>`).join("")}</select></div>
      <div class="field"><label>z (m)</label><input id="p-z" type="number" step="0.1" value="${b.z ?? 0}"></div>
      <div class="field"><label>anchor</label>${anchorSelect(b)}</div>
      <div class="field"><label>pitch x,y</label><span class="xy">${fmtXY(xy)}</span></div>
      <p class="hint">team & role propagate to this player in every frame; z is per-frame.</p>
      <button class="del" id="p-del">Delete player (this + later frames)</button>`;
    $("p-id").onchange = (e) => { renamePlayer(b, e.target.value.trim()); };
    $("p-team").onchange = (e) => { meta.team = e.target.value; draw(); renderSidebar(); scheduleSave(); };
    $("p-role").onchange = (e) => { meta.role = e.target.value; scheduleSave(); };
    $("p-z").onchange = (e) => { b.z = parseFloat(e.target.value) || 0; scheduleSave(); };
    wireAnchor(b, "player");
    $("p-del").onclick = deleteSelected;
  }
}
function anchorSelect(b) {
  const v = b.anchor || "default";
  return `<select id="anchor-sel">${[["default", "default"], ["feet", "feet"], ["center", "center"]].map(([val, t]) => `<option value="${val}" ${v === val ? "selected" : ""}>${t}</option>`).join("")}</select>`;
}
function wireAnchor(b, kind) {
  $("anchor-sel").onchange = (e) => { b.anchor = e.target.value === "default" ? null : e.target.value; draw(); renderSelectedPanel(); scheduleSave(); };
}
function renamePlayer(b, newId) {
  if (!newId || newId === b.id) return;
  if (!state.players[newId]) state.players[newId] = { ...playerMeta(b.id) };
  b.id = newId; selected = { kind: "player", id: newId };
  draw(); renderSidebar(); scheduleSave();
}
function fmtXY(xy) { return xy ? `(${xy[0].toFixed(2)}, ${xy[1].toFixed(2)})` : "— (no homography)"; }
function kickType(b) { return !b.kick ? "none" : (b.kick.byPlayerId != null ? "by" : "to"); }
function kickId(b) { return !b.kick ? "" : (b.kick.byPlayerId ?? b.kick.toPlayerId ?? ""); }
function setKick(b, type, id) {
  if (type === "none" || !id) b.kick = type === "none" ? null : b.kick;
  if (type === "by") b.kick = { byPlayerId: String(id) };
  else if (type === "to") b.kick = { toPlayerId: String(id) };
  else if (type === "none") b.kick = null;
}

// ---------- correct rosters UI ----------
function renderCorrectSets() {
  const ed = $("correct-editor");
  ed.innerHTML = `<div class="field">off <input id="cs-off" placeholder="1,2,3"></div>
    <div class="field">def <input id="cs-def" placeholder="4,5,6"></div>
    <div class="field">end <select id="cs-end"><option value="">(all subsequent)</option>${labelFrames().map((n) => `<option value="${n}">${frameIndex(n)}</option>`).join("")}</select></div>
    <button id="cs-add-current">+ Add current boxes as set (frame ${frameIndex(curFrame)})</button>
    <button id="cs-add">Add typed set from frame ${frameIndex(curFrame)}</button>`;
  const parse = (s) => s.split(/[,\s]+/).map((x) => x.trim()).filter(Boolean);
  const addSet = (offence, defence) => {
    state.correctSets.push({ offence, defence, start: curFrame, end: $("cs-end").value || null });
    renderSidebar(); scheduleSave();
  };
  $("cs-add").onclick = () => addSet(parse($("cs-off").value), parse($("cs-def").value));
  $("cs-add-current").onclick = () => {
    // group this frame's player boxes by their offence/defence mapping
    const off = [], def = [];
    for (const b of frameState(curFrame).players)
      (state.teamMapping[playerMeta(b.id).team] === "Offence" ? off : def).push(b.id);
    addSet(off, def);
  };
  const sets = $("correct-sets"); sets.innerHTML = "";
  state.correctSets.forEach((s, i) => {
    const row = document.createElement("div"); row.className = "set-row";
    row.innerHTML = `<span>${frameIndex(s.start)}–${s.end ? frameIndex(s.end) : "∞"}: O[${s.offence.join(",")}] D[${s.defence.join(",")}]</span><button class="del">×</button>`;
    row.querySelector(".del").onclick = () => { state.correctSets.splice(i, 1); renderSidebar(); scheduleSave(); };
    sets.appendChild(row);
  });
  renderFlagged();
}
function renderFlagged() {
  const box = $("flagged"); box.innerHTML = "";
  const flagged = labelFrames().map((n) => [n, frameFlags(n)]).filter(([, f]) => f);
  if (!state.correctSets.length) return;
  const strip = document.createElement("div"); strip.className = "";
  const title = document.createElement("div"); title.className = "muted";
  title.textContent = flagged.length ? `${flagged.length} frame(s) flagged:` : "No roster problems ✓";
  box.appendChild(title);
  flagged.forEach(([n, f]) => {
    const line = document.createElement("div"); line.className = "flagline";
    const set = applicableSet(n);
    // team letter from the roster spec for missing ids, from player meta for extras
    const specTeam = (id) => set.offence.map(String).includes(id) ? "O"
      : (set.defence.map(String).includes(id) ? "D" : "?");
    const metaTeam = (id) => state.players[id]
      ? state.teamMapping[state.players[id].team][0] : "?";
    const tag = (id, fn) => `${id}(${fn(id)})`;
    const parts = [];
    if (f.missing.length) parts.push("missing " + f.missing.map((id) => tag(id, specTeam)).join(", "));
    if (f.extra.length) parts.push("extra " + f.extra.map((id) => tag(id, metaTeam)).join(", "));
    const btn = document.createElement("button"); btn.className = "seg-sm";
    btn.textContent = `frame ${frameIndex(n)}`; btn.onclick = () => loadFrame(n);
    line.appendChild(btn);
    const span = document.createElement("span"); span.className = "muted"; span.textContent = " " + parts.join("; ");
    line.appendChild(span); box.appendChild(line);
  });
}

// ---------- persistence & export ----------
function cleanState() {
  // strip transient _seeded flags before saving
  const copy = JSON.parse(JSON.stringify(state));
  for (const f of Object.values(copy.frames)) delete f._seeded;
  return copy;
}
function scheduleSave() {
  $("save-status").textContent = "saving…";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      await fetch("/api/state", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cleanState()) });
      $("save-status").textContent = "all changes saved";
    } catch { $("save-status").textContent = "save failed!"; }
  }, 400);
}

// ---------- controls wiring ----------
function wireControls() {
  $("prev").onclick = () => navigate(-1);
  $("next").onclick = () => navigate(1);
  $("toprocess-toggle").onchange = toggleToProcess;
  $("start-label").onclick = () => setMode("calib");
  $("mode-calib").onclick = () => setMode("calib");
  $("mode-boxes").onclick = () => setMode("boxes");
  $("back-select").onclick = () => setMode("select");
  $("overlay-toggle").onchange = () => { pitchLines = null; refreshOverlay(); draw(); };
  $("copy-prev-dots").onclick = copyPrevDots;
  $("add-player").onclick = () => { pendingAdd = "player"; $("export-status").textContent = "click on the frame to place the new player"; };
  $("add-ball").onclick = () => { pendingAdd = "ball"; $("export-status").textContent = "click on the frame to place the ball"; };
  $("isolation-toggle").onchange = (e) => { isolation = e.target.checked; draw(); };
  $("box-search").oninput = (e) => { boxSearch = e.target.value; renderBoxList(); };
  $("teamA-toggle").onclick = () => {
    state.teamMapping.A = state.teamMapping.A === "Offence" ? "Defence" : "Offence";
    state.teamMapping.B = state.teamMapping.A === "Offence" ? "Defence" : "Offence";
    renderSidebar(); draw(); scheduleSave();
  };
  $("export").onclick = doExport;
}
function copyPrevDots() {
  const list = labelFrames(); const i = list.indexOf(curFrame);
  if (i <= 0) return;
  const prev = state.calibration[list[i - 1]] || [];
  if (!prev.length) return;
  state.calibration[curFrame] = prev.map((d) => ({ ...d }));
  selectedDot = -1; draw(); renderSidebar(); scheduleSave(); ensureHomography(curFrame); refreshOverlay();
}
async function doExport() {
  const status = $("export-status"); status.className = ""; status.textContent = "exporting…";
  const resp = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cleanState()) });
  if (!resp.ok) { const b = await resp.json().catch(() => ({})); status.className = "error"; status.textContent = "export failed: " + (b.error || resp.status); return; }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = "tracking.json"; a.click();
  URL.revokeObjectURL(url);
  status.className = ""; status.textContent = "downloaded tracking.json ✓";
}

init();

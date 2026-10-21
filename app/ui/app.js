// Latent UI. Talks to the Rust core through Tauri commands and events (docs/PLAN.md section 5).
// Built the way Mimic's is: the composer reads as a sentence, Controls slide in from the left, frames are takes.
const { invoke, convertFileSrc: fileSrc } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// Script errors go to logs/ui.log, since a desktop window has no visible console.
const trace = message => invoke("ui_error", { message });
addEventListener("error", e => trace(`${e.message} at ${e.filename}:${e.lineno}`));
addEventListener("unhandledrejection", e => trace(`Unhandled: ${e.reason}`));

const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const gb = n => (n >= 10 ? n.toFixed(0) : n.toFixed(1));
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

const state = {
  settings: null,
  hw: null,
  device: "",
  version: "",
  models: [],
  sources: [],
  frames: [],
  model: "",            // the model the next batch uses
  engine: {},           // model id -> "cold" | "loading" | "warm" | "busy" | "unloading"
  resident: null,
  idleLeft: null,
  attached: [],         // source ids for the next batch
  steps: null,          // null = the model's default
  seed: "",             // "" = random
  run: null,            // the batch being developed
  selected: null,       // frame id on the canvas
  follow: true,         // the canvas follows the developing frame until another frame is picked
  installing: {},       // model id -> latest install event
  hover: null,          // model id whose footprint the memory bar outlines
  error: null,          // the last batch's failure, shown until the next one starts
};

const model = () => state.models.find(m => m.id === state.model);
const installed = () => state.models.filter(m => m.installed);
const variant = m => m.variants.find(v => v.id === m.recommended) || m.variants[0]; // the core picks; this mirrors it
/** The size the next batch uses: for an edit, the model size closest to the first photo's shape; else the aspect picked. */
function sizeFor(m) {
  const src = m.can.edit && state.sources.find(s => s.id === state.attached[0]);
  if (!src) return m.sizes.find(s => s.aspect === state.settings.aspect) || m.sizes[0];
  const off = s => Math.abs(Math.log((s.width / s.height) / (src.width / src.height)));
  return m.sizes.reduce((best, s) => (off(s) < off(best) ? s : best));
}
const MAX_REFS = 10;
/** Click a source: models that take several references collect them; the others use one photo at a time. */
function toggleSource(id) {
  const multi = model()?.can.multiRef;
  if (state.attached.includes(id)) state.attached = state.attached.filter(x => x !== id);
  else state.attached = multi ? [...state.attached, id].slice(-MAX_REFS) : [id];
  render();
}
const frame = id => state.frames.find(f => f.id === id);
const unsaved = () => state.frames.filter(f => !f.saved).length;
const modelName = id => state.models.find(m => m.id === id)?.name ?? id;

function log(text, kind = "") {
  const el = $("#log");
  el.textContent = text;
  el.className = kind;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  log("Copied the install command.");
}

let saveTimer;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => invoke("save_settings", { settings: state.settings }).catch(e => log(String(e), "err")), 300);
}

// Caelestia's live Material You scheme (read from ~/.local/state/caelestia/scheme.json by the core), mapped onto
// Latent's tokens as in Mimic. Its primary color takes the accent's place.
let caelestia = null;
const CAELESTIA_MAP = {
  "--ground": "surface", "--surface": "surfaceContainerLow", "--surface-2": "surfaceContainer", "--surface-3": "surfaceContainerHighest",
  "--ink": "onSurface", "--ink-2": "onSurfaceVariant", "--ink-3": "outline", "--live": "primary", "--live-ink": "onPrimary",
  "--ready": "tertiary", "--warn": "error", "--danger": "error", "--field": "surfaceContainerLowest", "--field-line": "outlineVariant",
  "--checker": "surfaceContainer",
};

function applyCaelestia(on) {
  const root = document.documentElement.style;
  for (const [token, key] of Object.entries(CAELESTIA_MAP)) {
    const value = on && caelestia?.colors?.[key];
    value ? root.setProperty(token, value) : root.removeProperty(token);
  }
}

function applyTheme() {
  const t = state.settings.theme;
  const useCaelestia = t === "Caelestia" && caelestia;
  const light = useCaelestia ? caelestia.mode === "light"
    : t === "Light" || (t === "System" && matchMedia("(prefers-color-scheme: light)").matches);
  document.documentElement.dataset.theme = light ? "light" : "dark";
  applyCaelestia(useCaelestia);
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", applyTheme);

// ---------- top bar: the memory bar is the model picker ----------
function engineState(id) {
  if (state.resident === id) return state.engine[id] === "busy" ? "busy" : "warm";
  return state.engine[id] === "loading" ? "loading" : "cold";
}

/** Installed models that are pinned, in pin order. */
const pinned = () => (state.settings.pinned || []).map(id => state.models.find(m => m.id === id && m.installed)).filter(Boolean);
const isPinned = id => (state.settings.pinned || []).includes(id);

/** The top bar: pinned models, or every installed one when none are pinned; the model in use always shows. */
function topBarModels() {
  const list = pinned().length ? pinned() : installed();
  const current = model();
  return current && !list.includes(current) ? [...list, current] : list;
}

function togglePin(id) {
  const pins = (state.settings.pinned || []).filter(x => state.models.some(m => m.id === x && m.installed));
  state.settings.pinned = pins.includes(id) ? pins.filter(x => x !== id) : [...pins, id];
  saveSettings();
  render();
}

function renderTop() {
  const list = topBarModels();
  const hidden = installed().length - list.length;
  $("#modelBtns").innerHTML = list.map(m => {
    const s = engineState(m.id);
    const label = { warm: "loaded", busy: "developing", loading: "loading", cold: "not loaded" }[s];
    return `<button class="model-btn" data-id="${m.id}" aria-pressed="${m.id === state.model}">
      <b>${esc(m.name)}</b><span>${gb(variant(m).vram)} GB</span><span class="state" data-s="${s}">${label}</span></button>`;
  }).join("") + `<button class="model-btn" id="modelsBtn" title="${hidden ? `${hidden} more installed, not pinned` : "Download, remove and pin models"}">${
       !installed().length ? "Get a model…" : hidden ? `Models… <span class="meta">+${hidden}</span>` : "Models…"}</button>`;
  renderMem();
}

function renderMem() {
  const hw = state.hw;
  if (!hw) return;
  const gpu = hw.vramTotal > 0 && state.device !== "cpu";
  const [used, total, unit] = gpu ? [hw.vramUsed, hw.vramTotal, "VRAM"] : [hw.ramUsed, hw.ramTotal, "RAM"];
  $(".bar .fill").style.transform = `scaleX(${total ? Math.min(1, used / total) : 0})`;
  const ghost = $(".bar .ghost");
  const hovered = state.models.find(m => m.id === state.hover);
  let text = `<b>${gb(used)}</b> of ${gb(total)} GB ${unit}`;
  if (hovered && state.resident !== hovered.id && total) {
    const need = gpu ? variant(hovered).vram : variant(hovered).ram;
    ghost.style.left = `${Math.min(100, (used / total) * 100)}%`;
    ghost.style.width = `${(need / total) * 100}%`;
    ghost.classList.add("on");
    text += ` · ${esc(hovered.name)} needs about ${gb(need)} GB`;
  } else {
    ghost.classList.remove("on");
    if (state.resident) text += ` · ${esc(modelName(state.resident))} loaded`;
  }
  $("#memText").innerHTML = text;
}

function selectModel(id) {
  state.model = id;
  state.steps = null;
  if (!model()?.can.multiRef) state.attached = state.attached.slice(0, 1);
  render();
  if (state.resident !== id && !state.run) invoke("preload", { model: id, variant: null }).catch(e => log(String(e), "err"));
}

// ---------- sources ----------
function renderSources() {
  $("#sourceCount").textContent = state.sources.length || "";
  $("#sourceList").innerHTML = state.sources.map(s => `
    <button class="source" data-source="${s.id}" aria-pressed="${state.attached.includes(s.id)}" title="${state.attached.includes(s.id) ? "Detach" : "Edit this photo with the prompt"}">
      <img src="${fileSrc(s.path)}" alt="" loading="lazy">
      <span><span class="name">${esc(s.name)}</span><br><span class="dims">${s.width} × ${s.height}</span></span>
      <span class="x" data-delete="${s.id}" role="button" aria-label="Remove ${esc(s.name)}" title="Remove">✕</span>
    </button>`).join("");
}

async function importSources(paths) {
  try {
    const added = await invoke("import_sources", { paths: paths ?? null });
    if (!added.length) return;
    state.sources = [...added.reverse(), ...state.sources];
    renderSources();
    log(`Imported ${plural(added.length, "photo")}. Click one to edit it with the prompt.`);
  } catch (e) { log(String(e), "err"); }
}

// ---------- composer: a sentence of choices, as in Mimic ----------
function renderComposer() {
  closeMenu();
  const m = model();
  const tok = (menu, html) => `<button class="token" data-menu="${menu}" aria-expanded="false">${html}</button>`;
  const photo = state.sources.find(s => s.id === state.attached[0]);
  const count = tok("count", plural(state.settings.count, "frame"));
  if (!m) $("#recipe").innerHTML = `Make it with ${tok("model", "…")}`;
  else if (state.attached.length > 1) $("#recipe").innerHTML = `Combine ${tok("photo", `<img src="${fileSrc(photo.path)}" alt="">${state.attached.length} photos`)} with ${tok("model", esc(m.name))} · ${count}`;
  else if (photo) $("#recipe").innerHTML = `Edit ${tok("photo", `<img src="${fileSrc(photo.path)}" alt="">${esc(photo.name)}`)} with ${tok("model", esc(m.name))} · ${count}`;
  else $("#recipe").innerHTML = `Make it with ${tok("model", esc(m.name))} at ${tok("aspect", sizeFor(m).aspect)} · ${count}`;
  $("#controlsBtn .dot").hidden = state.steps === null && state.seed === "";
  renderPreflight();
}

/** Why Generate can't run right now (blocking), or what it will do. */
function preflight() {
  const m = model();
  if (state.run) return { busy: true, text: "Developing. Esc cancels the batch; the model stays loaded." };
  if (!installed().length) return { block: true, text: "Download a model first.", fix: "models" };
  if (!m) return { block: true, text: "Pick a model." };
  if (state.attached.length && !m.can.edit) return { block: true, warn: true, text: `${m.name} makes new images. Detach the photo, or pick a model that can edit.` };
  if (variant(m).fits === "no") return { block: true, warn: true, text: `${m.name} needs about ${gb(variant(m).vram)} GB, more than this PC has.` };
  if (state.error) return { error: true, text: state.error };
  if (!$("#prompt").value.trim()) return { block: true, text: "Write what to make." };
  const hw = state.hw;
  const free = hw && hw.vramTotal > 0 && state.device !== "cpu" ? hw.vramTotal - hw.vramUsed : null;
  if (free !== null && state.resident !== m.id && free < variant(m).vram) {
    return { warn: true, text: `Only ${gb(free)} GB of VRAM is free and ${m.name} needs about ${gb(variant(m).vram)} GB. Close other GPU apps (a local LLM server, a game) or it will run slowly.` };
  }
  const s = sizeFor(m);
  const steps = state.steps ?? m.options.steps.default;
  const load = state.resident === m.id ? "loaded, starts now" : "loads first (about a minute)";
  return { text: `${s.width} × ${s.height} · ${steps} steps · ${state.seed === "" ? "random seed" : "seed " + state.seed} · ${load}` };
}

function renderPreflight() {
  const p = preflight();
  const el = $("#preflight");
  el.className = "preflight" + (p.warn ? " block" : "") + (p.error ? " error" : "");
  el.innerHTML = esc(p.text) + (p.fix === "models" ? ` <button data-open-models>Open Models</button>` : "");
  const btn = $("#genBtn");
  btn.innerHTML = p.busy ? "Cancel <kbd>Esc</kbd>" : "Generate <kbd>Ctrl Enter</kbd>";
  btn.classList.toggle("live", !p.busy);
  btn.disabled = !p.busy && !!p.block;
}

// ---------- menus (open from their token) ----------
let menuFor = null;
const COUNTS = [1, 2, 4, 8];
function menuHtml(kind) {
  const m = model();
  if (kind === "model") {
    const item = x => {
      const v = variant(x);
      const does = x.can.edit ? "makes and edits images" : "makes new images";
      return `<button class="mi" data-pick-model="${x.id}" aria-checked="${x.id === state.model}"><b>${esc(x.name)}<em>${{ warm: "loaded", busy: "developing", loading: "loading", cold: "not loaded" }[engineState(x.id)]}</em></b>
        <span>${does} · about ${gb(v.vram)} GB · ${x.options.steps.default} steps</span></button>`;
    };
    const pins = pinned();
    const rest = installed().filter(x => !pins.includes(x));
    return (pins.length
      ? `<p class="menu-title">Pinned</p>${pins.map(item).join("")}${rest.length ? `<p class="menu-title">Not pinned</p>${rest.map(item).join("")}` : ""}`
      : `<p class="menu-title">Model</p>${rest.map(item).join("")}`)
      + `<div class="menu-sep"></div><button class="mi" data-open-models><b>More models…</b><span>Download, remove and pin models</span></button>`;
  }
  if (kind === "aspect") return `<p class="menu-title">Shape</p>` + m.sizes.map(s =>
    `<button class="mi" data-aspect="${s.aspect}" aria-checked="${s.aspect === sizeFor(m).aspect}"><b>${s.aspect}<em>${s.width} × ${s.height}</em></b></button>`).join("");
  if (kind === "count") return `<p class="menu-title">Frames per batch</p>` + COUNTS.map(n =>
    `<button class="mi" data-count="${n}" aria-checked="${n === state.settings.count}"><b>${plural(n, "frame")}</b><span>${n === 1 ? "One at a time" : `Seeds in a row, so each frame differs`}</span></button>`).join("");
  if (kind === "photo") return `<p class="menu-title">${m.can.multiRef ? `Photos to use (up to ${MAX_REFS})` : "Photo to edit"}</p>` + state.sources.map(s =>
    `<button class="mi" data-attach="${s.id}" aria-checked="${state.attached.includes(s.id)}"><b>${esc(s.name)}<em>${s.width} × ${s.height}</em></b></button>`).join("")
    + `<div class="menu-sep"></div><button class="mi" data-detach><b>No photo</b><span>Make a new image instead</span></button>`;
  return "";
}

function openMenu(kind, trigger) {
  if (menuFor === trigger) return closeMenu();
  closeMenu();
  const m = $("#menu");
  m.innerHTML = menuHtml(kind);
  m.classList.add("open");
  menuFor = trigger;
  trigger.setAttribute("aria-expanded", "true");
  const r = trigger.getBoundingClientRect(), mw = m.offsetWidth, mh = m.offsetHeight;
  const left = Math.max(12, Math.min(r.left, innerWidth - mw - 12));
  const above = r.top - mh - 8 > 8;
  m.style.left = `${left}px`;
  m.style.top = `${above ? r.top - mh - 8 : r.bottom + 8}px`;
  m.style.setProperty("--origin", `${r.left - left + r.width / 2}px ${above ? "100%" : "0"}`);
}

function closeMenu() {
  if (!menuFor) return false;
  menuFor.setAttribute("aria-expanded", "false");
  menuFor = null;
  $("#menu").classList.remove("open");
  return true;
}

// ---------- controls drawer ----------
function renderControls() {
  const m = model();
  $("#controlsFor").textContent = m ? `for ${m.name}` : "";
  if (!m) { $("#controlsBody").innerHTML = `<p class="ctl-none">Pick a model first.</p>`; return; }
  const r = m.options.steps;
  const steps = state.steps ?? r.default;
  const pct = r.max > r.min ? ((steps - r.min) / (r.max - r.min)) * 100 : 100;
  const fixed = r.min === r.max;
  $("#controlsBody").innerHTML = (fixed ? `
    <div class="ctl">
      <div class="ctl-top"><b>Steps</b><span class="ctl-val def">${r.default}</span></div>
      <p>${esc(m.name)} runs its own ${r.default}-step schedule, so the step count can't be changed.</p>
    </div>` : `
    <div class="ctl">
      <div class="ctl-top"><b id="ctl-steps">Steps</b>
        ${state.steps !== null ? `<button class="ctl-reset" data-reset="steps" title="Back to ${r.default}">Reset</button>` : ""}
        <span class="ctl-val ${state.steps === null ? "def" : ""}">${steps}</span></div>
      <input class="slider" type="range" id="steps" min="${r.min}" max="${r.max}" step="1" value="${steps}" aria-labelledby="ctl-steps" style="--pct:${pct}%">
      <p>Passes that refine the image. ${m.name} is tuned for ${r.default}; more is slower and rarely better.</p>
    </div>`) + `
    <div class="ctl">
      <div class="ctl-top"><b id="ctl-seed">Seed</b>
        ${state.seed !== "" ? `<button class="ctl-reset" data-reset="seed" title="Back to random">Random</button>` : ""}
        <span class="ctl-val ${state.seed === "" ? "def" : ""}">${state.seed === "" ? "random" : esc(state.seed)}</span></div>
      <input type="text" id="seed" inputmode="numeric" placeholder="Random" value="${esc(state.seed)}" aria-labelledby="ctl-seed" autocomplete="off" spellcheck="false">
      <p>The same seed, prompt and settings make the same image. Frames in a batch use the seeds after it.</p>
    </div>`;
}

function openControls(open = !$("#controls").classList.contains("open")) {
  const d = $("#controls");
  d.classList.toggle("open", open);
  d.inert = !open;
  $("#controlsBtn").setAttribute("aria-expanded", open);
  if (open) renderControls();
}

// ---------- generate ----------
async function generate() {
  if (state.run) return cancel();
  const p = preflight();
  if (p.block) return;
  const m = model();
  const seed = state.seed === "" ? null : Number(state.seed);
  if (seed !== null && !(Number.isInteger(seed) && seed >= 0)) { state.error = "The seed is a whole number, or empty for a random one."; return renderPreflight(); }
  const s = sizeFor(m);
  const request = {
    model: m.id, prompt: $("#prompt").value.trim(), sources: state.attached, width: s.width, height: s.height,
    count: state.settings.count, seed, options: state.steps !== null ? { steps: state.steps } : {},
  };
  state.error = null;
  // Set before the command returns: a warm model can report its first step before the batch id arrives.
  state.run = { batch: null, count: request.count, model: m.id, aspect: s.aspect, width: s.width, height: s.height, prompt: request.prompt,
                index: 0, step: 0, steps: state.steps ?? m.options.steps.default, preview: null, secPerStep: null, started: Date.now() };
  state.follow = true;
  render();
  try {
    const { batch } = await invoke("generate", { request });
    if (state.run && state.run.batch === null) state.run.batch = batch;
  } catch (e) {
    state.run = null;
    state.error = String(e);
    render();
  }
}

function cancel() {
  if (!state.run?.batch) return;
  invoke("cancel", { batch: state.run.batch });
  log("Cancelling. The model stays loaded.");
}

const ours = batch => state.run && (state.run.batch === null || state.run.batch === batch);

// ---------- canvas ----------
function showFull(path) {
  const img = $("#canvasImg");
  if (img.dataset.path === path) return;
  img.classList.remove("on");
  img.onload = () => img.classList.add("on"); // the finished frame fades in over its last preview
  img.dataset.path = path;
  img.src = fileSrc(path);
}

function clearFull() {
  const img = $("#canvasImg");
  img.classList.remove("on");
  img.removeAttribute("src");
  delete img.dataset.path;
}

function renderCanvas() {
  const empty = $("#canvasEmpty"), step = $("#canvasStep"), preview = $("#canvasPreview");
  const run = state.run, f = frame(state.selected);
  empty.hidden = true;
  step.hidden = true;

  if (run && state.follow) {
    clearFull();
    delete preview.dataset.frame;
    if (run.preview) preview.src = `${fileSrc(run.preview)}?s=${run.index}-${run.step}`;
    else preview.removeAttribute("src");
    step.innerHTML = state.engine[run.model] === "loading"
      ? `Loading <b>${esc(modelName(run.model))}</b>. The first load reads the whole model from disk.`
      : run.step === 0
        ? `Frame ${run.index + 1} of ${run.count} · reading the prompt`
        : `Frame ${run.index + 1} of ${run.count} · step <b>${run.step}</b> of ${run.steps}` +
          (run.secPerStep ? ` · ${run.secPerStep.toFixed(1)} s per step` : "");
    step.hidden = false;
    return;
  }
  if (f) {
    if (preview.dataset.frame !== f.id) { preview.src = fileSrc(f.thumb); preview.dataset.frame = f.id; }
    showFull(f.path);
    return;
  }
  preview.removeAttribute("src");
  clearFull();
  empty.hidden = false;
  empty.innerHTML = installed().length ? emptyReady() : emptyNoModel();
}

const MARK = () => $(".wordmark .mark").outerHTML;

function emptyReady() {
  return `${MARK()}<h3>Your first frame develops here.</h3><p>Write what to make below, then press Ctrl Enter.</p>`;
}

function emptyNoModel() {
  const m = state.models[0];
  if (!m) return "<h3>No models are available in this version.</h3>";
  const v = variant(m);
  const fit = { yes: "Fits this PC.", tight: "Fits, with little room to spare.", no: "Needs more memory than this PC has." }[v.fits];
  return `${MARK()}<h3>Download a model to start.</h3>
    <div class="starter"><div><b>${esc(m.name)}</b> · ${esc(m.license)}</div>
    <div>Makes new images in ${m.options.steps.default} steps. ${gb(v.disk)} GB download, about ${gb(v.vram)} GB of VRAM while it works. ${fit}</div>
    ${installRow(m)}</div>`;
}

function installRow(m) {
  const i = state.installing[m.id];
  if (i && i.phase !== "failed") {
    const pct = i.total && i.done != null ? Math.min(1, i.done / i.total) : null;
    const bytes = pct !== null ? ` · ${gb(i.done / 1e9)} of ${gb(i.total / 1e9)} GB` : "";
    return `<div class="progress${pct === null ? " indeterminate" : ""}"><span style="transform:scaleX(${pct ?? 1})"></span></div>
      <div class="meta">${esc(i.label)}${bytes}</div>`;
  }
  const failed = i?.phase === "failed" ? `<div class="preflight error">${esc(i.label)}</div>` : "";
  const copy = i?.phase === "failed" && i.command
    ? `<button class="btn small" data-copy="${esc(i.command)}">Copy install command</button>` : "";
  return `${failed}<div class="row"><button class="btn small" data-install="${m.id}">${failed ? "Try again" : "Download"}</button>${copy}</div>`;
}

/** The catalog again from the core, after a setting that changes which variant runs. */
async function refreshModels() {
  await new Promise(r => setTimeout(r, 400)); // after the settings save lands
  try { state.models = (await invoke("bootstrap")).models; render(); } catch {}
}

async function installModel(id) {
  state.installing[id] = { phase: "env", label: "Starting" };
  renderModels();
  try {
    state.models = await invoke("install_model", { id });
    delete state.installing[id];
    if (!state.model) state.model = id;
    log(`${modelName(id)} is ready.`);
  } catch (e) {
    // Keep the command the "install" event may already have put here (the missing-uv hint).
    state.installing[id] = { ...state.installing[id], phase: "failed", label: String(e) };
  }
  render();
}

// ---------- frames: each batch is a card like a Mimic take, newest first ----------
function tile(f, live) {
  return `<button class="tile" role="option" data-frame="${f.id}" aria-selected="${f.id === state.selected && !(live && state.follow)}" style="--ar:${f.width} / ${f.height}" title="Seed ${f.seed}">
    <img src="${fileSrc(f.thumb)}" alt="Frame ${f.index + 1}, seed ${f.seed}" loading="lazy">${f.saved ? '<span class="saved" title="Saved"></span>' : ""}</button>`;
}

function renderFrames() {
  const batches = new Map();
  for (const f of state.frames) {
    if (!batches.has(f.batch)) batches.set(f.batch, []);
    batches.get(f.batch).push(f);
  }
  const run = state.run;
  const runKey = run?.batch ?? "pending";
  if (run && !batches.has(runKey)) batches.set(runKey, []);
  const sel = frame(state.selected);
  const entries = [...batches.entries()];
  $("#frameList").innerHTML = entries.map(([batch, frames], n) => {
    const live = !!run && batch === runKey;
    const first = frames[0];
    const current = live ? state.follow : sel?.batch === batch;
    const name = modelName(first?.model ?? run?.model);
    const sized = state.models.find(x => x.id === first?.model)?.sizes.find(z => z.width === first.width && z.height === first.height);
    const shape = live ? run.aspect : sized?.aspect ?? `${first.width} × ${first.height}`;
    const dur = live
      ? `<span class="take-dur live">developing</span>`
      : `<span class="take-dur">${(frames.reduce((a, f) => a + f.elapsed, 0) / frames.length).toFixed(1)} s each</span>`;
    let tiles = frames.map(f => tile(f, live)).join("");
    if (live) {
      for (let i = frames.length; i < run.count; i++) {
        const ar = `${run.width} / ${run.height}`;
        tiles += i === run.index
          ? `<button class="tile developing" data-developing aria-selected="${state.follow}" style="--ar:${ar}" title="Developing">${run.preview ? `<img src="${fileSrc(run.preview)}?s=${run.index}-${run.step}" alt="">` : ""}
              <span class="prog" style="transform:scaleX(${run.steps ? run.step / run.steps : 0})"></span></button>`
          : `<div class="tile queued" style="--ar:${ar}"></div>`;
      }
    }
    const prompt = live ? run.prompt : first.recipe?.prompt;
    let detail = "";
    if (current) {
      detail = `<p class="take-text">${esc(prompt)}</p>`;
      if (!live && sel) {
        detail += sel.saved ? `<p class="take-path">Saved to <button data-reveal>${esc(sel.saved)}</button></p>`
                            : `<p class="take-path">Frame ${sel.index + 1} · seed ${sel.seed} · ${sel.width} × ${sel.height} · made in ${sel.elapsed.toFixed(1)} s</p>`;
        detail += `<div class="take-actions">
          <button class="btn" data-act="save" ${sel.saved ? "disabled" : ""} title="Ctrl S">Save</button>
          <button class="btn" data-act="saveas" title="Ctrl Shift S">Save as…</button>
          <button class="btn" data-act="reuse" title="Put this frame's prompt and settings back (R)">Reuse</button>
          <button class="btn" data-act="source" title="Edit this frame next (E)">Use as source</button>
          <button class="btn" data-act="delete" title="Delete">Delete</button></div>`;
      }
    }
    return `<div class="batch" aria-current="${current}">
      <div class="take-top"><span class="take-name">Batch ${n + 1} <span class="who">· ${esc(name)} · ${esc(shape)}</span></span>${dur}</div>
      <div class="sheet">${tiles}</div>${detail}</div>`;
  }).reverse().join("");
  const u = unsaved();
  $("#unsaved").textContent = u ? `${u} unsaved` : "";
}

function select(id) {
  state.selected = id;
  state.follow = false;
  renderFrames();
  renderCanvas();
}

function move(delta) {
  const order = [...state.frames].sort((a, b) => b.createdAt - a.createdAt || a.index - b.index);
  if (!order.length) return;
  const i = order.findIndex(f => f.id === state.selected);
  const next = order[Math.max(0, Math.min(order.length - 1, i < 0 ? 0 : i + delta))];
  select(next.id);
  document.querySelector(`[data-frame="${next.id}"]`)?.scrollIntoView({ block: "nearest" });
}

async function saveFrame(as = false) {
  const f = frame(state.selected);
  if (!f) return;
  try {
    const path = await invoke(as ? "save_frame_as" : "save_frame", { id: f.id });
    if (!path) return;
    f.saved = path;
    log(`Saved ${path}`);
    renderFrames();
  } catch (e) { log(String(e), "err"); }
}

function reuse() {
  const f = frame(state.selected);
  if (!f) return;
  const r = f.recipe;
  $("#prompt").value = r.prompt;
  autosize();
  if (state.models.some(m => m.id === r.model && m.installed)) state.model = r.model;
  const s = model()?.sizes.find(x => x.width === r.width && x.height === r.height);
  if (s) state.settings.aspect = s.aspect;
  state.steps = r.options?.steps ?? null;
  state.seed = String(r.seed);
  state.attached = (r.sources || []).filter(id => state.sources.some(x => x.id === id));
  render();
  log(`Reused frame ${f.index + 1}'s recipe, seed included. Set the seed back to random in Controls for new variations.`);
}

async function toSource() {
  const f = frame(state.selected);
  if (!f) return;
  try {
    const s = await invoke("frame_to_source", { frame: f.id });
    state.sources.unshift(s);
    state.attached = [s.id];
    render();
    log("The frame is now a source. Describe the change you want.");
    $("#prompt").focus();
  } catch (e) { log(String(e), "err"); }
}

async function deleteFrame() {
  const f = frame(state.selected);
  if (!f) return;
  const order = [...state.frames].sort((a, b) => b.createdAt - a.createdAt || a.index - b.index);
  const i = order.indexOf(f);
  await invoke("delete_frame", { id: f.id });
  state.frames = state.frames.filter(x => x !== f);
  state.selected = (order[i + 1] || order[i - 1])?.id ?? null;
  render();
}

// ---------- status ----------
function renderStatus() {
  const left = state.idleLeft;
  $("#idle").textContent = state.resident && left != null && !state.run
    ? `${modelName(state.resident)} unloads after ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} idle` : "";
}

// ---------- dialogs ----------
// Dialogs stack (an explainer opens over Settings); the newest is moved last in the page, so it's on top and closes first.
function openDialog(id) {
  closeMenu();
  document.body.append($(id));
  $(id).classList.add("open");
  $(id).querySelector("[data-close]")?.focus();
}
function closeDialogs() {
  const open = [...document.querySelectorAll(".overlay.open")].pop();
  open?.classList.remove("open");
  return !!open;
}

function renderModels() {
  if ($("#modelsDlg").classList.contains("open")) {
    $("#modelsBody").innerHTML = state.models.map(m => {
      const v = variant(m);
      const fit = { yes: "Fits this PC", tight: "Fits, just", no: "Doesn't fit this PC" }[v.fits];
      const does = m.can.edit ? "Makes and edits images" : "Makes new images";
      const account = m.account && !m.installed
        ? `<div class="meta">Needs a free Hugging Face account and its token ${`<button class="info" data-help="hfToken" aria-label="About Hugging Face tokens" title="What's this?">i</button>`}${state.settings.hfToken ? "" : " · no token set yet"}</div>` : "";
      const action = m.installed
        ? `<div class="row"><button class="btn small pin" data-pin="${m.id}" aria-pressed="${isPinned(m.id)}" title="${isPinned(m.id) ? "Shown in the top bar" : "Show in the top bar"}">${isPinned(m.id) ? "Pinned" : "Pin to top bar"}</button>
            <span class="grow"></span><button class="quiet danger" data-remove="${m.id}">Remove</button></div>`
        : account + installRow(m);
      return `<div class="model-card"><div class="title"><b>${esc(m.name)}</b><span class="fit-${v.fits}">${fit}</span></div>
        <div class="facts">${does} in ${m.options.steps.default} steps. ${gb(v.disk)} GB download · about ${gb(v.vram)} GB VRAM (${esc(v.label)}) ·
        ${esc(m.license)}${m.commercial ? "" : ", non-commercial"}</div>${action}</div>`;
    }).join("") + `<p class="meta">Models live in ${esc(state.settings.modelsDir)} <button class="info" data-help="modelsDir" aria-label="About the models folder" title="What's this?">i</button>. Removing one frees its disk space; Latent can download it again any time.</p>
      <p class="meta">Pinned models are the ones in the top bar. With none pinned, every installed model shows there; the rest stay in the model menu either way.</p>`;
  }
  if (!installed().length) renderCanvas();
}

const SELECTS = {
  get theme() { return caelestia ? ["Caelestia", "Dark", "Light", "System"] : ["Dark", "Light", "System"]; },
  promptSize: ["Small", "Medium", "Large"],
  format: ["PNG", "JPEG"],
  idle: ["5 minutes", "10 minutes", "30 minutes", "Never"],
  device: ["Automatic", "GPU", "CPU"],
};

const link = (url, text) => `<button class="link" data-url="${url}">${esc(text)}</button>`;

/** Explainers for the settings that need one. Each returns its body; some read live facts first. */
const HELP = {
  hfToken: {
    title: "Hugging Face token",
    body: () => {
      const gated = state.models.filter(m => m.account);
      return `
      <p>A few models are <b>gated</b>: their makers let anyone download them, but only from a free Hugging Face account
      that has accepted the model's license. Latent proves which account is asking with a <b>token</b>, a password made
      for apps, which you can delete at any time.</p>
      <p>Models here that need one: ${gated.map(m => `<b>${esc(m.name)}</b>`).join(", ") || "none"}. The others download without it.</p>
      <ol>
        <li>Sign in or make a free account at ${link("https://huggingface.co/join", "huggingface.co")}.</li>
        <li>Open each model's page and click <b>Agree and access repository</b>: ${gated.map(m => link(m.licenseUrl, m.name)).join(", ")}.</li>
        <li>Make a token at ${link("https://huggingface.co/settings/tokens", "huggingface.co/settings/tokens")}: <b>Create new token</b>, type <b>Read</b>.</li>
        <li>Paste it into Settings and press <b>Test</b>. Latent says whose token it is and which models it can download.</li>
      </ol>
      <p class="meta">The token stays in Latent's settings file on this PC and is only sent to Hugging Face, when a gated
      model downloads. Don't share it. If it ever leaks, delete it on the tokens page and make a new one.</p>`;
    },
  },
  device: {
    title: "Run models on",
    body: () => `
      <p><b>Automatic</b> uses this PC's GPU when it can run models, and the CPU otherwise.
      <b>GPU</b> insists on the graphics card; <b>CPU</b> never uses it (much slower, but works anywhere).</p>
      <p>This PC: <b>${esc(state.hw?.gpu || "no GPU found")}</b>, running on <b>${esc(state.device.toUpperCase())}</b>.${state.hw?.note ? " " + esc(state.hw.note) : ""}</p>
      ${state.hw?.fix ? `<p>To let the GPU run models: <code>${esc(state.hw.fix)}</code></p>` : ""}
      <p class="meta">A change applies the next time a model loads.</p>`,
  },
  lowVram: {
    title: "Prefer less VRAM",
    body: () => `
      <p>Some models come in more than one version: full quality, and a lighter one that stores its weights in FP8 and
      needs a fraction of the graphics memory. Normally Latent runs the best version that fits.</p>
      <p>Turn this on when something else needs the GPU at the same time, like a local LLM server or a game. Every
      model then runs its lightest version, so both fit. Expect a small loss of detail.</p>
      <p class="meta">Applies the next time a model loads. To apply it now, pick the model again in the top bar after it unloads.</p>`,
  },
  idle: {
    title: "Unload an idle model",
    body: () => `
      <p>A loaded model keeps its memory (often 10 GB or more of VRAM) even while you're not generating. After this
      long without a batch, Latent unloads it and the GPU is free for games and other apps.</p>
      <p class="meta">The next batch loads it again, which takes 20 to 60 seconds. <b>Never</b> keeps it loaded until you
      pick another model or close Latent.</p>`,
  },
  modelsDir: {
    title: "Models folder",
    body: async () => {
      let used = "";
      try { used = `It uses <b>${gb((await invoke("models_disk")) / 1e9)} GB</b> right now.`; } catch {}
      return `
      <p>Everything Latent downloads lives here: one Python environment with PyTorch (about 6 GB) and each model's
      weights (5 to 33 GB each). ${used}</p>
      <p>Pick a folder on a drive with room to spare. Changing the folder doesn't move what's already downloaded:
      move the folder yourself first, then point Latent at the new place, or download the models again.</p>
      <p><button class="btn small" data-reveal-path="${esc(state.settings.modelsDir)}">Open the folder</button></p>`;
    },
  },
  format: {
    title: "File format",
    body: () => `
      <p><b>PNG</b> saves the frame exactly as it came out, with its <b>recipe</b> (prompt, model, size, seed) and an
      AI-generated mark inside the file, so Reuse works on it later.</p>
      <p><b>JPEG</b> files are several times smaller, but have no place for the recipe or the mark.</p>
      <p class="meta">Save as… offers both either way.</p>`,
  },
  preload: {
    title: "Load the last model on start",
    body: () => `
      <p>Latent loads the model you used last in the background as soon as it opens, so the first batch starts
      right away instead of waiting 20 to 60 seconds.</p>
      <p class="meta">The model's memory is taken from the start, whether you generate or not.</p>`,
  },
};

async function openHelp(key) {
  const h = HELP[key];
  if (!h) return;
  $("#helpTitle").textContent = h.title;
  $("#helpBody").innerHTML = await h.body();
  openDialog("#helpDlg");
}

function renderSettings() {
  const s = state.settings;
  const info = key => `<button class="info" data-help="${key}" aria-label="About ${esc(HELP[key].title)}" title="What's this?">i</button>`;
  const label = (text, key) => `<span>${text}${key ? " " + info(key) : ""}</span>`;
  const select = key => `<select data-set="${key}">${SELECTS[key].map(o => `<option${o === s[key] ? " selected" : ""}>${o}</option>`).join("")}</select>`;
  const toggle = key => `<input type="checkbox" class="switch" data-set="${key}"${s[key] ? " checked" : ""}>`;
  const folder = (key, title) => `<div class="path"><input readonly value="${esc(s[key])}"><button class="btn small" data-folder="${key}" data-title="${title}">Change…</button></div>`;
  $("#settingsBody").innerHTML = `
    <div class="set-group">Appearance</div>
    <label class="setting">${label(caelestia ? "Theme (Caelestia follows your shell's colors)" : "Theme")}${select("theme")}</label>
    <label class="setting">${label("Prompt text size")}${select("promptSize")}</label>

    <div class="set-group">Saving</div>
    <div class="setting">${label("Save folder")}${folder("saveDir", "Save frames to")}</div>
    <label class="setting">${label("File format", "format")}${select("format")}</label>
    <label class="setting">${label("Warn about unsaved frames when closing")}${toggle("warnUnsaved")}</label>

    <div class="set-group">Models and memory</div>
    <div class="setting">${label("Models folder", "modelsDir")}${folder("modelsDir", "Keep models in")}</div>
    <label class="setting">${label("Prefer less VRAM", "lowVram")}${toggle("lowVram")}</label>
    <label class="setting">${label("Run models on", "device")}${select("device")}</label>
    <label class="setting">${label("Unload an idle model after", "idle")}${select("idle")}</label>
    <label class="setting">${label("Load the last model on start", "preload")}${toggle("preload")}</label>
    <div class="setting">${label("Hugging Face token", "hfToken")}
      <div class="path"><input type="password" data-set="hfToken" value="${esc(s.hfToken)}" placeholder="Only for gated models" autocomplete="off" spellcheck="false">
      <button class="btn small" id="testTokenBtn"${s.hfToken ? "" : " disabled"}>Test</button></div></div>
    <p class="meta token-result" id="tokenResult"></p>

    <div class="set-group">About</div>
    <p class="meta">Latent ${esc(state.version)} · ${esc(state.hw?.gpu || "no GPU found")}${state.hw?.note ? " · " + esc(state.hw.note) : ""}</p>
    <div class="row"><button class="btn small" id="checkUpdateBtn">Check for updates</button><button class="btn small" data-reveal-path="${esc(state.logs)}">Open logs folder</button></div>`;
}

async function testToken() {
  const out = $("#tokenResult");
  out.className = "meta token-result";
  out.textContent = "Asking Hugging Face…";
  try {
    const r = await invoke("test_hf_token", { token: state.settings.hfToken });
    const lines = state.models.filter(m => m.account).map(m =>
      r.access[m.id] ? `${esc(m.name)}: can download` : `${esc(m.name)}: not yet. Accept its license on ${link(m.licenseUrl, "its page")}, then test again`);
    out.innerHTML = `Works. Signed in as <b>${esc(r.name)}</b>.<br>${lines.join("<br>")}`;
  } catch (e) {
    out.className = "meta token-result err";
    out.textContent = String(e);
  }
}

function applyPromptSize() {
  document.documentElement.style.setProperty("--prompt-size", { Small: "21px", Medium: "26px", Large: "32px" }[state.settings.promptSize] || "26px");
  autosize();
}

// ---------- updates, as in Mimic: the release notes are the tag message ----------
let update = null;

function notesHtml(text) {
  let html = "", list = false;
  for (const line of text.split("\n")) {
    const bullet = /^\s*[-*]\s+(.*)/.exec(line), head = /^#{1,6}\s+(.*)/.exec(line);
    if (list && !bullet) { html += "</ul>"; list = false; }
    if (bullet) { if (!list) { html += "<ul>"; list = true; } html += `<li>${esc(bullet[1])}</li>`; }
    else if (head) html += `<h5>${esc(head[1])}</h5>`;
    else if (line.trim()) html += `<p>${esc(line)}</p>`;
  }
  return html + (list ? "</ul>" : "");
}

async function checkUpdate(quiet) {
  try { update = await invoke("check_update"); }
  catch (e) { if (!quiet) log(String(e), "err"); return; }
  $("#updateBtn").hidden = !update;
  if (update) {
    $("#updateBtn").textContent = `Update to ${update.version}`;
    if (!quiet) openUpdate();
  } else if (!quiet) log("Latent is up to date.");
}

function openUpdate() {
  if (!update) return;
  const n = unsaved();
  const pm = update.packageManager; // e.g. "pacman": that package manager updates Latent, not Latent itself
  $("#upTitle").textContent = pm ? `Latent ${update.version} is out` : `Latent ${update.version} is ready to install`;
  $("#upCurrent").textContent = pm
    ? `You have ${update.current}, installed with ${pm}. Update it the same way: set pkgver=${update.version} in packaging/arch/PKGBUILD, then run makepkg -si there.`
    : `You have ${update.current}. Latent downloads the new version, installs it and opens again.`;
  $("#upInstall").hidden = !!pm;
  $("#upLater").textContent = pm ? "Close" : "Later";
  $("#upNotes").innerHTML = notesHtml(update.notes || "");
  $("#upWarn").hidden = !n || !!pm;
  $("#upWarn").textContent = `${plural(n, "unsaved frame")} ${n === 1 ? "is" : "are"} cleared when Latent restarts. Save ${n === 1 ? "it" : "them"} first if you want to keep ${n === 1 ? "it" : "them"}.`;
  $("#upStatus").textContent = "";
  $(".up-progress").hidden = true;
  $("#upInstall").disabled = false;
  $("#upLater").disabled = false;
  $("#updateDlg").classList.add("open");
  $("#upInstall").focus();
}

async function installUpdate() {
  $("#upInstall").disabled = true;
  $("#upLater").disabled = true;
  $(".up-progress").hidden = false;
  $("#upBar").style.transform = "scaleX(0)";
  $("#upStatus").textContent = "Downloading…";
  trace(`installing update ${update.version}`);
  try {
    await invoke("install_update"); // restarts Latent when it's done
    $("#upStatus").textContent = "Restarting…";
  } catch (e) {
    $("#upStatus").textContent = String(e);
    $("#upInstall").disabled = false;
    $("#upLater").disabled = false;
  }
}

listen("update-progress", ({ payload }) => {
  if (!payload.total) { $("#upStatus").textContent = `Downloading… ${(payload.done / 1e6).toFixed(1)} MB`; return; }
  $("#upBar").style.transform = `scaleX(${payload.done / payload.total})`;
  $("#upStatus").textContent = payload.done >= payload.total ? "Installing…" : `Downloading… ${Math.round(payload.done / payload.total * 100)}%`;
});

// ---------- render ----------
function render() {
  renderTop();
  renderSources();
  renderComposer();
  renderFrames();
  renderCanvas();
  renderStatus();
  renderModels();
  if ($("#controls").classList.contains("open")) renderControls();
}

function autosize() {
  const p = $("#prompt");
  p.style.height = "auto";
  p.style.height = `${p.scrollHeight}px`;
}

// ---------- input ----------
$("#menu").addEventListener("click", e => {
  const b = e.target.closest(".mi");
  if (!b || b.disabled) return;
  closeMenu();
  if (b.dataset.pickModel) return selectModel(b.dataset.pickModel);
  if (b.dataset.aspect) { state.settings.aspect = b.dataset.aspect; saveSettings(); return renderComposer(); }
  if (b.dataset.count) { state.settings.count = Number(b.dataset.count); saveSettings(); return renderComposer(); }
  if (b.dataset.attach) return toggleSource(b.dataset.attach);
  if (b.dataset.detach !== undefined) { state.attached = []; return render(); }
  if (b.dataset.openModels !== undefined) { openDialog("#modelsDlg"); return renderModels(); }
});

document.addEventListener("click", async e => {
  const tokenBtn = e.target.closest("[data-menu]");
  if (tokenBtn) return openMenu(tokenBtn.dataset.menu, tokenBtn);
  if (menuFor && !e.target.closest("#menu")) closeMenu();
  const t = e.target.closest("button, [data-delete]");
  if (!t || t.closest("#menu")) return;
  if (t.dataset.delete) {
    e.stopPropagation();
    await invoke("delete_source", { id: t.dataset.delete }).catch(err => log(String(err), "err"));
    state.sources = state.sources.filter(s => s.id !== t.dataset.delete);
    state.attached = state.attached.filter(id => id !== t.dataset.delete);
    return render();
  }
  if (t.dataset.reveal !== undefined) return invoke("reveal", { path: frame(state.selected)?.saved });
  if (t.dataset.help) return openHelp(t.dataset.help);
  if (t.dataset.url) return invoke("open_url", { url: t.dataset.url }).catch(err => log(String(err), "err"));
  if (t.dataset.revealPath) return invoke("reveal", { path: t.dataset.revealPath });
  if (t.id === "testTokenBtn") return testToken();
  if (t.dataset.openModels !== undefined) { openDialog("#modelsDlg"); return renderModels(); }
  if (t.dataset.id && t.classList.contains("model-btn")) return selectModel(t.dataset.id);
  if (t.id === "modelsBtn") { openDialog("#modelsDlg"); return renderModels(); }
  if (t.id === "settingsBtn") { renderSettings(); return openDialog("#settingsDlg"); }
  if (t.dataset.close !== undefined) return t.closest(".overlay").classList.remove("open");
  if (t.dataset.install) return installModel(t.dataset.install);
  if (t.dataset.copy) return copyText(t.dataset.copy);
  if (t.dataset.pin) return togglePin(t.dataset.pin);
  if (t.dataset.remove) {
    const m = state.models.find(x => x.id === t.dataset.remove);
    if (!confirm(`Delete ${m.name}'s files (${gb(variant(m).disk)} GB)? You can download it again later.`)) return;
    try { state.models = await invoke("remove_model", { id: m.id }); } catch (err) { log(String(err), "err"); }
    if (state.model === m.id) state.model = installed()[0]?.id ?? "";
    return render();
  }
  if (t.dataset.folder) {
    const dir = await invoke("pick_folder", { title: t.dataset.title });
    if (dir) { state.settings[t.dataset.folder] = dir; saveSettings(); renderSettings(); }
    return;
  }
  if (t.dataset.source) return toggleSource(t.dataset.source);
  if (t.dataset.frame) return select(t.dataset.frame);
  if (t.dataset.developing !== undefined) { state.follow = true; renderFrames(); return renderCanvas(); }
  if (t.dataset.reset === "steps") { state.steps = null; renderControls(); return renderComposer(); }
  if (t.dataset.reset === "seed") { state.seed = ""; renderControls(); return renderComposer(); }
  switch (t.dataset.act) {
    case "save": return saveFrame();
    case "saveas": return saveFrame(true);
    case "reuse": return reuse();
    case "source": return toSource();
    case "delete": return deleteFrame();
  }
  switch (t.id) {
    case "updateBtn": return openUpdate();
    case "upInstall": return installUpdate();
    case "upLater": if (!$("#upInstall").disabled) $("#updateDlg").classList.remove("open"); return;
    case "checkUpdateBtn": return checkUpdate(false);
    case "genBtn": return generate();
    case "importBtn": return importSources();
    case "controlsBtn": return openControls();
    case "controlsClose": return openControls(false);
    case "controlsReset": state.steps = null; state.seed = ""; renderControls(); return renderComposer();
  }
});

document.addEventListener("mouseover", e => {
  const id = e.target.closest(".model-btn[data-id]")?.dataset.id ?? null;
  if (id !== state.hover) { state.hover = id; renderMem(); }
});

document.addEventListener("change", e => {
  const key = e.target.dataset.set;
  if (!key) return;
  state.settings[key] = e.target.type === "checkbox" ? e.target.checked : e.target.value;
  saveSettings();
  if (key === "theme") applyTheme();
  if (key === "promptSize") applyPromptSize();
  if (key === "lowVram" || key === "device") refreshModels();
});

document.addEventListener("input", e => {
  if (e.target.dataset.set === "hfToken") {
    state.settings.hfToken = e.target.value.trim();
    saveSettings();
    $("#testTokenBtn").disabled = !state.settings.hfToken;
    $("#tokenResult").textContent = "";
  }
  if (e.target.id === "prompt") { autosize(); state.error = null; renderPreflight(); }
  if (e.target.id === "steps") {
    const m = model();
    const v = Number(e.target.value);
    state.steps = v === m.options.steps.default ? null : v;
    const card = e.target.closest(".ctl");
    card.querySelector(".ctl-val").textContent = v;
    card.querySelector(".ctl-val").classList.toggle("def", state.steps === null);
    e.target.style.setProperty("--pct", `${((v - m.options.steps.min) / (m.options.steps.max - m.options.steps.min)) * 100}%`);
    renderComposer();
  }
  if (e.target.id === "seed") {
    state.seed = e.target.value.trim();
    state.error = null;
    const val = e.target.closest(".ctl").querySelector(".ctl-val");
    val.textContent = state.seed || "random";
    val.classList.toggle("def", state.seed === "");
    renderComposer();
  }
});

document.addEventListener("keydown", e => {
  const typing = e.target.matches("textarea, input, select");
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && e.key === "Enter") { e.preventDefault(); return generate(); }
  if (e.key === "Escape") {
    if (closeMenu() || closeDialogs()) return;
    if ($("#controls").classList.contains("open")) return openControls(false);
    if (state.run) return cancel();
    if (typing) return e.target.blur();
  }
  if (ctrl && e.key === ",") { e.preventDefault(); renderSettings(); return openDialog("#settingsDlg"); }
  if (ctrl && e.key.toLowerCase() === "s") { e.preventDefault(); return saveFrame(e.shiftKey); }
  if (typing || ctrl || e.altKey || document.querySelector(".overlay.open")) return;
  const k = e.key.toLowerCase();
  if (k === "arrowright" || k === "arrowdown" || k === "j") { e.preventDefault(); return move(1); }
  if (k === "arrowleft" || k === "arrowup" || k === "k") { e.preventDefault(); return move(-1); }
  if (k === "e") return toSource();
  if (k === "r") return reuse();
  if (k === "delete") return deleteFrame();
});

// Paste an image from the clipboard as a source.
document.addEventListener("paste", async e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
  if (!item) return;
  e.preventDefault();
  const blob = item.getAsFile();
  const data = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  try {
    const s = await invoke("paste_source", { png: data });
    state.sources.unshift(s);
    renderSources();
    log("Pasted image added to Sources.");
  } catch (err) { log(String(err), "err"); }
});

const IMAGE = /\.(png|jpe?g|webp)$/i;
listen("tauri://drag-drop", ({ payload }) => {
  const paths = (payload.paths || []).filter(p => IMAGE.test(p));
  if (paths.length) importSources(paths);
});

// ---------- events from the core ----------
listen("hw", ({ payload }) => {
  state.hw = payload.hw;
  state.resident = payload.resident;
  state.idleLeft = payload.idleLeft;
  renderMem();
  renderStatus();
});

listen("engine", ({ payload: p }) => {
  state.engine[p.model] = p.state;
  if (p.state === "warm" || p.state === "busy") state.resident = p.model;
  if (p.state === "cold" && state.resident === p.model) state.resident = null;
  if (p.state === "loading") log(`Loading ${modelName(p.model)}…`);
  if (p.state === "warm") log(state.run ? `${modelName(p.model)} is loaded. Developing…` : `${modelName(p.model)} is loaded.`);
  if (p.message) log(p.message, "err");
  renderTop();
  renderPreflight();
  if (state.run) renderCanvas();
});

listen("progress", ({ payload: p }) => {
  if (!ours(p.batch)) return;
  Object.assign(state.run, { batch: p.batch, index: p.index, step: p.step, steps: p.steps, preview: p.preview, secPerStep: p.secPerStep });
  renderFrames();
  renderCanvas();
});

listen("frame", ({ payload: f }) => {
  state.frames.push(f);
  if (ours(f.batch)) {
    Object.assign(state.run, { batch: f.batch, index: f.index + 1, step: 0, preview: null });
    if (state.follow) state.selected = f.id;
    // The last frame of the batch stays on the canvas instead of an empty "next frame" view.
    if (f.index + 1 >= state.run.count) state.follow = false;
  }
  renderFrames();
  renderCanvas();
});

listen("batch", ({ payload: p }) => {
  if (!ours(p.batch)) return;
  const run = state.run;
  state.run = null;
  state.follow = false;
  if (p.state === "failed") {
    state.error = [p.error?.message, p.error?.fix].filter(Boolean).join(" ");
    log("The batch stopped.", "err");
  } else if (p.state === "cancelled") {
    log("Cancelled. The model is still loaded.");
  } else {
    const n = state.frames.filter(f => f.batch === p.batch).length;
    log(`${plural(n, "frame")} in ${((Date.now() - run.started) / 1000).toFixed(0)} s. Ctrl S saves the selected one.`);
  }
  render();
});

listen("install", ({ payload: p }) => {
  if (p.phase === "done") return;
  state.installing[p.id] = p;
  renderModels();
});

listen("log", ({ payload }) => log(payload));
listen("caelestia", ({ payload }) => { caelestia = payload; if (state.settings?.theme === "Caelestia") applyTheme(); });
listen("fallback", ({ payload }) => log(`Running on the CPU: this PyTorch can't use the ${payload.fallback?.wanted} GPU.`, "warn"));

window.__TAURI__.window.getCurrentWindow().onCloseRequested(e => {
  const n = unsaved();
  if (state.settings?.warnUnsaved && n && !confirm(`${plural(n, "frame")} ${n > 1 ? "are" : "is"} not saved and will be lost. Close Latent anyway?`)) e.preventDefault();
});

// ---------- start ----------
(async () => {
  const b = await invoke("bootstrap");
  Object.assign(state, { settings: b.settings, hw: b.hw, device: b.device, models: b.models, sources: b.sources, frames: b.frames, resident: b.resident, version: b.version, logs: b.logs, platform: b.platform });
  caelestia = b.caelestia || null;
  const list = installed();
  state.model = list.some(m => m.id === b.settings.lastModel) ? b.settings.lastModel : list[0]?.id ?? "";
  applyTheme();
  render();
  applyPromptSize();
  if (state.settings.preload && state.model && !state.resident) {
    invoke("preload", { model: state.model, variant: null }).catch(e => log(String(e), "err"));
  }
  log(list.length ? "Ready." : "Download a model to start.");
  $("#prompt").focus();
  checkUpdate(true);
  setTimeout(() => $("#splash").classList.add("out"), 450);
  setTimeout(() => $("#splash").remove(), 900);
})();

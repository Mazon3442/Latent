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

let saveTimer;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => invoke("save_settings", { settings: state.settings }).catch(e => log(String(e), "err")), 300);
}

function applyTheme() {
  const t = state.settings.theme;
  const light = t === "Light" || (t === "System" && matchMedia("(prefers-color-scheme: light)").matches);
  document.documentElement.dataset.theme = light ? "light" : "dark";
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", applyTheme);

// ---------- top bar: the memory bar is the model picker ----------
function engineState(id) {
  if (state.resident === id) return state.engine[id] === "busy" ? "busy" : "warm";
  return state.engine[id] === "loading" ? "loading" : "cold";
}

function renderTop() {
  const list = installed();
  $("#modelBtns").innerHTML = list.map(m => {
    const s = engineState(m.id);
    const label = { warm: "loaded", busy: "developing", loading: "loading", cold: "not loaded" }[s];
    return `<button class="model-btn" data-id="${m.id}" aria-pressed="${m.id === state.model}">
      <b>${esc(m.name)}</b><span>${gb(variant(m).vram)} GB</span><span class="state" data-s="${s}">${label}</span></button>`;
  }).join("") + `<button class="model-btn" id="modelsBtn">${list.length ? "Models…" : "Get a model…"}</button>`;
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
  if (kind === "model") return `<p class="menu-title">Model</p>` + installed().map(x => {
    const v = variant(x);
    const does = x.can.edit ? "makes and edits images" : "makes new images";
    return `<button class="mi" data-pick-model="${x.id}" aria-checked="${x.id === state.model}"><b>${esc(x.name)}<em>${{ warm: "loaded", busy: "developing", loading: "loading", cold: "not loaded" }[engineState(x.id)]}</em></b>
      <span>${does} · about ${gb(v.vram)} GB · ${x.options.steps.default} steps</span></button>`;
  }).join("") + `<div class="menu-sep"></div><button class="mi" data-open-models><b>More models…</b><span>Download or remove models</span></button>`;
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
  const pct = ((steps - r.min) / (r.max - r.min)) * 100;
  $("#controlsBody").innerHTML = `
    <div class="ctl">
      <div class="ctl-top"><b id="ctl-steps">Steps</b>
        ${state.steps !== null ? `<button class="ctl-reset" data-reset="steps" title="Back to ${r.default}">Reset</button>` : ""}
        <span class="ctl-val ${state.steps === null ? "def" : ""}">${steps}</span></div>
      <input class="slider" type="range" id="steps" min="${r.min}" max="${r.max}" step="1" value="${steps}" aria-labelledby="ctl-steps" style="--pct:${pct}%">
      <p>Passes that refine the image. ${m.name} is tuned for ${r.default}; more is slower and rarely better.</p>
    </div>
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
  return `${failed}<div class="row"><button class="btn small" data-install="${m.id}">${failed ? "Try again" : "Download"}</button></div>`;
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
    state.installing[id] = { phase: "failed", label: String(e) };
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
function openDialog(id) {
  closeMenu();
  $(id).classList.add("open");
  $(id).querySelector("[data-close]").focus();
}
function closeDialogs() {
  const open = document.querySelector(".overlay.open");
  open?.classList.remove("open");
  return !!open;
}

function renderModels() {
  if ($("#modelsDlg").classList.contains("open")) {
    $("#modelsBody").innerHTML = state.models.map(m => {
      const v = variant(m);
      const fit = { yes: "Fits this PC", tight: "Fits, just", no: "Doesn't fit this PC" }[v.fits];
      const does = m.can.edit ? "Makes and edits images" : "Makes new images";
      const action = m.installed
        ? `<div class="row"><span class="meta">Installed.</span><span class="grow"></span><button class="quiet danger" data-remove="${m.id}">Remove</button></div>`
        : installRow(m);
      return `<div class="model-card"><div class="title"><b>${esc(m.name)}</b><span class="fit-${v.fits}">${fit}</span></div>
        <div class="facts">${does} in ${m.options.steps.default} steps. ${gb(v.disk)} GB download · about ${gb(v.vram)} GB VRAM (${esc(v.label)}) ·
        ${esc(m.license)}${m.commercial ? "" : ", non-commercial"}</div>${action}</div>`;
    }).join("") + `<p class="meta">Models live in ${esc(state.settings.modelsDir)}. Editing models (FLUX.2 klein) and smaller ones for other PCs come in the next versions.</p>`;
  }
  if (!installed().length) renderCanvas();
}

const SELECTS = {
  theme: ["Dark", "Light", "System"],
  idle: ["5 minutes", "10 minutes", "30 minutes", "Never"],
  device: ["Automatic", "GPU", "CPU"],
};

function renderSettings() {
  const s = state.settings;
  const select = key => `<select data-set="${key}">${SELECTS[key].map(o => `<option${o === s[key] ? " selected" : ""}>${o}</option>`).join("")}</select>`;
  const folder = (key, title) => `<div class="path"><input readonly value="${esc(s[key])}"><button class="btn small" data-folder="${key}" data-title="${title}">Change…</button></div>`;
  $("#settingsBody").innerHTML = `
    <div class="set-group">Appearance</div>
    <label class="setting"><span>Theme</span>${select("theme")}</label>
    <div class="set-group">Saving</div>
    <div class="setting"><span>Save folder</span>${folder("saveDir", "Save frames to")}</div>
    <label class="setting"><span>Warn about unsaved frames when closing</span><input type="checkbox" data-set="warnUnsaved"${s.warnUnsaved ? " checked" : ""} style="width:auto"></label>
    <div class="set-group">Models and memory</div>
    <div class="setting"><span>Models folder</span>${folder("modelsDir", "Keep models in")}</div>
    <label class="setting"><span>Unload an idle model after</span>${select("idle")}</label>
    <label class="setting"><span>Run models on</span>${select("device")}</label>
    <label class="setting"><span>Hugging Face token</span><input type="password" data-set="hfToken" value="${esc(s.hfToken)}" placeholder="Only for gated models" autocomplete="off"></label>
    <p class="meta">Latent ${esc(state.version)} · ${esc(state.hw?.gpu || "no GPU found")}${state.hw?.note ? " · " + esc(state.hw.note) : ""}</p>`;
}

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
  if (t.dataset.openModels !== undefined) { openDialog("#modelsDlg"); return renderModels(); }
  if (t.dataset.id && t.classList.contains("model-btn")) return selectModel(t.dataset.id);
  if (t.id === "modelsBtn") { openDialog("#modelsDlg"); return renderModels(); }
  if (t.id === "settingsBtn") { renderSettings(); return openDialog("#settingsDlg"); }
  if (t.dataset.close !== undefined) return closeDialogs();
  if (t.dataset.install) return installModel(t.dataset.install);
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
});

document.addEventListener("input", e => {
  if (e.target.dataset.set === "hfToken") { state.settings.hfToken = e.target.value; saveSettings(); }
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
listen("fallback", ({ payload }) => log(`Running on the CPU: this PyTorch can't use the ${payload.fallback?.wanted} GPU.`, "warn"));

window.__TAURI__.window.getCurrentWindow().onCloseRequested(e => {
  const n = unsaved();
  if (state.settings?.warnUnsaved && n && !confirm(`${plural(n, "frame")} ${n > 1 ? "are" : "is"} not saved and will be lost. Close Latent anyway?`)) e.preventDefault();
});

// ---------- start ----------
(async () => {
  const b = await invoke("bootstrap");
  Object.assign(state, { settings: b.settings, hw: b.hw, device: b.device, models: b.models, sources: b.sources, frames: b.frames, resident: b.resident, version: b.version });
  const list = installed();
  state.model = list.some(m => m.id === b.settings.lastModel) ? b.settings.lastModel : list[0]?.id ?? "";
  applyTheme();
  render();
  autosize();
  log(list.length ? "Ready." : "Download a model to start.");
  $("#prompt").focus();
  setTimeout(() => $("#splash").classList.add("out"), 450);
  setTimeout(() => $("#splash").remove(), 900);
})();

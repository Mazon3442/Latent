// Latent UI — frontend JavaScript
// Communicates with Tauri backend via invoke() and listen().
// Implements all Tauri commands/events from docs/PLAN.md §5.

(() => {
  "use strict";

  // ── State ──────────────────────────────────────────────────────
  const state = {
    settings: null,
    hw: null,
    models: [],
    sources: [],
    frames: [],
    batches: [],
    unsaved: 0,
    resident: null,
    selectedSourceId: null,
    selectedFrameId: null,
    activeSize: null,
    prompt: "",
    negative: "",
    steps: 8,
    guidance: 4.5,
    sampler: null,
    strength: 0.55,
    count: 1,
    seed: null,
    generating: false,
    currentBatch: null,
    currentSeed: null,
    theme: "dark",
  };

  // ── DOM refs ───────────────────────────────────────────────────
  const dom = {
    // Top bar
    memoryFill: document.getElementById("memoryFill"),
    memoryLabel: document.getElementById("memoryLabel"),
    topbarModels: document.getElementById("topbarModels"),
    btnModels: document.getElementById("btnModels"),
    btnSettings: document.getElementById("btnSettings"),
    // Sources panel
    sourcesList: document.getElementById("sourcesList"),
    sourcesCount: document.getElementById("sourcesCount"),
    dropZone: document.getElementById("dropZone"),
    // Canvas
    canvas: document.getElementById("canvas"),
    canvasPlaceholder: document.getElementById("canvasPlaceholder"),
    canvasContent: document.getElementById("canvasContent"),
    canvasImage: document.getElementById("canvasImage"),
    canvasOverlay: document.getElementById("canvasOverlay"),
    canvasStepInfo: document.getElementById("canvasStepInfo"),
    stepText: document.getElementById("stepText"),
    compareSlider: document.getElementById("compareSlider"),
    // Prompt area
    promptInput: document.getElementById("promptInput"),
    promptMeta: document.getElementById("promptMeta"),
    sourceLabel: document.getElementById("sourceLabel"),
    sizeChips: document.getElementById("sizeChips"),
    sizeDetail: document.getElementById("sizeDetail"),
    batchInfo: document.getElementById("batchInfo"),
    seedDisplay: document.getElementById("seedDisplay"),
    btnSave: document.getElementById("btnSave"),
    btnUseSource: document.getElementById("btnUseSource"),
    optionsDrawer: document.getElementById("optionsDrawer"),
    btnOptions: document.getElementById("btnOptions"),
    optionSteps: document.getElementById("optionSteps"),
    optionGuidance: document.getElementById("optionGuidance"),
    optionSampler: document.getElementById("optionSampler"),
    optionStrength: document.getElementById("optionStrength"),
    optionNegative: document.getElementById("optionNegative"),
    inputSteps: document.getElementById("inputSteps"),
    inputGuidance: document.getElementById("inputGuidance"),
    inputSampler: document.getElementById("inputSampler"),
    inputStrength: document.getElementById("inputStrength"),
    strengthValue: document.getElementById("strengthValue"),
    inputNegative: document.getElementById("inputNegative"),
    btnGenerate: document.getElementById("btnGenerate"),
    btnCancel: document.getElementById("btnCancel"),
    // Frames panel
    framesList: document.getElementById("framesList"),
    framesCount: document.getElementById("framesCount"),
    unsavedBadge: document.getElementById("unsavedBadge"),
    framesHint: document.getElementById("framesHint"),
    // Status bar
    statusStatus: document.getElementById("statusStatus"),
    statusDetail: document.getElementById("statusDetail"),
    statusUnload: document.getElementById("statusUnload"),
    // Drawers
    settingsOverlay: document.getElementById("settingsOverlay"),
    settingsDrawer: document.getElementById("settingsDrawer"),
    btnCloseSettings: document.getElementById("btnCloseSettings"),
    modelsOverlay: document.getElementById("modelsOverlay"),
    modelsDrawer: document.getElementById("modelsDrawer"),
    btnCloseModels: document.getElementById("btnCloseModels"),
    modelsBody: document.getElementById("modelsBody"),
    // Settings fields
    settingTheme: document.getElementById("settingTheme"),
    settingSaveFolder: document.getElementById("settingSaveFolder"),
    settingFormat: document.getElementById("settingFormat"),
    settingWarnClose: document.getElementById("settingWarnClose"),
    settingModelsFolder: document.getElementById("settingModelsFolder"),
    settingIdleUnload: document.getElementById("settingIdleUnload"),
    settingWarmStart: document.getElementById("settingWarmStart"),
    settingComputeTarget: document.getElementById("settingComputeTarget"),
    settingHfToken: document.getElementById("settingHfToken"),
    settingBatchSize: document.getElementById("settingBatchSize"),
    settingDefaultAspect: document.getElementById("settingDefaultAspect"),
  };

  // ── Bootstrap ──────────────────────────────────────────────────

  async function bootstrap() {
    try {
      const data = await invoke("bootstrap");
      state.settings = data.settings;
      state.hw = data.hw;
      state.models = data.models;
      state.sources = data.sources || [];
      state.frames = data.frames || [];
      state.batches = data.batches || [];
      state.unsaved = data.unsaved || 0;
      state.resident = data.resident || null;
      state.theme = state.settings.theme === "system"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
        : state.settings.theme;

      applyTheme();
      renderTopbar();
      renderMemoryBar();
      renderSources();
      renderFrames();
      renderStatusbar();
      populateModelsDrawer();
      populateSettings();
      updatePromptState();
    } catch (err) {
      console.error("Bootstrap failed:", err);
      setStatus("Error: " + err, "error");
    }
  }

  // ── Theme ──────────────────────────────────────────────────────

  function applyTheme() {
    const effective = state.settings.theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
      : state.settings.theme;
    document.documentElement.setAttribute("data-theme", effective);
    dom.settingTheme.value = state.settings.theme;
  }

  // ── Memory bar ─────────────────────────────────────────────────

  function renderMemoryBar() {
    const gpu = state.hw?.devices?.find((d) => d.vram_total_mb);
    if (!gpu || !gpu.vram_total_mb) {
      dom.memoryLabel.textContent = "no GPU";
      dom.memoryFill.style.width = "0%";
      dom.memoryFill.classList.remove("warning");
      return;
    }
    const used = gpu.vram_used_mb || 0;
    const total = gpu.vram_total_mb;
    const pct = Math.round((used / total) * 100);
    dom.memoryFill.style.width = pct + "%";
    dom.memoryFill.classList.toggle("warning", pct > 85);
    dom.memoryLabel.textContent = `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GB`;
  }

  // ── Top bar ────────────────────────────────────────────────────

  function renderTopbar() {
    dom.topbarModels.innerHTML = "";
    state.models.forEach((model) => {
      const pill = document.createElement("div");
      pill.className = "model-pill";
      if (state.resident?.model === model.id) {
        pill.classList.add("active");
        if (state.resident?.state === "warm") pill.classList.add("warm");
        if (state.resident?.state === "busy") pill.classList.add("busy");
        if (state.resident?.state === "cold") pill.classList.add("cold");
      }
      const label = document.createElement("span");
      label.className = "pill-name";
      label.textContent = model.name;
      const stateLabel = document.createElement("span");
      stateLabel.className = "pill-state";
      if (model.installed) {
        stateLabel.textContent = "✓";
      } else {
        stateLabel.textContent = "–";
      }
      pill.appendChild(label);
      pill.appendChild(stateLabel);
      pill.addEventListener("click", () => openModelsDrawer());
      dom.topbarModels.appendChild(pill);
    });
  }

  // ── Sources panel ──────────────────────────────────────────────

  function renderSources() {
    dom.sourcesList.innerHTML = "";
    dom.sourcesCount.textContent = state.sources.length;

    state.sources.forEach((src) => {
      const tile = document.createElement("div");
      tile.className = "source-tile" + (state.selectedSourceId === src.id ? " selected" : "");
      tile.dataset.id = src.id;

      const thumb = document.createElement("img");
      thumb.className = "source-thumb";
      thumb.src = src.thumb;
      thumb.alt = "";
      tile.appendChild(thumb);

      const info = document.createElement("div");
      info.className = "source-info";
      const name = document.createElement("span");
      name.className = "source-name";
      name.textContent = src.name;
      info.appendChild(name);

      if (src.width && src.height) {
        const meta = document.createElement("span");
        meta.className = "source-meta";
        meta.textContent = `${src.width}×${src.height}`;
        info.appendChild(meta);
      }
      tile.appendChild(info);

      const del = document.createElement("button");
      del.className = "source-delete";
      del.textContent = "×";
      del.title = "Remove";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSource(src.id);
      });
      tile.appendChild(del);

      tile.addEventListener("click", () => {
        state.selectedSourceId = src.id;
        renderSources();
        updatePromptState();
      });

      dom.sourcesList.appendChild(tile);
    });

    // Drop zone styling
    dom.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dom.dropZone.classList.add("drag-over");
    });
    dom.dropZone.addEventListener("dragleave", () => {
      dom.dropZone.classList.remove("drag-over");
    });
    dom.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dom.dropZone.classList.remove("drag-over");
      handleDropFiles(e.dataTransfer?.files);
    });
  }

  async function handleDropFiles(files) {
    if (!files || files.length === 0) return;
    const paths = Array.from(files).map((f) => f.path || f.webkitRelativePath || f.name);
    const result = await invoke("import_sources", { paths });
    state.sources.push(...result);
    renderSources();
    updatePromptState();
  }

  async function deleteSource(id) {
    await invoke("delete_source", { id });
    state.sources = state.sources.filter((s) => s.id !== id);
    if (state.selectedSourceId === id) state.selectedSourceId = null;
    renderSources();
    updatePromptState();
  }

  // ── Canvas ─────────────────────────────────────────────────────

  function showCanvasImage(path) {
    dom.canvasPlaceholder.classList.add("hidden");
    dom.canvasContent.classList.remove("hidden");
    dom.canvasImage.src = path;
    dom.canvasImage.classList.remove("hidden");
    dom.canvasOverlay.classList.add("hidden");
    dom.compareSlider.classList.add("hidden");
    dom.btnSave.style.display = "inline";
    dom.btnUseSource.style.display = "inline";
  }

  function hideCanvasImage() {
    dom.canvasPlaceholder.classList.remove("hidden");
    dom.canvasContent.classList.add("hidden");
    dom.btnSave.style.display = "none";
    dom.btnUseSource.style.display = "none";
  }

  function showStepProgress(step, steps, preview) {
    dom.canvasOverlay.classList.remove("hidden");
    dom.stepText.textContent = `${step} / ${steps} steps`;
    if (preview) {
      dom.canvasImage.src = preview;
    }
  }

  function hideStepProgress() {
    dom.canvasOverlay.classList.add("hidden");
  }

  // ── Size chips ─────────────────────────────────────────────────

  function renderSizeChips(model) {
    dom.sizeChips.innerHTML = "";
    if (!model || !model.sizes) return;
    const sizes = model.sizes;
    const unique = [...new Set(sizes.map(([w, h]) => `${w}×${h}`))];
    unique.forEach((label) => {
      const chip = document.createElement("button");
      chip.className = "size-chip" + (state.activeSize === label ? " active" : "");
      chip.textContent = label;
      chip.addEventListener("click", () => {
        state.activeSize = label;
        renderSizeChips(model);
        updatePromptState();
      });
      dom.sizeChips.appendChild(chip);
    });
  }

  function updateSizeDetail() {
    if (state.activeSize) {
      const [w, h] = state.activeSize.split("×").map(Number);
      dom.sizeDetail.textContent = `${w} × ${h}`;
    } else {
      dom.sizeDetail.textContent = "";
    }
  }

  // ── Seed ───────────────────────────────────────────────────────

  function updateSeedDisplay() {
    if (state.seed !== null) {
      dom.seedDisplay.textContent = `seed: ${state.seed}`;
    } else {
      dom.seedDisplay.textContent = "seed: random";
    }
  }

  // ── Source label ───────────────────────────────────────────────

  function updateSourceLabel() {
    if (state.selectedSourceId) {
      const src = state.sources.find((s) => s.id === state.selectedSourceId);
      if (src) {
        dom.sourceLabel.innerHTML = `📎 ${src.name}`;
        const remove = document.createElement("span");
        remove.className = "remove-source";
        remove.textContent = " ×";
        remove.title = "Remove source";
        remove.addEventListener("click", () => deleteSource(src.id));
        dom.sourceLabel.appendChild(remove);
        dom.sourceLabel.style.display = "inline-flex";
      }
    } else {
      dom.sourceLabel.style.display = "none";
    }
  }

  // ── Options drawer ─────────────────────────────────────────────

  function toggleOptions() {
    dom.optionsDrawer.classList.toggle("hidden");
  }

  function updateOptionsForModel(model) {
    if (!model) return;

    const opt = model.options;

    // Steps
    if (opt.steps) {
      dom.optionSteps.classList.remove("hidden");
      dom.inputSteps.min = opt.steps.min;
      dom.inputSteps.max = opt.steps.max;
      dom.inputSteps.value = state.steps || opt.steps.default;
    } else {
      dom.optionSteps.classList.add("hidden");
    }

    // Guidance
    if (opt.guidance) {
      dom.optionGuidance.classList.remove("hidden");
      dom.inputGuidance.min = opt.guidance.min;
      dom.inputGuidance.max = opt.guidance.max;
      dom.inputGuidance.step = 0.5;
      dom.inputGuidance.value = state.guidance || opt.guidance.default;
    } else {
      dom.optionGuidance.classList.add("hidden");
    }

    // Sampler
    if (opt.sampler) {
      dom.optionSampler.classList.remove("hidden");
      dom.inputSampler.innerHTML = "";
      opt.sampler.forEach((s) => {
        const optEl = document.createElement("option");
        optEl.value = s;
        optEl.textContent = s;
        dom.inputSampler.appendChild(optEl);
      });
    } else {
      dom.optionSampler.classList.add("hidden");
    }

    // Strength
    if (opt.strength) {
      dom.optionStrength.classList.remove("hidden");
      dom.inputStrength.min = opt.strength.min;
      dom.inputStrength.max = opt.strength.max;
      dom.inputStrength.step = opt.strength.max < 10 ? 0.05 : 1;
      dom.inputStrength.value = state.strength || opt.strength.default;
      dom.strengthValue.textContent = state.strength || opt.strength.default;
    } else {
      dom.optionStrength.classList.add("hidden");
    }

    // Negative
    if (model.can.negative) {
      dom.optionNegative.classList.remove("hidden");
    } else {
      dom.optionNegative.classList.add("hidden");
    }
  }

  // ── Prompt state ───────────────────────────────────────────────

  function updatePromptState() {
    const model = getResidentModel();
    updateSizeChipsForModel(model);
    updateSourceLabel();
    updateSizeDetail();
    updateSeedDisplay();

    // Enable/disable generate
    const canGenerate = model && !state.generating && state.prompt.trim().length > 0;
    const canEdit = model && model.can.edit && state.selectedSourceId && !state.generating;
    dom.btnGenerate.disabled = !canGenerate && !canEdit;

    // Show/hide use as source button for frames
    if (state.selectedFrameId) {
      dom.btnUseSource.style.display = "inline";
    }
  }

  function updateSizeChipsForModel(model) {
    if (model) {
      renderSizeChips(model);
      updateOptionsForModel(model);
    } else {
      dom.sizeChips.innerHTML = "";
      dom.optionSteps.classList.add("hidden");
      dom.optionGuidance.classList.add("hidden");
      dom.optionSampler.classList.add("hidden");
      dom.optionStrength.classList.add("hidden");
      dom.optionNegative.classList.add("hidden");
    }
  }

  function getResidentModel() {
    if (!state.resident) return null;
    return state.models.find((m) => m.id === state.resident.model) || null;
  }

  // ── Generate ───────────────────────────────────────────────────

  async function generate() {
    const model = getResidentModel();
    if (!model) {
      setStatus("No model loaded", "error");
      return;
    }

    const activeSize = state.activeSize || (model.sizes?.[0] || [512, 512]);
    const [width, height] = activeSize.split("×").map(Number);

    const request = {
      model: model.id,
      variant: state.resident?.variant || model.defaultVariant,
      prompt: state.prompt.trim(),
      negative: state.negative.trim() || undefined,
      sources: state.selectedSourceId ? [state.selectedSourceId] : [],
      width,
      height,
      count: state.count,
      seed: state.seed,
      options: {
        steps: parseInt(dom.inputSteps.value) || model.options.steps?.default || 8,
        guidance: parseFloat(dom.inputGuidance.value) || model.options.guidance?.default || 4.5,
      },
    };

    if (dom.inputSampler.value && model.options.sampler?.includes(dom.inputSampler.value)) {
      request.options.sampler = dom.inputSampler.value;
    }
    if (model.options.strength) {
      request.options.strength = parseFloat(dom.inputStrength.value) || model.options.strength.default;
    }

    try {
      state.generating = true;
      state.currentBatch = null;
      dom.btnGenerate.disabled = true;
      dom.btnCancel.classList.remove("hidden");
      setStatus("Generating...", "developing");
      hideStepProgress();

      const result = await invoke("generate", { request });
      state.currentBatch = result.batch;
      state.batchInfo.textContent = `batch ${result.batch.slice(0, 8)}`;
    } catch (err) {
      console.error("Generate error:", err);
      setStatus("Error: " + err, "error");
      state.generating = false;
      dom.btnCancel.classList.add("hidden");
    }
  }

  async function cancelGeneration() {
    if (state.currentBatch) {
      try {
        await invoke("cancel", { batch: state.currentBatch });
      } catch (e) {
        console.error("Cancel error:", e);
      }
    }
    state.generating = false;
    dom.btnCancel.classList.add("hidden");
    dom.btnGenerate.disabled = !state.prompt.trim();
    setStatus("Cancelled", "");
  }

  // ── Frames ─────────────────────────────────────────────────────

  function renderFrames() {
    dom.framesList.innerHTML = "";
    dom.framesCount.textContent = state.frames.length;
    dom.unsavedBadge.style.display = state.unsaved > 0 ? "inline" : "none";
    dom.unsavedBadge.textContent = state.unsaved;
    dom.framesHint.style.display = state.frames.length === 0 ? "block" : "none";

    if (state.frames.length === 0) return;

    // Group by batch
    const batchMap = new Map();
    state.frames.forEach((frame) => {
      if (!batchMap.has(frame.batch)) {
        batchMap.set(frame.batch, []);
      }
      batchMap.get(frame.batch).push(frame);
    });

    // Sort batches newest first
    const batches = [...batchMap.entries()].sort((a, b) => {
      const aLatest = a[1].at(-1)?.created_at || "";
      const bLatest = b[1].at(-1)?.created_at || "";
      return bLatest.localeCompare(aLatest);
    });

    batches.forEach(([batchId, frames]) => {
      const batchEl = document.createElement("div");
      batchEl.className = "frames-batch";

      const header = document.createElement("div");
      header.className = "batch-header";

      const modelInfo = frames[0]?.model;
      const modelName = state.models.find((m) => m.id === modelInfo)?.name || modelInfo || batchId.slice(0, 8);
      header.innerHTML = `<span class="batch-model">${modelName}</span>`;

      const batchState = getBatchState(batchId);
      if (batchState) {
        const status = document.createElement("span");
        status.className = `batch-status ${batchState.state}`;
        if (batchState.state === "running") {
          status.textContent = "running...";
          status.classList.add("running");
        } else if (batchState.state === "done") {
          status.textContent = "done";
        } else if (batchState.state === "failed") {
          status.textContent = `failed: ${batchState.message?.slice(0, 30)}`;
          status.classList.add("failed");
        } else if (batchState.state === "cancelled") {
          status.textContent = "cancelled";
          status.classList.add("cancelled");
        }
        header.appendChild(status);
      } else {
        header.innerHTML += `<span class="batch-model">${frames.length} frame${frames.length > 1 ? "s" : ""}</span>`;
      }

      batchEl.appendChild(header);

      const grid = document.createElement("div");
      grid.className = "frames-grid";

      frames.forEach((frame) => {
        const tile = document.createElement("div");
        tile.className = "frame-tile" + (state.selectedFrameId === frame.id ? " selected" : "");
        tile.dataset.frameId = frame.id;

        if (state.currentBatch === frame.batch) {
          tile.classList.add("developing");
        }

        const img = document.createElement("img");
        img.src = frame.thumb || frame.path;
        img.alt = "";
        tile.appendChild(img);

        // Step badge for developing frames
        if (state.currentBatch === frame.batch) {
          const badge = document.createElement("span");
          badge.className = "frame-step-badge";
          badge.textContent = "…";
          tile.appendChild(badge);
        }

        // Failed overlay
        if (getBatchState(frame.batch)?.state === "failed") {
          tile.classList.add("failed");
        }

        tile.addEventListener("click", () => selectFrame(frame));
        grid.appendChild(tile);
      });

      batchEl.appendChild(grid);
      dom.framesList.appendChild(batchEl);
    });
  }

  function getBatchState(batchId) {
    // Check pending batches (still running)
    // This would need to be tracked from events
    return null;
  }

  function selectFrame(frame) {
    state.selectedFrameId = frame.id;
    showCanvasImage(frame.path);
    renderFrames();
    updatePromptState();
  }

  // ── Status bar ─────────────────────────────────────────────────

  function renderStatusbar() {
    updateStatusDisplay();
  }

  function setStatus(msg, type) {
    dom.statusStatus.textContent = msg;
    dom.statusStatus.className = "status-status" + (type ? " " + type : "");
  }

  function updateStatusDisplay() {
    if (state.resident) {
      dom.statusStatus.textContent = state.resident.state === "warm" ? "Ready" : state.resident.state;
      dom.statusUnload.textContent = `${state.resident.model} (${state.resident.variant})`;
    } else {
      dom.statusStatus.textContent = "Ready";
      dom.statusUnload.textContent = "";
    }
  }

  // ── Save ───────────────────────────────────────────────────────

  async function saveFrame() {
    if (!state.selectedFrameId) return;
    try {
      const path = await invoke("save_frame", { frame_id: state.selectedFrameId });
      setStatus(`Saved to ${path}`, "");
    } catch (err) {
      setStatus(`Save error: ${err}`, "error");
    }
  }

  async function saveFrameAs() {
    if (!state.selectedFrameId) return;
    try {
      const result = await invoke("save_frame_as", { frame_id: state.selectedFrameId });
      if (result) setStatus(`Saved to ${result}`, "");
    } catch (err) {
      setStatus(`Save error: ${err}`, "error");
    }
  }

  async function useAsSource() {
    if (!state.selectedFrameId) return;
    try {
      await invoke("frame_to_source", { frame: state.selectedFrameId });
      setStatus("Frame added as source", "");
    } catch (err) {
      setStatus(`Error: ${err}`, "error");
    }
  }

  // ── Models drawer ──────────────────────────────────────────────

  function openModelsDrawer() {
    dom.modelsOverlay.classList.remove("hidden");
  }

  function closeModelsDrawer() {
    dom.modelsOverlay.classList.add("hidden");
  }

  function populateModelsDrawer() {
    dom.modelsBody.innerHTML = "";

    state.models.forEach((model) => {
      const card = document.createElement("div");
      card.className = "model-card";

      const header = document.createElement("div");
      header.className = "model-card-header";

      const name = document.createElement("span");
      name.className = "model-card-name";
      name.textContent = model.name;
      header.appendChild(name);

      const lic = document.createElement("span");
      lic.className = "model-card-lic";
      lic.textContent = model.license;
      header.appendChild(lic);

      card.appendChild(header);

      const desc = document.createElement("div");
      desc.className = "model-card-desc";
      desc.textContent = model.desc;
      card.appendChild(desc);

      const caps = document.createElement("div");
      caps.className = "model-card-capabilities";
      const capLabels = {
        generate: "Generate",
        edit: "Edit",
        inpaint: "Inpaint",
        multi_ref: "Multi-ref",
        negative: "Negative prompt",
      };
      Object.entries(capLabels).forEach(([key, label]) => {
        if (model.can[key]) {
          const badge = document.createElement("span");
          badge.className = "cap-badge";
          badge.textContent = label;
          caps.appendChild(badge);
        }
      });
      card.appendChild(caps);

      // Variants
      const variantsEl = document.createElement("div");
      variantsEl.className = "model-variants";

      model.variants.forEach((variant) => {
        const row = document.createElement("div");
        row.className = "variant-row";

        const label = document.createElement("span");
        label.className = "variant-label";
        label.textContent = variant.label;
        row.appendChild(label);

        const vram = document.createElement("span");
        vram.className = "variant-vram";
        vram.textContent = `${(variant.vram_mb / 1024).toFixed(1)} GB VRAM`;
        row.appendChild(vram);

        const fit = document.createElement("span");
        fit.className = `variant-fit ${variant.fits}`;
        fit.textContent = variant.fits === "yes" ? "✓ fits" : variant.fits === "tight" ? "~ tight" : "✕";
        row.appendChild(fit);

        const actions = document.createElement("span");
        actions.className = "variant-actions";

        const isInstalled = model.installed;
        const isResident = state.resident?.model === model.id && state.resident?.variant === variant.id;

        if (isResident) {
          const btn = document.createElement("button");
          btn.className = "btn btn-secondary";
          btn.style.fontSize = "var(--font-size-xs)";
          btn.textContent = "Resident";
          btn.disabled = true;
          actions.appendChild(btn);
        } else if (isInstalled) {
          const btn = document.createElement("button");
          btn.className = "btn btn-secondary";
          btn.style.fontSize = "var(--font-size-xs)";
          btn.textContent = "Load";
          btn.addEventListener("click", () => preloadModel(model.id, variant.id));
          actions.appendChild(btn);

          const unloadBtn = document.createElement("button");
          unloadBtn.className = "btn btn-secondary";
          unloadBtn.style.fontSize = "var(--font-size-xs)";
          unloadBtn.textContent = "Unload";
          unloadBtn.addEventListener("click", () => unloadModel());
          actions.appendChild(unloadBtn);
        } else {
          const btn = document.createElement("button");
          btn.className = "btn btn-primary";
          btn.style.fontSize = "var(--font-size-xs)";
          btn.textContent = `Install ${(variant.disk_mb / 1024).toFixed(0)} GB`;
          btn.addEventListener("click", () => installModel(model.id, variant.id));
          actions.appendChild(btn);
        }

        row.appendChild(actions);
        variantsEl.appendChild(row);
      });

      card.appendChild(variantsEl);
      dom.modelsBody.appendChild(card);
    });
  }

  async function preloadModel(modelId, variantId) {
    try {
      await invoke("preload", {
        model: { id: modelId },
        variant: variantId,
      });
      const m = state.models.find((m) => m.id === modelId);
      if (m) renderSizeChipsForModel(m);
      renderTopbar();
      updateStatusDisplay();
      populateModelsDrawer();
    } catch (err) {
      setStatus(`Preload error: ${err}`, "error");
    }
  }

  async function unloadModel() {
    try {
      await invoke("unload");
      state.resident = null;
      state.currentBatch = null;
      renderTopbar();
      updateStatusDisplay();
      populateModelsDrawer();
      updatePromptState();
    } catch (err) {
      setStatus(`Unload error: ${err}`, "error");
    }
  }

  async function installModel(modelId, variantId) {
    try {
      await invoke("install_model", {
        model_id: modelId,
        variant: variantId,
      });
      populateModelsDrawer();
    } catch (err) {
      setStatus(`Install error: ${err}`, "error");
    }
  }

  // ── Settings drawer ────────────────────────────────────────────

  function openSettings() {
    dom.settingsOverlay.classList.remove("hidden");
  }

  function closeSettings() {
    dom.settingsOverlay.classList.add("hidden");
  }

  function populateSettings() {
    const s = state.settings;
    if (!s) return;
    dom.settingTheme.value = s.theme || "dark";
    dom.settingSaveFolder.value = s.save_folder || "";
    dom.settingFormat.value = s.save_format || "png";
    dom.settingWarnClose.checked = s.warn_close !== false;
    dom.settingModelsFolder.value = s.models_folder || "";
    dom.settingComputeTarget.value = s.compute_target || "auto";
    dom.settingHfToken.value = s.hf_token || "";
    dom.settingBatchSize.value = s.batch_size || 1;
    dom.settingDefaultAspect.value = s.default_aspect || "1:1";

    const mins = s.idle_unload_minutes;
    if (mins === "never") {
      dom.settingIdleUnload.value = "never";
    } else {
      dom.settingIdleUnload.value = String(mins || 10);
    }

    dom.settingWarmStart.checked = s.warm_start || false;
  }

  async function saveSettings() {
    const s = {
      theme: dom.settingTheme.value,
      save_folder: dom.settingSaveFolder.value,
      save_format: dom.settingFormat.value,
      warn_close: dom.settingWarnClose.checked,
      models_folder: dom.settingModelsFolder.value,
      idle_unload_minutes: dom.settingIdleUnload.value === "never"
        ? "never"
        : parseInt(dom.settingIdleUnload.value) || 10,
      warm_start: dom.settingWarmStart.checked,
      compute_target: dom.settingComputeTarget.value,
      hf_token: dom.settingHfToken.value,
      batch_size: parseInt(dom.settingBatchSize.value) || 1,
      default_aspect: dom.settingDefaultAspect.value,
    };

    try {
      await invoke("save_settings", { settings: s });
      state.settings = s;
      applyTheme();
      closeSettings();
    } catch (err) {
      setStatus(`Save settings error: ${err}`, "error");
    }
  }

  // ── Event listeners (from Tauri) ────────────────────────────────

  function setupEventListeners() {
    setupUIListeners();

    // Setup Tauri event listeners
    listen("hw", (event) => {
      state.hw = event.payload;
      renderMemoryBar();
    });

    listen("engine", (event) => {
      const { model, variant, state: engState, elapsed } = event.payload;

      if (model && variant) {
        state.resident = { model, variant, state: engState };
        renderTopbar();
        populateModelsDrawer();

        // Update status bar
        dom.statusStatus.textContent = engState === "warm" ? "Ready" : engState;
        dom.statusStatus.className = "status-status" + (engState === "warm" ? "" : " developing");
        dom.statusUnload.textContent = `${model} (${variant})`;

        // Update prompt state based on new resident model
        const m = state.models.find((m) => m.id === model);
        if (m) updateSizeChipsForModel(m);
        updatePromptState();
      } else {
        // State change without model info
        if (event.payload.state === "cold") {
          state.resident = null;
          renderTopbar();
          populateModelsDrawer();
          dom.statusUnload.textContent = "";
          updatePromptState();
        }
      }
    });

    listen("progress", (event) => {
      const { batch, index, step, steps, preview, secPerStep } = event.payload;

      if (state.currentBatch === batch) {
        showStepProgress(step, steps, preview);
        dom.statusDetail.textContent = `Step ${step}/${steps}` + (secPerStep ? ` (${secPerStep.toFixed(1)}s/step)` : "");
      }
    });

    listen("frame", (event) => {
      const frame = event.payload;

      // Add frame to state
      state.frames.unshift(frame);
      state.unsaved = state.frames.filter((f) => !f.saved).length;

      // Show in canvas if it's the current batch
      if (state.currentBatch === frame.batch) {
        showCanvasImage(frame.path);
        hideStepProgress();
      }

      renderFrames();
      updatePromptState();
    });

    listen("batch", (event) => {
      const { batch, state: batchState, error, frame_id } = event.payload;

      if (state.currentBatch === batch) {
        state.currentBatch = null;
        dom.btnCancel.classList.add("hidden");
        dom.btnGenerate.disabled = !state.prompt.trim();

        if (batchState === "done") {
          setStatus("Done", "");
        } else if (batchState === "cancelled") {
          setStatus("Cancelled", "");
        } else if (batchState === "failed") {
          setStatus(`Failed: ${error?.message || "unknown"}`, "error");
        }

        dom.statusDetail.textContent = "";
      }

      renderFrames();
    });

    listen("install", (event) => {
      const { id, phase, done, message } = event.payload;
      if (phase === "done") {
        setStatus(`Model ${id} installed`, "");
        populateModelsDrawer();
      } else if (phase === "failed") {
        setStatus(`Install failed: ${message || "unknown"}`, "error");
      } else {
        setStatus(`Installing ${id}: ${phase}`, "");
      }
    });
  }

  // ── UI listeners (separate from Tauri listeners) ───────────────

  function setupUIListeners() {
    // Generate / Cancel
    dom.btnGenerate.addEventListener("click", generate);
    dom.btnCancel.addEventListener("click", cancelGeneration);

    // Prompt input
    dom.promptInput.addEventListener("input", () => {
      state.prompt = dom.promptInput.value;
      dom.promptInput.style.height = "auto";
      dom.promptInput.style.height = dom.promptInput.scrollHeight + "px";
      updatePromptState();
    });

    dom.promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        generate();
      }
      if (e.key === "Escape" && !dom.optionsDrawer.classList.contains("hidden")) {
        toggleOptions();
      }
    });

    // Options
    dom.btnOptions.addEventListener("click", toggleOptions);
    dom.inputSteps.addEventListener("input", () => {
      state.steps = parseInt(dom.inputSteps.value);
    });
    dom.inputGuidance.addEventListener("input", () => {
      state.guidance = parseFloat(dom.inputGuidance.value);
    });
    dom.inputStrength.addEventListener("input", () => {
      state.strength = parseFloat(dom.inputStrength.value);
      dom.strengthValue.textContent = dom.inputStrength.value;
    });
    dom.inputNegative.addEventListener("input", () => {
      state.negative = dom.inputNegative.value;
    });

    // Save
    dom.btnSave.addEventListener("click", saveFrame);
    dom.btnUseSource.addEventListener("click", useAsSource);

    // Settings
    dom.btnSettings.addEventListener("click", openSettings);
    dom.btnCloseSettings.addEventListener("click", () => {
      saveSettings();
      closeSettings();
    });
    dom.settingsOverlay.addEventListener("click", (e) => {
      if (e.target === dom.settingsOverlay) {
        saveSettings();
        closeSettings();
      }
    });

    // Models
    dom.btnModels.addEventListener("click", openModelsDrawer);
    dom.btnCloseModels.addEventListener("click", closeModelsDrawer);
    dom.modelsOverlay.addEventListener("click", (e) => {
      if (e.target === dom.modelsOverlay) closeModelsDrawer();
    });

    // Paste from clipboard
    document.addEventListener("paste", (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const blob = item.getAsFile();
          const reader = new FileReader();
          reader.onload = async () => {
            const base64 = reader.result.split(",")[1];
            await invoke("paste_source", { png: base64 });
            const data = await invoke("bootstrap");
            state.sources = data.sources || [];
            renderSources();
            updatePromptState();
          };
          reader.readAsDataURL(blob);
          break;
        }
      }
    });

    // Settings controls
    dom.settingTheme.addEventListener("change", () => {
      state.settings.theme = dom.settingTheme.value;
      applyTheme();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      // Ctrl+, → settings
      if (e.key === "," && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        openSettings();
      }
      // Ctrl+M → models
      if (e.key === "m" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        openModelsDrawer();
      }
      // Escape → close drawers
      if (e.key === "Escape") {
        if (!dom.settingsOverlay.classList.contains("hidden")) {
          saveSettings();
          closeSettings();
        } else if (!dom.modelsOverlay.classList.contains("hidden")) {
          closeModelsDrawer();
        }
      }
    });
  }

  // ── Init ───────────────────────────────────────────────────────

  async function init() {
    setupEventListeners();
    await bootstrap();
  }

  // Start when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

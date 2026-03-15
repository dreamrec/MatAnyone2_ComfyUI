import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "MatAnyoneInteractiveSAM";
const STATE_WIDGET_NAME = "editor_state";
const STYLE_ID = "matanyone-editor-style";

let activeDialog = null;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .ma2-overlay {
      position: fixed;
      inset: 0;
      z-index: 100000;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(7, 11, 17, 0.72);
      backdrop-filter: blur(10px);
    }
    .ma2-shell {
      width: min(1400px, 96vw);
      height: min(900px, 94vh);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background:
        radial-gradient(circle at top left, rgba(87, 199, 133, 0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(54, 128, 255, 0.16), transparent 30%),
        linear-gradient(180deg, rgba(18, 24, 33, 0.98), rgba(12, 16, 23, 0.98));
      color: #f4f7fb;
      box-shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
      overflow: hidden;
    }
    .ma2-header,
    .ma2-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .ma2-header {
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .ma2-title {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .ma2-title h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .ma2-subtitle,
    .ma2-status {
      font-size: 12px;
      color: rgba(244, 247, 251, 0.72);
    }
    .ma2-toolbar,
    .ma2-actions {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    .ma2-body {
      min-height: 0;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr) 270px;
      gap: 14px;
    }
    .ma2-panel {
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      overflow: hidden;
    }
    .ma2-panel h3 {
      margin: 0;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(244, 247, 251, 0.72);
    }
    .ma2-targets,
    .ma2-candidates {
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
      overflow: auto;
    }
    .ma2-target {
      width: 100%;
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
      color: inherit;
      cursor: pointer;
      text-align: left;
    }
    .ma2-target.active {
      border-color: rgba(87, 199, 133, 0.7);
      background: rgba(87, 199, 133, 0.12);
    }
    .ma2-target-main {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }
    .ma2-target-name {
      font-size: 14px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .ma2-target-meta {
      font-size: 12px;
      color: rgba(244, 247, 251, 0.66);
    }
    .ma2-viewer {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
      border-radius: 20px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background:
        linear-gradient(45deg, rgba(255, 255, 255, 0.05) 25%, transparent 25%, transparent 75%, rgba(255, 255, 255, 0.05) 75%),
        linear-gradient(45deg, rgba(255, 255, 255, 0.05) 25%, transparent 25%, transparent 75%, rgba(255, 255, 255, 0.05) 75%),
        rgba(5, 8, 13, 0.7);
      background-size: 24px 24px;
      background-position: 0 0, 12px 12px;
      overflow: hidden;
      position: relative;
    }
    .ma2-viewer img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      user-select: none;
      cursor: crosshair;
      border-radius: 14px;
    }
    .ma2-candidate {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: inherit;
      cursor: pointer;
    }
    .ma2-candidate.active {
      border-color: rgba(54, 128, 255, 0.75);
      background: rgba(54, 128, 255, 0.12);
    }
    .ma2-candidate img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.18);
    }
    .ma2-candidate-score {
      font-size: 12px;
      color: rgba(244, 247, 251, 0.76);
    }
    .ma2-button {
      padding: 9px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.06);
      color: inherit;
      cursor: pointer;
      font: inherit;
    }
    .ma2-button:hover {
      background: rgba(255, 255, 255, 0.1);
    }
    .ma2-button.primary {
      background: linear-gradient(135deg, rgba(87, 199, 133, 0.9), rgba(54, 128, 255, 0.9));
      border-color: transparent;
      color: #081019;
      font-weight: 700;
    }
    .ma2-button.danger {
      background: rgba(244, 91, 105, 0.14);
      border-color: rgba(244, 91, 105, 0.28);
    }
    .ma2-button.selected {
      border-color: rgba(87, 199, 133, 0.7);
      background: rgba(87, 199, 133, 0.18);
    }
    .ma2-button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .ma2-empty {
      padding: 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
      color: rgba(244, 247, 251, 0.68);
      font-size: 13px;
      line-height: 1.5;
    }
    .ma2-error {
      color: #ff9aa8;
      font-size: 12px;
      min-height: 16px;
    }
    .ma2-spinner {
      position: absolute;
      top: 14px;
      right: 14px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(12, 16, 23, 0.82);
      border: 1px solid rgba(255, 255, 255, 0.12);
      font-size: 12px;
    }
    @media (max-width: 1100px) {
      .ma2-shell {
        width: 98vw;
        height: 96vh;
        padding: 14px;
      }
      .ma2-body {
        grid-template-columns: 1fr;
      }
    }
  `;
  document.head.appendChild(style);
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function getWidgetValue(node, name, fallback = "") {
  const widget = getWidget(node, name);
  return widget?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) {
    return;
  }
  const oldValue = widget.value;
  widget.value = value;
  widget.callback?.(value);
  node.onWidgetChanged?.(name, value, oldValue, widget);
  app.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

function safeParseState(rawState) {
  if (!rawState || !String(rawState).trim()) {
    return { targets: [], active_index: 0 };
  }
  try {
    const parsed = JSON.parse(String(rawState));
    if (!Array.isArray(parsed.targets)) {
      return { targets: [], active_index: 0 };
    }
    return {
      targets: parsed.targets.map((target, index) => ({
        name: target?.name || `Mask ${index + 1}`,
        mask_choice: target?.mask_choice || "best",
        points: Array.isArray(target?.points) ? target.points.map((point) => [Number(point[0]), Number(point[1])]) : [],
        labels: Array.isArray(target?.labels) ? target.labels.map((label) => Number(label)) : []
      })),
      active_index: Number.isFinite(parsed.active_index) ? Number(parsed.active_index) : 0
    };
  } catch {
    return { targets: [], active_index: 0 };
  }
}

function normalizeState(state) {
  const targets = Array.isArray(state?.targets) ? state.targets : [];
  const normalizedTargets = targets.map((target, index) => ({
    name: target?.name || `Mask ${index + 1}`,
    mask_choice: ["best", "0", "1", "2"].includes(String(target?.mask_choice)) ? String(target.mask_choice) : "best",
    points: Array.isArray(target?.points) ? target.points.map((point) => [Math.round(Number(point[0]) || 0), Math.round(Number(point[1]) || 0)]) : [],
    labels: Array.isArray(target?.labels) ? target.labels.map((label) => (Number(label) === 0 ? 0 : 1)) : []
  }));
  const activeIndex = normalizedTargets.length
    ? Math.min(Math.max(Number(state?.active_index) || 0, 0), normalizedTargets.length - 1)
    : 0;
  return { targets: normalizedTargets, active_index: activeIndex };
}

function stringifyState(state) {
  return JSON.stringify(normalizeState(state), null, 2);
}

function summarizeState(state) {
  const normalized = normalizeState(state);
  const pointCount = normalized.targets.reduce((sum, target) => sum + target.points.length, 0);
  const maskCount = normalized.targets.filter((target) => target.points.length > 0).length;
  if (!maskCount && !pointCount) {
    return "Open editor";
  }
  return `${maskCount} mask${maskCount === 1 ? "" : "s"}, ${pointCount} point${pointCount === 1 ? "" : "s"}`;
}

function createEmptyTarget(index) {
  return {
    name: `Mask ${index + 1}`,
    mask_choice: "best",
    points: [],
    labels: []
  };
}

function resolveInputSourceNode(node) {
  let sourceNode = node.getInputNode?.(0);
  if (!sourceNode) {
    return null;
  }

  if (sourceNode.isSubgraphNode?.()) {
    const link = node.getInputLink?.(0);
    if (!link) {
      return null;
    }
    const resolved = sourceNode.resolveSubgraphOutputLink?.(link.origin_slot);
    sourceNode = resolved?.outputNode ?? null;
  }

  return sourceNode;
}

function buildViewUrl(entry) {
  const params = new URLSearchParams();
  params.set("filename", entry.filename);
  params.set("type", entry.type || "temp");
  if (entry.subfolder) {
    params.set("subfolder", entry.subfolder);
  }
  return api.apiURL ? api.apiURL(`/view?${params.toString()}`) : `/view?${params.toString()}`;
}

function getNodePreviewUrl(node) {
  const entry = node?.imgs?.[node.imageIndex ?? 0] || node?.imgs?.[0];
  if (!entry) {
    return null;
  }
  if (typeof entry === "string") {
    return entry;
  }
  if (entry.src) {
    return entry.src;
  }
  if (entry.filename) {
    return buildViewUrl(entry);
  }
  return null;
}

async function loadImage(url) {
  return await new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    image.src = url;
  });
}

function imageToDataUrl(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function resolvePreviewUrl(url) {
  if (!url) return url;
  // /view URLs need to go through ComfyUI's API base
  if (url.startsWith("/view?")) {
    return api.apiURL ? api.apiURL(url) : url;
  }
  return url;
}

function updateEditorButton(node) {
  if (!node.__matanyoneEditorButton) {
    return;
  }
  const state = safeParseState(getWidgetValue(node, STATE_WIDGET_NAME, ""));
  node.__matanyoneEditorButton.name = `Open Editor (${summarizeState(state)})`;
  app.graph?.setDirtyCanvas?.(true, true);
}

function setupNode(node) {
  if (!node.widgets) {
    return;
  }

  const stateWidget = getWidget(node, STATE_WIDGET_NAME);
  if (stateWidget && !stateWidget.__matanyoneHidden) {
    stateWidget.hidden = true;
    stateWidget.computeSize = () => [0, -4];
    stateWidget.__matanyoneHidden = true;
  }

  if (!node.__matanyoneEditorButton) {
    node.__matanyoneEditorButton = node.addWidget(
      "button",
      "Open Editor",
      null,
      () => openEditor(node),
      { serialize: false }
    );
  }

  updateEditorButton(node);
}

class MatAnyoneEditorDialog {
  constructor(node) {
    this.node = node;
    this.sessionId = null;
    this.baseImageUrl = null;
    this.previewUrl = null;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.isBusy = false;
    this.errorMessage = "";
    this.pointMode = "positive";
    this.state = normalizeState(safeParseState(getWidgetValue(node, STATE_WIDGET_NAME, "")));
    if (!this.state.targets.length) {
      this.state.targets = [createEmptyTarget(0)];
      this.state.active_index = 0;
    }
  }

  async open() {
    ensureStyles();
    this.build();
    document.body.appendChild(this.overlay);
    document.addEventListener("keydown", this.onKeyDown);
    await this.prepareImage();
    await this.refresh();
  }

  build() {
    this.onKeyDown = (event) => {
      if (event.key === "Escape") {
        this.close();
      }
    };

    this.overlay = document.createElement("div");
    this.overlay.className = "ma2-overlay";
    this.overlay.addEventListener("mousedown", (event) => {
      if (event.target === this.overlay) {
        this.close();
      }
    });

    this.shell = document.createElement("div");
    this.shell.className = "ma2-shell";
    this.overlay.appendChild(this.shell);

    const header = document.createElement("div");
    header.className = "ma2-header";

    const titleWrap = document.createElement("div");
    titleWrap.className = "ma2-title";
    const title = document.createElement("h2");
    title.textContent = "MatAnyone Interactive Editor";
    this.subtitle = document.createElement("div");
    this.subtitle.className = "ma2-subtitle";
    titleWrap.appendChild(title);
    titleWrap.appendChild(this.subtitle);

    const toolbar = document.createElement("div");
    toolbar.className = "ma2-toolbar";
    this.modePositiveButton = this.createButton("Positive", () => {
      this.pointMode = "positive";
      this.render();
    });
    this.modeNegativeButton = this.createButton("Negative", () => {
      this.pointMode = "negative";
      this.render();
    });
    toolbar.appendChild(this.modePositiveButton);
    toolbar.appendChild(this.modeNegativeButton);
    toolbar.appendChild(this.createButton("New Mask", () => this.addTarget()));
    toolbar.appendChild(this.createButton("Undo Point", () => this.undoPoint()));
    toolbar.appendChild(this.createButton("Clear Mask", () => this.clearTarget()));
    toolbar.appendChild(this.createButton("Delete Mask", () => this.deleteTarget(), "danger"));
    toolbar.appendChild(this.createButton("Refresh", () => this.refresh()));

    header.appendChild(titleWrap);
    header.appendChild(toolbar);
    this.shell.appendChild(header);

    const body = document.createElement("div");
    body.className = "ma2-body";

    const targetsPanel = document.createElement("div");
    targetsPanel.className = "ma2-panel";
    const targetsTitle = document.createElement("h3");
    targetsTitle.textContent = "Masks";
    this.targetsList = document.createElement("div");
    this.targetsList.className = "ma2-targets";
    targetsPanel.appendChild(targetsTitle);
    targetsPanel.appendChild(this.targetsList);

    const viewerPanel = document.createElement("div");
    viewerPanel.className = "ma2-panel";
    const viewerTitle = document.createElement("h3");
    viewerTitle.textContent = "Preview";
    this.viewer = document.createElement("div");
    this.viewer.className = "ma2-viewer";
    this.previewImage = document.createElement("img");
    this.previewImage.draggable = false;
    this.previewImage.addEventListener("pointerdown", (event) => this.handleImagePointer(event));
    this.previewImage.addEventListener("contextmenu", (event) => event.preventDefault());
    this.viewer.appendChild(this.previewImage);
    this.spinner = document.createElement("div");
    this.spinner.className = "ma2-spinner";
    this.spinner.textContent = "Updating…";
    this.spinner.hidden = true;
    this.viewer.appendChild(this.spinner);
    this.errorNode = document.createElement("div");
    this.errorNode.className = "ma2-error";
    viewerPanel.appendChild(viewerTitle);
    viewerPanel.appendChild(this.viewer);
    viewerPanel.appendChild(this.errorNode);

    const candidatesPanel = document.createElement("div");
    candidatesPanel.className = "ma2-panel";
    const candidatesTitle = document.createElement("h3");
    candidatesTitle.textContent = "Candidates";
    this.candidatesList = document.createElement("div");
    this.candidatesList.className = "ma2-candidates";
    candidatesPanel.appendChild(candidatesTitle);
    candidatesPanel.appendChild(this.candidatesList);

    body.appendChild(targetsPanel);
    body.appendChild(viewerPanel);
    body.appendChild(candidatesPanel);
    this.shell.appendChild(body);

    const footer = document.createElement("div");
    footer.className = "ma2-footer";
    this.statusNode = document.createElement("div");
    this.statusNode.className = "ma2-status";
    this.statusNode.textContent = "Left click adds a positive point. Right click adds a negative point.";

    const actions = document.createElement("div");
    actions.className = "ma2-actions";
    actions.appendChild(this.createButton("Close", () => this.close()));
    actions.appendChild(this.createButton("Apply To Node", () => this.apply(), "primary"));

    footer.appendChild(this.statusNode);
    footer.appendChild(actions);
    this.shell.appendChild(footer);
  }

  createButton(label, onClick, variant = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `ma2-button ${variant}`.trim();
    button.textContent = label;
    button.addEventListener("click", onClick);
    return button;
  }

  getActiveTarget() {
    if (!this.state.targets.length) {
      this.state.targets.push(createEmptyTarget(0));
      this.state.active_index = 0;
    }
    const activeIndex = Math.min(Math.max(this.state.active_index || 0, 0), this.state.targets.length - 1);
    this.state.active_index = activeIndex;
    return this.state.targets[activeIndex];
  }

  async prepareImage() {
    // Check the node itself first (it saves input frame as UI result after execution),
    // then fall back to checking the upstream source node
    let imageUrl = getNodePreviewUrl(this.node);
    if (!imageUrl) {
      const sourceNode = resolveInputSourceNode(this.node);
      imageUrl = getNodePreviewUrl(sourceNode);
    }
    if (!imageUrl) {
      throw new Error("Queue the workflow once so the editor can access the frame preview.");
    }
    // Keep the /view URL for the session (avoids base64 in request body)
    this.sourceViewUrl = imageUrl;
    const image = await loadImage(imageUrl);
    this.baseImageUrl = imageUrl;
    this.imageWidth = image.naturalWidth;
    this.imageHeight = image.naturalHeight;
    this.previewImage.src = this.baseImageUrl;
    this.subtitle.textContent = `${this.imageWidth} x ${this.imageHeight} · ${getWidgetValue(this.node, "sam_model_type", "vit_h")}`;
    await this.ensureSession();
  }

  async ensureSession() {
    if (this.sessionId) {
      return;
    }
    // Try sending the /view URL reference first (lightweight).
    // Fall back to base64 data URL if the server returns an error.
    let imagePayload = this.sourceViewUrl || "";
    // Extract just the /view?... path from full URLs
    if (imagePayload.includes("/view?")) {
      const viewIndex = imagePayload.indexOf("/view?");
      imagePayload = imagePayload.substring(viewIndex);
    }
    // If we couldn't extract a /view URL, fall back to base64
    if (!imagePayload.startsWith("/view?")) {
      const image = await loadImage(this.sourceViewUrl || this.baseImageUrl);
      imagePayload = imageToDataUrl(image);
    }
    const response = await api.fetchApi("/matanyone2/interactive/create_session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imagePayload,
        sam_model_type: getWidgetValue(this.node, "sam_model_type", "vit_h"),
        checkpoint_path: getWidgetValue(this.node, "checkpoint_path", ""),
        device: getWidgetValue(this.node, "device", "auto")
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Failed to create SAM session (${response.status})`);
    }
    this.sessionId = payload.session_id;
  }

  async refresh() {
    if (!this.sessionId) {
      await this.ensureSession();
    }

    this.errorMessage = "";
    this.setBusy(true);
    try {
      const response = await api.fetchApi("/matanyone2/interactive/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: this.sessionId,
          editor_state: normalizeState(this.state),
          preview_opacity: Number(getWidgetValue(this.node, "preview_opacity", 0.65))
        })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Interactive prediction failed (${response.status})`);
      }
      this.state = normalizeState(payload.state || this.state);
      this.previewUrl = resolvePreviewUrl(payload.preview_url) || this.baseImageUrl;
      this.activeCandidates = Array.isArray(payload.active_candidates)
        ? payload.active_candidates.map((c) => ({ ...c, preview_url: resolvePreviewUrl(c.preview_url) }))
        : [];
    } catch (error) {
      this.errorMessage = error instanceof Error ? error.message : String(error);
      this.previewUrl = this.baseImageUrl;
      this.activeCandidates = [];
    } finally {
      this.setBusy(false);
      this.render();
    }
  }

  setBusy(isBusy) {
    this.isBusy = isBusy;
    if (this.spinner) {
      this.spinner.hidden = !isBusy;
    }
  }

  render() {
    const activeTarget = this.getActiveTarget();
    this.previewImage.src = this.previewUrl || this.baseImageUrl || "";
    this.errorNode.textContent = this.errorMessage || "";
    this.statusNode.textContent = `${summarizeState(this.state)} · Left click = positive, right click = negative`;

    this.modePositiveButton.classList.toggle("selected", this.pointMode === "positive");
    this.modeNegativeButton.classList.toggle("selected", this.pointMode === "negative");

    this.targetsList.replaceChildren();
    this.state.targets.forEach((target, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `ma2-target ${index === this.state.active_index ? "active" : ""}`.trim();
      button.addEventListener("click", () => {
        this.state.active_index = index;
        this.refresh();
      });

      const main = document.createElement("div");
      main.className = "ma2-target-main";
      const name = document.createElement("div");
      name.className = "ma2-target-name";
      name.textContent = target.name || `Mask ${index + 1}`;
      const meta = document.createElement("div");
      meta.className = "ma2-target-meta";
      meta.textContent = `${target.points.length} point${target.points.length === 1 ? "" : "s"} · choice ${target.mask_choice}`;
      main.appendChild(name);
      main.appendChild(meta);

      const dot = document.createElement("div");
      dot.className = "ma2-target-meta";
      dot.textContent = index === this.state.active_index ? "Editing" : "";

      button.appendChild(main);
      button.appendChild(dot);
      this.targetsList.appendChild(button);
    });

    this.candidatesList.replaceChildren();
    if (!this.activeCandidates?.length) {
      const empty = document.createElement("div");
      empty.className = "ma2-empty";
      empty.textContent = activeTarget.points.length
        ? "No alternate masks were returned for this prompt."
        : "Add points on the image to see SAM mask candidates here.";
      this.candidatesList.appendChild(empty);
    } else {
      this.activeCandidates.forEach((candidate) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `ma2-candidate ${candidate.selected ? "active" : ""}`.trim();
        button.addEventListener("click", async () => {
          const target = this.getActiveTarget();
          target.mask_choice = String(candidate.index);
          await this.refresh();
        });

        const image = document.createElement("img");
        image.src = candidate.preview_url;
        image.alt = `Mask candidate ${candidate.index}`;

        const score = document.createElement("div");
        score.className = "ma2-candidate-score";
        score.textContent = `Candidate ${candidate.index} · score ${candidate.score.toFixed(3)}`;

        button.appendChild(image);
        button.appendChild(score);
        this.candidatesList.appendChild(button);
      });
    }
  }

  async handleImagePointer(event) {
    if (this.isBusy) {
      return;
    }
    event.preventDefault();

    const rect = this.previewImage.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return;
    }

    const x = Math.round(((event.clientX - rect.left) / rect.width) * this.imageWidth);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * this.imageHeight);
    const label = event.button === 2 ? 0 : this.pointMode === "negative" ? 0 : 1;

    const target = this.getActiveTarget();
    target.points.push([Math.max(0, Math.min(this.imageWidth - 1, x)), Math.max(0, Math.min(this.imageHeight - 1, y))]);
    target.labels.push(label);
    target.mask_choice = "best";
    await this.refresh();
  }

  addTarget() {
    this.state.targets.push(createEmptyTarget(this.state.targets.length));
    this.state.active_index = this.state.targets.length - 1;
    this.render();
    void this.refresh();
  }

  undoPoint() {
    const target = this.getActiveTarget();
    if (!target.points.length) {
      return;
    }
    target.points.pop();
    target.labels.pop();
    target.mask_choice = "best";
    void this.refresh();
  }

  clearTarget() {
    const target = this.getActiveTarget();
    target.points = [];
    target.labels = [];
    target.mask_choice = "best";
    void this.refresh();
  }

  deleteTarget() {
    if (!this.state.targets.length) {
      return;
    }
    this.state.targets.splice(this.state.active_index, 1);
    if (!this.state.targets.length) {
      this.state.targets.push(createEmptyTarget(0));
      this.state.active_index = 0;
    } else {
      this.state.active_index = Math.min(this.state.active_index, this.state.targets.length - 1);
    }
    void this.refresh();
  }

  apply() {
    const serialized = stringifyState(this.state);
    setWidgetValue(this.node, STATE_WIDGET_NAME, serialized);
    updateEditorButton(this.node);
    this.close();
  }

  async close() {
    if (this.overlay?.isConnected) {
      this.overlay.remove();
    }
    document.removeEventListener("keydown", this.onKeyDown);
    const sessionId = this.sessionId;
    this.sessionId = null;
    if (activeDialog === this) {
      activeDialog = null;
    }
    if (sessionId) {
      try {
        await api.fetchApi("/matanyone2/interactive/close_session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId })
        });
      } catch {
        // Session cleanup is best-effort only.
      }
    }
  }
}

async function openEditor(node) {
  try {
    if (activeDialog) {
      await activeDialog.close();
    }
    const dialog = new MatAnyoneEditorDialog(node);
    activeDialog = dialog;
    await dialog.open();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("[MatAnyone] Failed to open interactive editor:", error);
    window.alert(message);
  }
}

app.registerExtension({
  name: "MatAnyone2.InteractiveEditor",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME && nodeType?.comfyClass !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedPatched() {
      const result = onNodeCreated?.apply(this, arguments);
      setupNode(this);
      return result;
    };

    // Auto-open the editor after execution if no masks have been set yet
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function onExecutedPatched(output) {
      onExecuted?.apply(this, arguments);
      if (output?.has_masks?.[0] === false && !activeDialog) {
        // Small delay to let the UI settle after execution
        const node = this;
        setTimeout(() => openEditor(node), 600);
      }
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function getExtraMenuOptionsPatched(_, options) {
      getExtraMenuOptions?.apply(this, arguments);
      options.unshift({
        content: "Open MatAnyone Editor",
        callback: () => openEditor(this)
      });
    };
  }
});

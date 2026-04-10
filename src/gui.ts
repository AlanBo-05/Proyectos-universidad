// gui.ts — Glassmorphism GUI for multi-object pipeline

export interface GUIState {
  ambient:      number;
  diffuse:      number;
  specular:     number;
  shininess:    number;
  lightX:       number;
  lightY:       number;
  lightZ:       number;
  autoRotLight: boolean;
  lightColor:   string;
  wireThick:    number;
}

interface GUICallbacks {
  onLoadBeacon:     () => Promise<void>;
  onLoadTeapot:     () => Promise<void>;
  onLoadCube:       () => void;
  onLoadSphere:     () => void;
  onRemoveObject:   (uid: number) => void;
  onShadingChange:  (uid: number, mode: number) => void;
  onColorChange:    (uid: number, hex: string) => void;
  onWireThickChange:(v: number) => void;
  getObjects:       () => { uid: number; name: string }[];
  gui:              GUIState;
}

const SHADING_MODES = [
  { id: 0, label: "Flat" },
  { id: 1, label: "Gouraud" },
  { id: 2, label: "Phong" },
  { id: 3, label: "Wireframe" },
  { id: 4, label: "Normals" },
  { id: 5, label: "Texture" },
];

const OBJ_COLORS: Record<string, string> = {
  Beacon: "#e8992e",
  Teapot: "#4a9eff",
  Cube:   "#cc55ff",
  Sphere: "#44e896",
};

// Track per-instance shading mode (by uid)
const objShadingMode = new Map<number, number>();

export function initGUI(cb: GUICallbacks) {
  const g = cb.gui;

  const overlay = document.createElement("div");
  overlay.id = "gui-overlay";
  overlay.innerHTML = `
<div class="gui-panel" id="gui-panel">
  <div class="gui-header">
    <span class="gui-title"> 3D Pipeline</span>
    <button class="gui-toggle" id="gui-toggle" title="Collapse">−</button>
  </div>
  <div class="gui-body" id="gui-body">

    <!-- Load models -->
    <section class="gui-section">
      <div class="gui-label">Load Models</div>
      <div class="btn-row">
        <button class="load-btn" id="btn-beacon">Beacon</button>
        <button class="load-btn" id="btn-teapot">Teapot</button>
        <button class="load-btn" id="btn-cube">Cube</button>
        <button class="load-btn" id="btn-sphere">Sphere</button>
      </div>
      <div class="status-line" id="load-status">Beacon loaded</div>
    </section>

    <!-- Per-object controls (injected dynamically) -->
    <section class="gui-section" id="obj-section">
      <div class="gui-label">Objects</div>
      <div id="obj-list"></div>
    </section>

    <!-- Material -->
    <section class="gui-section">
      <div class="gui-label">Material</div>
      ${slider("ambient",   "Ambient  Ka",  0, 1,   0.01, g.ambient)}
      ${slider("diffuse",   "Diffuse  Kd",  0, 1,   0.01, g.diffuse)}
      ${slider("specular",  "Specular Ks",  0, 1,   0.01, g.specular)}
      ${slider("shininess", "Shininess n",  1, 256, 1,    g.shininess)}
    </section>

    <!-- Light -->
    <section class="gui-section">
      <div class="gui-label">Light</div>
      ${slider("lightX", "X", -10, 10, 0.1, g.lightX)}
      ${slider("lightY", "Y", -10, 10, 0.1, g.lightY)}
      ${slider("lightZ", "Z", -10, 10, 0.1, g.lightZ)}
      <label class="check-row">
        <input type="checkbox" id="autoRotLight" ${g.autoRotLight ? "checked" : ""}> Auto-rotate
      </label>
      <div class="color-row">
        <span>Light Color</span>
        <input type="color" id="lightColor" value="${g.lightColor}">
      </div>
    </section>

    <!-- Wireframe -->
    <section class="gui-section">
      <div class="gui-label">Wireframe</div>
      ${slider("wireThick", "Edge width", 0.5, 4, 0.1, g.wireThick)}
    </section>

    <!-- Controls hint -->
    <div class="gui-hint">
      🖱 Drag = orbit &nbsp;|&nbsp; Scroll = zoom<br>
      WASD/QE = pan &nbsp;|&nbsp; R = reset camera
    </div>
  </div>
</div>`;

  document.body.appendChild(overlay);

  // ── Collapse toggle ─────────────────────────────────────────────────────────
  const toggleBtn = document.getElementById("gui-toggle")!;
  const guiBody   = document.getElementById("gui-body")!;
  toggleBtn.addEventListener("click", () => {
    const collapsed = guiBody.style.display === "none";
    guiBody.style.display = collapsed ? "" : "none";
    toggleBtn.textContent = collapsed ? "−" : "+";
  });

  // ── Load buttons ─────────────────────────────────────────────────────────────
  function setStatus(msg: string) {
    document.getElementById("load-status")!.textContent = msg;
  }
  function withLoading(label: string, fn: () => Promise<void>) {
    setStatus(`Loading ${label}…`);
    fn().then(() => {
      setStatus(`${label} loaded ✓`);
      rebuildObjList();
    }).catch(err => {
      setStatus(`Error: ${err.message}`);
      console.error(err);
    });
  }

  document.getElementById("btn-beacon")!.addEventListener("click", () =>
    withLoading("Beacon", cb.onLoadBeacon));
  document.getElementById("btn-teapot")!.addEventListener("click", () =>
    withLoading("Teapot", cb.onLoadTeapot));

  document.getElementById("btn-cube")!.addEventListener("click", () => {
    cb.onLoadCube();
    setStatus("Cube added ✓");
    rebuildObjList();
  });
  document.getElementById("btn-sphere")!.addEventListener("click", () => {
    cb.onLoadSphere();
    setStatus("Sphere added ✓");
    rebuildObjList();
  });

  // ── Material sliders ─────────────────────────────────────────────────────────
  (["ambient", "diffuse", "specular", "shininess"] as const).forEach(id => {
    const el    = document.getElementById(id) as HTMLInputElement;
    const valEl = document.getElementById(`${id}-val`)!;
    el.addEventListener("input", () => {
      (g as any)[id] = parseFloat(el.value);
      valEl.textContent = el.value;
    });
  });

  // ── Light sliders ─────────────────────────────────────────────────────────────
  (["lightX", "lightY", "lightZ"] as const).forEach(id => {
    const el    = document.getElementById(id) as HTMLInputElement;
    const valEl = document.getElementById(`${id}-val`)!;
    el.addEventListener("input", () => {
      (g as any)[id] = parseFloat(el.value);
      valEl.textContent = el.value;
    });
  });

  (document.getElementById("autoRotLight") as HTMLInputElement)
    .addEventListener("change", e => { g.autoRotLight = (e.target as HTMLInputElement).checked; });

  (document.getElementById("lightColor") as HTMLInputElement)
    .addEventListener("input", e => { g.lightColor = (e.target as HTMLInputElement).value; });

  // ── Wireframe slider ─────────────────────────────────────────────────────────
  const wtEl    = document.getElementById("wireThick") as HTMLInputElement;
  const wtValEl = document.getElementById("wireThick-val")!;
  wtEl.addEventListener("input", () => {
    const v = parseFloat(wtEl.value);
    g.wireThick = v;
    wtValEl.textContent = v.toFixed(1);
    cb.onWireThickChange(v);
  });

  // ── Per-object list ──────────────────────────────────────────────────────────
  function rebuildObjList() {
    const list  = document.getElementById("obj-list")!;
    const items = cb.getObjects();
    list.innerHTML = "";

    if (items.length === 0) {
      list.innerHTML = `<div class="empty-hint">No objects loaded</div>`;
      return;
    }

    for (const { uid, name } of items) {
      if (!objShadingMode.has(uid)) objShadingMode.set(uid, 2); // default Phong
      const curMode = objShadingMode.get(uid)!;
      const defColor = OBJ_COLORS[name] ?? "#4a9eff";

      const card = document.createElement("div");
      card.className = "obj-card";
      card.innerHTML = `
        <div class="obj-card-header">
          <span class="obj-name">${name}</span>
          <button class="obj-remove" title="Remove">✕</button>
        </div>
        <div class="shading-btns">
          ${SHADING_MODES.map(m =>
            `<button class="shade-btn ${curMode === m.id ? "active" : ""}" data-mode="${m.id}">${m.label}</button>`
          ).join("")}
        </div>
        <div class="color-row">
          <span>Color</span>
          <input type="color" class="obj-color" value="${defColor}">
        </div>`;
      list.appendChild(card);

      // Delete button
      card.querySelector<HTMLButtonElement>(".obj-remove")!.addEventListener("click", () => {
        cb.onRemoveObject(uid);
        objShadingMode.delete(uid);
        rebuildObjList();
      });

      // Shading buttons
      card.querySelectorAll<HTMLButtonElement>(".shade-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const mode = Number(btn.dataset.mode);
          objShadingMode.set(uid, mode);
          cb.onShadingChange(uid, mode);
          card.querySelectorAll(".shade-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
        });
      });

      // Color picker
      const colorInput = card.querySelector<HTMLInputElement>(".obj-color")!;
      colorInput.addEventListener("input", () => {
        cb.onColorChange(uid, colorInput.value);
      });
    }
  }

  // Build initial list after first load
  rebuildObjList();

  // Expose so loading callbacks can refresh
  (window as unknown as Record<string, unknown>)["_rebuildObjList"] = rebuildObjList;
}

// ─── Helper ──────────────────────────────────────────────────────────────────
function slider(id: string, label: string, min: number, max: number, step: number, val: number) {
  return `
  <div class="slider-row">
    <span class="slider-label">${label}</span>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}">
    <span class="slider-val" id="${id}-val">${val}</span>
  </div>`;
}

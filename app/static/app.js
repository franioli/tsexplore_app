// -----------------------------------------------------------------------------
// Global state (used by template too)
// -----------------------------------------------------------------------------
window.currentDate = window.currentDate ?? null; // YYYYMMDD
window.selectedNode = window.selectedNode ?? null; // {x, y}

// Expose for inline template JS (index.html uses it)
window.fetchVelocityMap = fetchVelocityMap;

// -----------------------------------------------------------------------------
// Small DOM helpers
// -----------------------------------------------------------------------------
function el(id) {
  return document.getElementById(id);
}

function on(id, event, handler) {
  const node = el(id);
  if (!node) return false;
  node.addEventListener(event, handler);
  return true;
}

function valueOf(id, fallback = null) {
  const node = el(id);
  if (!node) return fallback;
  const v = (node.value ?? "").toString().trim();
  return v === "" ? fallback : v;
}

function checkedOf(id, fallback = false) {
  const node = el(id);
  if (!node) return fallback;
  return !!node.checked;
}

function toNumberOrNull(x) {
  if (x === null || x === undefined) return null;
  const s = String(x).trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

// -----------------------------------------------------------------------------
// dt selection helpers (shared by map/timeseries/nearest)
// -----------------------------------------------------------------------------
function getDtSelectionParams() {
  const mode = (valueOf("dt-mode", "auto") || "auto").toLowerCase();
  const dtRaw = valueOf("dt-days", "");
  const tolRaw = valueOf("dt-tolerance", "");

  if (mode === "auto") return {};

  if (mode === "exact") {
    if (!dtRaw) return {};
    return { dt_days: dtRaw };
  }

  if (mode === "closest") {
    if (!dtRaw) return {};
    const out = { prefer_dt_days: dtRaw };
    if (tolRaw !== null && tolRaw !== "") out.prefer_dt_tolerance = tolRaw;
    return out;
  }

  return {};
}

function applyDtParams(url) {
  const params = getDtSelectionParams();
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
}

function syncDtControls() {
  const modeEl = el("dt-mode");
  const dtEl = el("dt-days");
  const tolEl = el("dt-tolerance");
  if (!modeEl || !dtEl || !tolEl) return;

  const mode = (modeEl.value || "auto").toLowerCase();
  if (mode === "auto") {
    dtEl.disabled = true;
    tolEl.disabled = true;
    return;
  }
  if (mode === "exact") {
    dtEl.disabled = false;
    tolEl.disabled = true;
    return;
  }
  if (mode === "closest") {
    dtEl.disabled = false;
    tolEl.disabled = false;
    return;
  }
}


// -----------------------------------------------------------------------------
// Data loading
// -----------------------------------------------------------------------------
async function startLoad() {
  const startDate = valueOf("load-start-picker", null); // expected "YYYY-MM-DD" (flatpickr)
  const endDate = valueOf("load-end-picker", null);

  const url = new URL("/api/loader/load", window.location.origin);
  if (startDate) url.searchParams.set("start_date", startDate);
  if (endDate) url.searchParams.set("end_date", endDate);

  console.info("Requesting load", { startDate, endDate, url: url.toString() });

  const res = await fetch(url.toString(), { method: "POST" });
  if (!res.ok) {
    console.error("load request failed", await res.text());
    alert("Failed to start loading: " + (await res.text()));
    return;
  }

  console.info("load started successfully");

  const progressEl = el("load-progress");
  const statusEl = el("load-status");
  if (progressEl) progressEl.style.display = "block";
  if (statusEl) statusEl.textContent = "Loading...";

  const poll = setInterval(async () => {
    const urlProg = new URL("/api/loader/progress", window.location.origin);
    const r = await fetch(urlProg.toString());
    if (!r.ok) {
      if (statusEl) statusEl.textContent = "Failed to read progress.";
      console.error("progress polling failed", await r.text());
      clearInterval(poll);
      return;
    }

    const p = await r.json();
    console.debug("load-range progress", p);

    if (progressEl) {
      if (p.total && p.total > 0) {
        progressEl.value = Math.round((p.done / p.total) * 100);
      } else {
        progressEl.value = 0;
      }
    }

    if (!p.in_progress) {
      if (statusEl) statusEl.textContent = p.error ? `Error: ${p.error}` : "Load complete";
      clearInterval(poll);

      window.location.reload();
    }
  }, 500);
}
// -----------------------------------------------------------------------------
// API calls + plotting
// -----------------------------------------------------------------------------
async function fetchVelocityMap() {
  if (!window.currentDate) return;

  if (typeof Plotly === "undefined") {
    console.error("Plotly is not available on window");
    return;
  }

  // Required controls (guarded). If missing, we still try reasonable defaults.
  const useVelocity = checkedOf("use-velocity", true);
  const plotType = valueOf("plot-type", "scatter");
  const colorscale = valueOf("colorscale", "Reds");

  const cmin = valueOf("cmin", null);
  const cmax = valueOf("cmax", null);

  const markerSize = valueOf("marker-size", "6");
  const markerOpacity = valueOf("marker-opacity", "auto");

  const url = new URL("/api/velocitymap", window.location.origin);
  url.searchParams.set("reference_date", window.currentDate);
  url.searchParams.set("use_velocity", String(useVelocity));
  url.searchParams.set("plot_type", plotType);

  if (cmin !== null) url.searchParams.set("cmin", cmin);
  if (cmax !== null) url.searchParams.set("cmax", cmax);

  url.searchParams.set("colorscale", colorscale);
  url.searchParams.set("marker_size", markerSize);
  url.searchParams.set("marker_opacity", markerOpacity);
  url.searchParams.set("downsample_points", "5000");

  applyDtParams(url);

  if (window.selectedNode) {
    url.searchParams.set("selected_x", String(window.selectedNode.x));
    url.searchParams.set("selected_y", String(window.selectedNode.y));
  }

  const upperDiv = el("upper");
  if (!upperDiv) {
    console.warn("Missing #upper div; cannot render velocity map");
    return;
  }

  const res = await fetch(url.toString());
  if (!res.ok) {
    console.error("Failed to fetch velocity map:", await res.text());
    return;
  }

  const fig = await res.json();
  await Plotly.newPlot("upper", fig.data, fig.layout);

  // Attach click handler once per render (Plotly replaces handlers on newPlot anyway)
  if (typeof upperDiv.on === "function") {
    upperDiv.on("plotly_click", async (data) => {
      const p = data?.points?.[0];
      if (!p) return;
      await fetchTimeseriesAt(p.x, p.y);
    });
  }
}

async function fetchTimeseriesAt(x, y, runInversion = false) {
  if (!window.currentDate) return;

  if (typeof Plotly === "undefined") {
    console.error("Plotly is not available on window");
    return;
  }

  // 1) nearest node (server-side)
  const radius = toNumberOrNull(valueOf("radius", "10")) ?? 10;

  const urlNearest = new URL("/api/nearest", window.location.origin);
  urlNearest.searchParams.set("reference_date", window.currentDate);
  urlNearest.searchParams.set("x", String(x));
  urlNearest.searchParams.set("y", String(y));
  urlNearest.searchParams.set("radius", String(radius));
  applyDtParams(urlNearest); // ok if backend ignores unknown params

  const resN = await fetch(urlNearest.toString());
  if (!resN.ok) {
    alert("No node within radius");
    return;
  }

  const node = await resN.json();
  window.selectedNode = { x: node.x, y: node.y };

  // refresh map to draw selected marker
  await fetchVelocityMap();

  // 2) timeseries
  const useVelocity = checkedOf("use-velocity", true);

  const componentsSelect = el("components");
  const components = componentsSelect
    ? Array.from(componentsSelect.selectedOptions).map((o) => o.value).join(",") || "V"
    : "V";

  const markerMode = valueOf("marker-mode", "lines+markers");
  const showErrorBand = checkedOf("error-band", false);

  const xminIso = (() => {
    const d = valueOf("xmin-date-picker", null);
    return d ? d : null;
  })();

  const xmaxIso = (() => {
    const d = valueOf("xmax-date-picker", null);
    return d ? d : null;
  })();

  const ymin = valueOf("ymin", null);
  const ymax = valueOf("ymax", null);

  const urlTS = new URL("/api/timeseries", window.location.origin);
  urlTS.searchParams.set("node_x", String(node.x));
  urlTS.searchParams.set("node_y", String(node.y));
  urlTS.searchParams.set("use_velocity", String(useVelocity));
  urlTS.searchParams.set("components", components);
  urlTS.searchParams.set("marker_mode", markerMode);
  urlTS.searchParams.set("show_error_band", String(showErrorBand));
  urlTS.searchParams.set("ts_inversion", String(runInversion));

  if (xminIso) urlTS.searchParams.set("xmin_date", xminIso);
  if (xmaxIso) urlTS.searchParams.set("xmax_date", xmaxIso);
  if (ymin !== null) urlTS.searchParams.set("ymin", ymin);
  if (ymax !== null) urlTS.searchParams.set("ymax", ymax);

  applyDtParams(urlTS);

  const lowerDiv = el("lower");
  if (!lowerDiv) {
    console.warn("Missing #lower div; cannot render time series");
    return;
  }

  const resTS = await fetch(urlTS.toString());
  if (!resTS.ok) {
    console.error("Failed to fetch timeseries:", await resTS.text());
    return;
  }

  const figTS = await resTS.json();
  await Plotly.newPlot("lower", figTS.data, figTS.layout);
}

async function runTSInversion() {
  if (!window.selectedNode) {
    alert("Please select a node first by clicking on the velocity map");
    return;
  }

  const btn = el("ts-inversion");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Running...";
  }

  try {
    await fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y, true);
  } catch (err) {
    console.error("Error running TS inversion:", err);
    alert("Error running TS inversion. Check console for details.");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Run TS inversion";
    }
  }
}
// -----------------------------------------------------------------------------
// Initialization / event wiring
// -----------------------------------------------------------------------------
function wireControls() {
  // Data load range button (now always enabled)
  on("load-data", "click", startLoad);

  // Velocity map reactive controls
  on("use-velocity", "change", () => {
    if (!window.currentDate) return;
    fetchVelocityMap();
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  });

  on("plot-type", "change", () => window.currentDate && fetchVelocityMap());
  on("colorscale", "change", () => window.currentDate && fetchVelocityMap());
  on("cmin", "input", () => window.currentDate && fetchVelocityMap());
  on("cmax", "input", () => window.currentDate && fetchVelocityMap());
  on("marker-size", "input", () => window.currentDate && fetchVelocityMap());
  on("marker-opacity", "input", () => window.currentDate && fetchVelocityMap());

  // Time series reactive controls
  const tsRefresh = () => {
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  };
  on("components", "change", tsRefresh);
  on("marker-mode", "change", tsRefresh);
  on("xmin-date-picker", "change", tsRefresh);
  on("xmax-date-picker", "change", tsRefresh);
  on("ymin", "input", tsRefresh);
  on("ymax", "input", tsRefresh);
  on("error-band", "change", tsRefresh);

  // dt controls
  on("dt-mode", "change", () => {
    syncDtControls();
    if (!window.currentDate) return;
    fetchVelocityMap();
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  });
  on("dt-days", "input", () => {
    if (!window.currentDate) return;
    fetchVelocityMap();
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  });
  on("dt-tolerance", "input", () => {
    if (!window.currentDate) return;
    fetchVelocityMap();
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  });

  // TS inversion button
  on("ts-inversion", "click", runTSInversion);
}

function initialPlot() {
  // Intentionally empty: do NOT auto-load a plot on startup.
  // The main date picker (in the template) triggers fetchVelocityMap()
  // when the user selects a date.
}
// ...existing code...
// Single entry point
document.addEventListener("DOMContentLoaded", () => {
  syncDtControls();
  wireControls();
});
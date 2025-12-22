// -----------------------------------------------------------------------------
// Global state (used by template too)
// -----------------------------------------------------------------------------
window.currentDate = window.currentDate ?? null; // YYYYMMDD
window.selectedNode = window.selectedNode ?? null; // {x, y}
window.lastInversion = window.lastInversion ?? null; // last inversion result

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
// API calls
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
    // Try to parse server JSON error (HTTPException -> { detail: "..." })
    let msg;
    try {
      const body = await resN.json();
      msg = body.detail || body.message || JSON.stringify(body);
    } catch {
      // Fallback to plain text or status text
      msg = await resN.text().catch(() => resN.statusText || `status ${resN.status}`);
    }  
    console.warn("Nearest API error:", resN.status, msg);
    alert(`Nearest node error (${resN.status}): ${msg}`);
    return;
  }

  const node = await resN.json();
  window.selectedNode = { x: node.x, y: node.y };

  // enable inversion button now that a node is selected
  updateInversionButton();

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
  console.info("runTSInversion called");
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
    // Read inversion controls from the GUI
    const weight_method = valueOf("inv-weight-method", "residuals");
    const regularization_method = valueOf("inv-regularization-method", "laplacian");
    const lambda_scaling = toNumberOrNull(valueOf("inv-lambda-scaling", "1.0")) ?? 1.0;
    const iterates = parseInt(valueOf("inv-iterates", "10"), 10) || 10;
    const refresh_plot = checkedOf("inv-refresh-plot", false);

    // Read optional date-range filters for inversion
    const date_min = valueOf("xmin-date-picker", null);
    const date_max = valueOf("xmax-date-picker", null);

    // Ensure a time-series plot exists (fetch if missing)
    const lowerDiv = el("lower");
    const hasPlot = !!(lowerDiv && lowerDiv.data && lowerDiv.data.length > 0);
    if (!hasPlot) {
      console.debug("No timeseries plot found — fetching it before inversion");
      await fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
    }

    // If requested, refresh the timeseries plot (call timeseries endpoint to get a fresh base)
    if (refresh_plot) {
      console.debug("Refreshing timeseries plot before inversion (refresh_plot=true)");
      await fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
    }

    // Build request body: node coords
    const body = {
      node_x: window.selectedNode.x,
      node_y: window.selectedNode.y,
      iterates: iterates,
      regularization_method: regularization_method,
      lambda_scaling: lambda_scaling,
      weight_method: weight_method,
    };
    // Optional date filters
   if (date_min) body.date_min = date_min;
   if (date_max) body.date_max = date_max;
    console.debug("Running inversion with params:", body);
    const res = await fetch("/api/inversion/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      let msg = await (async () => {
        try {
          const body = await res.json();
          return body.detail || body.message || JSON.stringify(body);
        } catch {
          return await res.text().catch(() => res.statusText || `status ${res.status}`);
        }
      })();
      console.error("Inversion failed:", msg);
      alert("Inversion failed: " + msg);
      return;
    }

    const payload = await res.json();
    const inv = payload.node_inversion;
    console.debug("Inversion result", inv);
    if (!inv || !inv.dates || !inv.V_hat) {
      alert("No inversion result returned");
      return;
    }

    // Store last inversion globally for download
    window.lastInversion = inv;
    enableDownloadButton(true);

    try {
      // Ask server to build the Plotly trace for this inversion result
      const rTrace = await fetch("/api/inversion/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_inversion: inv, refresh_plot: refresh_plot }),
      });
      if (!rTrace.ok) {
        console.error("Failed to build inversion trace:", await rTrace.text());
        alert("Failed to build inversion overlay");
      } else {
        const tracePayload = await rTrace.json();
        const trace = tracePayload.trace;
        if (lowerDiv && lowerDiv.data && lowerDiv.data.length > 0) {
          await Plotly.addTraces("lower", trace);
        } else {
          await Plotly.newPlot("lower", [trace]);
        }
      }
    } catch (err) {
      console.error("Error adding inversion overlay:", err);
      alert("Error adding inversion overlay. See console.");
    }

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

async function downloadInversionResult() {
  try {
    // If button clicked when there is no inversion, inform the user
    if (!window.lastInversion || !window.selectedNode) {
      alert("No inversion available to download. Run inversion first.");
      return;
    }
    const body = {
      node_inversion: window.lastInversion,
      node_x: window.selectedNode.x,
      node_y: window.selectedNode.y,
    };

    const res = await fetch("/api/inversion/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let msg;
      try {
        msg = (await res.json()).detail || JSON.stringify(await res.json());
      } catch {
        msg = await res.text().catch(() => res.statusText || `${res.status}`);
      }
      throw new Error(msg);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    // try to pick filename from Content-Disposition (fallback to generic)
    const cd = res.headers.get("Content-Disposition") || '';
    const m = cd.match(/filename="([^"]+)"/);
    const fname = m ? m[1] : "node_inversion.txt";
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Download failed:", err);
    alert("Failed to download inversion: " + (err.message || err));
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

  // wire download button
  on("ts-inversion-download", "click", downloadInversionResult);

}

function initialPlot() {
  // Intentionally empty: do NOT auto-load a plot on startup.
  // The main date picker (in the template) triggers fetchVelocityMap()
  // when the user selects a date.
}

function updateInversionButton() {
  const btn = el("ts-inversion");
  if (!btn) return;
  const enabled = !!window.selectedNode;

  // Keep disabled attribute for accessibility + keyboard support
  btn.disabled = !enabled;
  // Use visual class (so clicks can still be handled to show messages)
  btn.classList.toggle("is-disabled", !enabled);
  btn.setAttribute("aria-disabled", String(!enabled));
  btn.title = enabled ? "Run TS inversion" : "Select a node to enable";

  // reset download availability when node is changed
  enableDownloadButton(false);
}

// enable/disable download button when inversion becomes available
function enableDownloadButton(enabled) {
  const btn = el("ts-inversion-download");
  if (!btn) return;

  // Use a visual "is-disabled" class instead of the disabled attribute so clicks can be handled
  btn.classList.toggle("is-disabled", !enabled);
  btn.setAttribute("aria-disabled", String(!enabled));
  if (!enabled) {
    btn.title = "Run an inversion first to enable download";
  } else {
    btn.title = "Download last inversion";
  }


}

// Single entry point
document.addEventListener("DOMContentLoaded", () => {
  syncDtControls();
  wireControls();
  updateInversionButton(); // ensure initial disabled state
  enableDownloadButton(false);

});
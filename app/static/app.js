// Initialize global state
let selectedNode = null;
let currentDate = null;

// Make these globally accessible
window.currentDate = currentDate;
window.selectedNode = selectedNode;
window.fetchVelocityMap = fetchVelocityMap;


async function startLoadRange() {
  if (!USE_DATABASE || !document.getElementById("load-start-picker") || !document.getElementById("load-end-picker")) {
    alert("Load range only available for database backend.");
    return;
  }

  const startDate = document.getElementById("load-start-picker").value;
  const endDate = document.getElementById("load-end-picker").value;
  if (!startDate || !endDate) {
    alert("Please select both start and end dates.");
    return;
  }

  const url = new URL("/api/loader/load-range", window.location.origin);
  // format start/end using YYYYMMDD or use iso; both accepted by API
  const params = { start_date: startDate, end_date: endDate };

  const res = await fetch(url.toString() + `?start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`, {
    method: "POST",
  });

  if (!res.ok) {
    alert("Failed to start loading: " + await res.text());
    return;
  }

  // Show progress bar and start polling
  document.getElementById("load-progress").style.display = "block";
  document.getElementById("load-status").textContent = "Loading...";

  const poll = setInterval(async () => {
    const urlProg = new URL("/api/loader/progress", window.location.origin);
    const r = await fetch(urlProg.toString());
    if (!r.ok) {
      document.getElementById("load-status").textContent = "Failed to read progress.";
      clearInterval(poll);
      return;
    }
    const p = await r.json();
    if (p.total && p.total > 0) {
      const percent = Math.round((p.done / p.total) * 100);
      document.getElementById("load-progress").value = percent;
    } else {
      document.getElementById("load-progress").value = 0;
    }

    if (!p.in_progress) {
      // finished (success or error)
      if (p.error) {
        document.getElementById("load-status").textContent = "Error: " + p.error;
      } else {
        document.getElementById("load-status").textContent = "Load complete";
        // reload map and timeseries UI
        if (window.currentDate) {
          fetchVelocityMap();
          if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
        }
      }
      clearInterval(poll);
    }
  }, 1000);
}


async function fetchVelocityMap() {
  if (!window.currentDate) {
    console.log("No current date set, skipping velocity map fetch");
    return;
  }
  
  const useVelocity = document.getElementById("use-velocity").checked;
  const plotType = document.getElementById("plot-type").value || "scatter";
  const cmin = document.getElementById("cmin").value || null;
  const cmax = document.getElementById("cmax").value || null;
  const colorscale = document.getElementById("colorscale").value;
  const markerSize = document.getElementById("marker-size").value || 6;
  const markerOpacity = document.getElementById("marker-opacity").value || 0.7;
  
  const url = new URL("/api/velocity-map", window.location.origin);
  url.searchParams.set("date", window.currentDate);
  url.searchParams.set("use_velocity", useVelocity);
  url.searchParams.set("plot_type", plotType);
  if (cmin) url.searchParams.set("cmin", cmin);
  if (cmax) url.searchParams.set("cmax", cmax);
  url.searchParams.set("colorscale", colorscale);
  url.searchParams.set("marker_size", markerSize);
  url.searchParams.set("marker_opacity", markerOpacity);
  url.searchParams.set("downsample_points", 5000);
  
  if (window.selectedNode) {
    url.searchParams.set("selected_x", window.selectedNode.x);
    url.searchParams.set("selected_y", window.selectedNode.y);
  }

  try {
    const res = await fetch(url);
    if (!res.ok) {
      console.error("Failed to fetch velocity map:", await res.text());
      return;
    }
    const fig = await res.json();
    await Plotly.newPlot("upper", fig.data, fig.layout);
    
    const upper = document.getElementById("upper");
    upper.on("plotly_click", async function (data) {
      const p = data.points[0];
      await fetchTimeseriesAt(p.x, p.y);
    });
  } catch (err) {
    console.error("Error fetching velocity map:", err);
  }
}


async function fetchTimeseriesAt(x, y, runInversion = false) {
  const radius = document.getElementById("radius").value || 10;

  try {
    const urlNearest = new URL("/api/nearest", window.location.origin);
    urlNearest.searchParams.set("date", window.currentDate);
    urlNearest.searchParams.set("x", x);
    urlNearest.searchParams.set("y", y);
    urlNearest.searchParams.set("radius", radius);
    
    const resN = await fetch(urlNearest);
    if (!resN.ok) {
      alert("No node within radius");
      return;
    }
    const node = await resN.json();
    window.selectedNode = { x: node.x, y: node.y };
    await fetchVelocityMap();

    const useVelocity = document.getElementById("use-velocity").checked;
    const componentsSelect = document.getElementById("components");
    const selectedOptions = Array.from(componentsSelect.selectedOptions).map(opt => opt.value);
    const components = selectedOptions.join(",") || "V";
    const markerMode = document.getElementById("marker-mode").value;
    const showErrorBand = document.getElementById("error-band").checked;

    const xminDate = document.getElementById("xmin-date-picker").value || null;
    const xmaxDate = document.getElementById("xmax-date-picker").value || null;
    const ymin = document.getElementById("ymin").value || null;
    const ymax = document.getElementById("ymax").value || null;

    const urlTS = new URL("/api/timeseries", window.location.origin);
    urlTS.searchParams.set("node_x", node.x);
    urlTS.searchParams.set("node_y", node.y);
    urlTS.searchParams.set("use_velocity", useVelocity);
    urlTS.searchParams.set("components", components);
    urlTS.searchParams.set("marker_mode", markerMode);
    urlTS.searchParams.set("show_error_band", showErrorBand);
    urlTS.searchParams.set("ts_inversion", runInversion);

    if (xminDate) urlTS.searchParams.set("xmin_date", xminDate);
    if (xmaxDate) urlTS.searchParams.set("xmax_date", xmaxDate);
    if (ymin) urlTS.searchParams.set("ymin", ymin);
    if (ymax) urlTS.searchParams.set("ymax", ymax);
        
    // Debug: Log the full URL
    console.log("Time series URL:", urlTS.toString());

    const resTS = await fetch(urlTS);
    if (!resTS.ok) {
      console.error("Failed to fetch timeseries:", await resTS.text());
      return;
    }
    const figTS = await resTS.json();
    await Plotly.newPlot("lower", figTS.data, figTS.layout);

  } catch (err) {
    console.error("Error fetching timeseries:", err);
  }
}


// Run TS inversion function
async function runTSInversion() {
  if (!window.selectedNode) {
    alert("Please select a node first by clicking on the velocity map");
    return;
  }

  const btn = document.getElementById("ts-inversion");
  
  // Disable button and change text
  btn.disabled = true;
  btn.textContent = "Running...";
  
  try {
    // Fetch time series with inversion flag
    await fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y, true);
  } catch (err) {
    console.error("Error running TS inversion:", err);
    alert("Error running TS inversion. Check console for details.");
  } finally {
    // Reset button state
    btn.disabled = false;
    btn.textContent = "Run TS inversion";
  }
}


// Auto-update velocity map when parameters change
document.getElementById("use-velocity").addEventListener("change", () => {
  if (window.currentDate) {
    fetchVelocityMap();
    if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
  }
});

document.getElementById("plot-type").addEventListener("change", () => {
  if (window.currentDate) fetchVelocityMap();
});

document.getElementById("colorscale").addEventListener("change", () => {
  if (window.currentDate) fetchVelocityMap();
});

document.getElementById("cmin").addEventListener("input", () => {
  if (window.currentDate) fetchVelocityMap();
});

document.getElementById("cmax").addEventListener("input", () => {
  if (window.currentDate) fetchVelocityMap();
});

document.getElementById("marker-size").addEventListener("input", () => {
  if (window.currentDate) fetchVelocityMap();
});

document.getElementById("marker-opacity").addEventListener("input", () => {
  if (window.currentDate) fetchVelocityMap();
});

// Auto-update time series when parameters change
document.getElementById("components").addEventListener("change", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("marker-mode").addEventListener("change", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("xmin-date-picker").addEventListener("change", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("xmax-date-picker").addEventListener("change", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("ymin").addEventListener("input", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("ymax").addEventListener("input", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});
document.getElementById("error-band").addEventListener("change", () => {
  if (window.selectedNode) fetchTimeseriesAt(window.selectedNode.x, window.selectedNode.y);
});

// Attach TS inversion button click handler
document.getElementById("ts-inversion").addEventListener("click", runTSInversion);


// wire the button on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("load-data");
  if (btn) btn.addEventListener("click", startLoadRange);

  // init datepickers for load range inputs using flatpickr and prefill values
  if (typeof flatpickr !== "undefined" && document.getElementById("load-start-picker")) {
    flatpickr("#load-start-picker", {
      enable: datesIso,
      dateFormat: "Y-m-d",
      allowInput: true,
    });
    flatpickr("#load-end-picker", {
      enable: datesIso,
      dateFormat: "Y-m-d",
      allowInput: true,
    });
  }

  // disable load controls when not using DB backend
  if (!USE_DATABASE) {
    const p = document.getElementById("load-start-picker");
    if (p) p.disabled = true;
    const q = document.getElementById("load-end-picker");
    if (q) q.disabled = true;
    const b = document.getElementById("load-data");
    if (b) b.disabled = true;
  }

// Initial load - wait for DOM and date picker to be ready
window.addEventListener("load", () => {
  // Small delay to ensure flatpickr is initialized
  setTimeout(() => {
    const dateRaw = document.getElementById("date-raw").value;
    if (dateRaw) {
      window.currentDate = dateRaw;
      console.log("Initial load with date:", dateRaw);
      fetchVelocityMap();
    }
  }, 100);
});
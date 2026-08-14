const $ = (id) => document.getElementById(id);

function riskLabel(value) {
  if (value < 20) return "nominal";
  if (value < 40) return "elevated";
  if (value < 70) return "high";
  return "critical";
}

function setGauge(value) {
  const arc = $("gauge-arc");
  const offset = 157 - (Math.min(100, Math.max(0, value)) / 100) * 157;
  arc.setAttribute("stroke-dashoffset", String(offset));
  arc.setAttribute(
    "stroke",
    value < 35 ? "#00e5a8" : value < 70 ? "#ffb020" : "#ff3b5c"
  );
  $("risk-value").textContent = value.toFixed(1);
  $("risk-label").textContent = riskLabel(value);
}

async function refreshStats() {
  try {
    const res = await fetch("/api/v1/stats");
    const data = await res.json();
    $("stat-fps").textContent = Number(data.fps).toFixed(1);
    $("stat-ms").textContent = `${Number(data.inference_ms).toFixed(1)} ms`;
    $("stat-frames").textContent = data.frames;
    $("stat-events").textContent = data.events;
    $("device-pill").textContent = `device ${data.device || "—"}`;
    $("backend-pill").textContent = `backend ${data.backend || "—"}`;
    $("frame-meta").textContent = `${data.frames} frames · ${Number(data.fps).toFixed(1)} fps`;
    setGauge(Number(data.risk) || 0);
  } catch (err) {
    $("frame-meta").textContent = "stats unavailable";
  }
}

function prependEvent(event) {
  const li = document.createElement("li");
  const sev = document.createElement("span");
  sev.className = `sev ${event.severity}`;
  sev.textContent = event.severity.toUpperCase();
  const body = document.createElement("div");
  body.innerHTML = `<strong>${event.name}</strong><div class="muted">${event.message || event.category}</div>`;
  const conf = document.createElement("span");
  conf.className = "muted";
  conf.textContent = `${(event.confidence * 100).toFixed(0)}%`;
  li.append(sev, body, conf);
  const list = $("events");
  list.prepend(li);
  while (list.children.length > 40) list.removeChild(list.lastChild);
}

function listenEvents() {
  const src = new EventSource("/api/v1/events/stream");
  src.onmessage = (msg) => {
    try {
      prependEvent(JSON.parse(msg.data));
    } catch (_) {
      /* ignore malformed */
    }
  };
}

$("upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  $("infer-out").textContent = "running…";
  const res = await fetch("/api/v1/infer/image", { method: "POST", body });
  const data = await res.json();
  $("infer-out").textContent = JSON.stringify(data, null, 2);
});

refreshStats();
setInterval(refreshStats, 1000);
listenEvents();

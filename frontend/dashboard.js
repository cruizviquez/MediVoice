function riskBadgeClass(risk) {
  if (risk === "high") return "bad";
  if (risk === "medium") return "warn";
  return "good";
}

let historyItems = [];
const overrides = new Map();
let selectedId = null;

const elQueueList = document.getElementById("queueList");
const elDetails = document.getElementById("details");
const elDetailsEmpty = document.getElementById("detailsEmpty");
const elCaseCount = document.getElementById("caseCount");
const elBackendStatus = document.getElementById("backendStatus");
const elActionMsg = document.getElementById("actionMsg");

function setBackendStatus(ok) {
  elBackendStatus.textContent = ok ? "Backend: OK" : "Backend: Offline";
  elBackendStatus.style.borderColor = ok ? "rgba(40, 167, 69, 0.45)" : "rgba(220, 53, 69, 0.45)";
}

async function pingHealth() {
  try {
    const r = await fetch("/health");
    setBackendStatus(r.ok);
  } catch {
    setBackendStatus(false);
  }
}

function renderQueue() {
  elQueueList.innerHTML = "";

  elCaseCount.textContent = `${historyItems.length} cases`;

  if (historyItems.length === 0) {
    const empty = document.createElement("div");
    empty.className = "emptyState";
    empty.textContent = "No history yet. Run a demo intake to see cases.";
    elQueueList.appendChild(empty);
    return;
  }

  for (const entry of historyItems) {
    const detail = overrides.get(entry.id)?.cached || null;
    const intent = entry.intent || detail?.result?.intent || "unknown";
    const summary = detail?.result?.pharmacist_task?.summary
      || detail?.result?.recommended_next_step
      || (entry.transcript ? entry.transcript.slice(0, 64) + (entry.transcript.length > 64 ? "..." : "") : "New intake");
    const risk = detail?.result?.risk_level || "";
    const time = entry.created_at ? new Date(entry.created_at).toLocaleString() : "";

    const row = document.createElement("div");
    row.className = "queueItem" + (entry.id === selectedId ? " active" : "");
    row.innerHTML = `
      <div class="row">
        <div class="big">${summary}</div>
        ${risk ? `<div class="badge ${riskBadgeClass(risk)}">${risk}</div>` : ""}
      </div>
      <div class="row" style="margin-top:6px">
        <div class="small">${intent}</div>
        <div class="small">${time}</div>
      </div>
    `;

    row.addEventListener("click", () => {
      selectedId = entry.id;
      renderQueue();
      renderDetails();
    });

    elQueueList.appendChild(row);
  }
}

function applyOverrides(entry) {
  const override = overrides.get(entry.id) || {};
  const result = entry.result || {};
  const task = { ...(result.pharmacist_task || {}) };
  const tags = new Set(task.tags || []);

  if (override.status === "resolved") tags.add("resolved");
  if (override.status === "follow_up") tags.add("follow_up");
  if (override.status === "escalated") tags.add("manual_escalation");

  let riskLevel = result.risk_level;
  if (override.status === "escalated") {
    riskLevel = "high";
    task.queue = "urgent_escalation";
    task.priority = "urgent";
    task.due_in_hours = 1;
  }

  return {
    ...entry,
    result: {
      ...result,
      risk_level: riskLevel,
      pharmacist_task: {
        ...task,
        tags: Array.from(tags)
      }
    }
  };
}

async function fetchHistory() {
  try {
    const res = await fetch("/history");
    if (!res.ok) throw new Error("Failed to load history");
    const payload = await res.json();
    historyItems = Array.isArray(payload.items) ? payload.items : [];
  } catch (err) {
    historyItems = [];
  }
}

async function fetchEntry(id) {
  const cached = overrides.get(id)?.cached;
  if (cached) return cached;
  const res = await fetch(`/history/${id}`);
  if (!res.ok) throw new Error("Failed to load entry");
  const entry = await res.json();
  const existing = overrides.get(id) || {};
  overrides.set(id, { ...existing, cached: entry });
  return entry;
}

async function renderDetails() {
  if (!selectedId) {
    elDetails.classList.add("hidden");
    elDetailsEmpty.classList.remove("hidden");
    return;
  }
  elDetails.classList.remove("hidden");
  elDetailsEmpty.classList.add("hidden");

  let entry;
  try {
    entry = await fetchEntry(selectedId);
  } catch (err) {
    elDetailsEmpty.textContent = "Failed to load case details.";
    elDetails.classList.add("hidden");
    elDetailsEmpty.classList.remove("hidden");
    return;
  }

  const effective = applyOverrides(entry);
  const r = effective.result || {};
  const task = r.pharmacist_task || {};
  const soap = r.soap_note || {};

  document.getElementById("badgeIntent").textContent = `intent: ${r.intent || "unknown"}`;
  const risk = r.risk_level || "low";
  const badgeRisk = document.getElementById("badgeRisk");
  badgeRisk.textContent = `risk: ${risk}`;
  badgeRisk.className = `badge ${riskBadgeClass(risk)}`;

  const status = overrides.get(selectedId)?.status || "open";
  document.getElementById("badgeStatus").textContent = `status: ${status}`;

  document.getElementById("nextStep").textContent = r.recommended_next_step || "(none)";

  document.getElementById("soapS").textContent = soap.subjective || "";
  document.getElementById("soapO").textContent = soap.objective || "Not available (voice-only intake).";
  document.getElementById("soapA").textContent = soap.assessment || "";
  document.getElementById("soapP").textContent = soap.plan || "";

  const taskSummary = task.summary || "(none)";
  const taskMeta = `${task.queue || "queue"} • ${task.priority || "normal"} • due ${task.due_in_hours ?? "?"}h`;
  document.getElementById("taskSummary").textContent = taskSummary;
  document.getElementById("taskMeta").textContent = taskMeta;

  const taskTags = document.getElementById("taskTags");
  taskTags.innerHTML = "";
  const tags = task.tags || [];
  if (tags.length === 0) {
    const span = document.createElement("span");
    span.className = "small";
    span.textContent = "No tags.";
    taskTags.appendChild(span);
  } else {
    for (const tag of tags) {
      const badge = document.createElement("span");
      badge.className = "badge ghost";
      badge.textContent = tag;
      taskTags.appendChild(badge);
    }
  }
}
document.getElementById("btnResolve").addEventListener("click", () => {
  if (!selectedId) return;
  const existing = overrides.get(selectedId) || {};
  overrides.set(selectedId, { ...existing, status: "resolved" });
  elActionMsg.textContent = "Case marked as resolved.";
  renderQueue();
  renderDetails();
});

document.getElementById("btnFollowup").addEventListener("click", () => {
  if (!selectedId) return;
  const existing = overrides.get(selectedId) || {};
  overrides.set(selectedId, { ...existing, status: "follow_up" });
  elActionMsg.textContent = "Follow-up task created.";
  renderQueue();
  renderDetails();
});

document.getElementById("btnEscalate").addEventListener("click", () => {
  if (!selectedId) return;
  const existing = overrides.get(selectedId) || {};
  overrides.set(selectedId, { ...existing, status: "escalated" });
  elActionMsg.textContent = "Case escalated.";
  renderQueue();
  renderDetails();
});

// init
(async function init(){
  await pingHealth();
  setInterval(pingHealth, 8000);
  await fetchHistory();
  if (historyItems.length > 0) {
    selectedId = historyItems[0].id;
  }
  renderQueue();
  await renderDetails();
})();

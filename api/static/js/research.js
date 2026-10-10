// Research Mode: Läufe starten, verfolgen, abbrechen, fortsetzen; Berichte anzeigen.
import { escapeHtml } from "./escape.js";

const ACTIVE = new Set(["created", "queued", "running"]);
const STATUS_LABEL = {
  created: "Starting",
  queued: "Queued",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  cancelled: "Cancelled",
  failed: "Failed",
  budget_exhausted: "Budget used up",
};

export function hasActiveRun(runs) {
  return runs.some((r) => ACTIVE.has(r.status));
}

export function parseSeedUrls(text) {
  return String(text || "")
    .split(/\s+/)
    .map((u) => u.trim())
    .filter((u) => /^https?:\/\//i.test(u))
    .slice(0, 50);
}

export function formatRun(run) {
  const c = run.counts || {};
  const minutes = run.active_seconds ? `${Math.round(run.active_seconds / 6) / 10} min` : "0 min";
  const budget = run.budget ? ` of ${Math.round(run.budget.max_duration_s / 60)} min` : "";
  return `${c.sources || 0} sources · ${c.claims || 0} statements · ${c.findings || 0} findings · ${c.errors || 0} errors · ${minutes}${budget}`;
}

export function renderRuns(runs) {
  if (!runs.length) return `<div class="muted">No research runs yet.</div>`;
  return runs
    .map((r) => {
      const actions = [`<button class="ghost-btn" data-research-report="${escapeHtml(r.id)}">Report</button>`];
      if (ACTIVE.has(r.status)) actions.push(`<button class="ghost-btn danger" data-research-cancel="${escapeHtml(r.id)}">Cancel</button>`);
      if (r.status === "paused" || r.status === "cancelled" || r.status === "failed") actions.push(`<button class="ghost-btn" data-research-resume="${escapeHtml(r.id)}">Resume</button>`);
      const state = ACTIVE.has(r.status) ? "warn" : r.status === "completed" ? "ok" : r.status === "failed" ? "err" : "unknown";
      return `<div class="model-row research-run"><span class="status-dot" data-state="${state}"></span><div class="research-run-body">
<div class="model-name">${escapeHtml(r.objective)}</div>
<div class="model-sub">${escapeHtml(STATUS_LABEL[r.status] || r.status)}${r.stop_reason ? ` – ${escapeHtml(r.stop_reason)}` : ""}</div>
<div class="model-sub">${escapeHtml(formatRun(r))}${r.model ? ` · model ${escapeHtml(r.model)}` : " · no model"}</div>
<div class="desktop-actions">${actions.join("")}</div></div></div>`;
    })
    .join("");
}

export function renderResearchForm(status, modelAvailable) {
  const provider = status.search_provider;
  const providerLine = provider
    ? `<p class="muted">Web search: <b>${escapeHtml(provider)}</b>. Pages are fetched politely (robots.txt, rate limits); web content is treated as untrusted data.</p>`
    : `<p class="muted">No web search provider is configured${status.search_problem ? ` (${escapeHtml(status.search_problem)})` : ""}. Research can still use the URLs you list below, or configure a self-hosted SearXNG instance:</p>
<form id="research-config" class="research-config"><input name="url" type="url" placeholder="http://127.0.0.1:8888" required><button class="ghost-btn" type="submit">Use SearXNG</button></form>`;
  const modelLine = modelAvailable
    ? ""
    : `<p class="setup-problem">No local model is available: NOVA will collect and rate sources but cannot extract or compare statements.</p>`;
  return `<div class="panel"><h3>New research</h3>${providerLine}${modelLine}
<form id="research-form" class="research-form">
<label>Objective<textarea name="objective" rows="3" minlength="5" maxlength="2000" required placeholder="e.g. Compare the energy efficiency of heat pumps and gas boilers in Germany"></textarea></label>
<div class="research-grid"><label>Time budget (minutes)<input name="minutes" type="number" min="1" max="1440" value="60"></label>
<label>Max. sources<input name="sources" type="number" min="1" max="500" value="30"></label></div>
<label>Start URLs (optional, one per line)<textarea name="urls" rows="2" placeholder="https://…"></textarea></label>
<button class="primary-btn" type="submit">Start research</button>
<div class="setup-problem" id="research-error" hidden></div>
</form></div>`;
}

export function formToRequest(form) {
  const data = new FormData(form);
  return {
    objective: String(data.get("objective") || "").trim(),
    duration_minutes: Number(data.get("minutes") || 60),
    max_sources: Number(data.get("sources") || 30),
    seed_urls: parseSeedUrls(data.get("urls")),
  };
}

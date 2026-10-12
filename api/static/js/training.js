// Training: Umgebung, Datensätze, Jobs, Evaluation, Freigabe/Aktivierung.
import { escapeHtml } from "./escape.js";

export function jobActive(job) {
  const t = (job.training || {}).status;
  const e = (job.evaluation || {}).status;
  return job.running || ["loading", "training", "saving"].includes(t) || ["loading", "evaluating"].includes(e);
}

function pct(done, total) {
  return total ? Math.floor((done / total) * 100) : 0;
}

export function renderEnv(env) {
  const setup = env.setup || {};
  const probe = env.probe;
  if (["starting", "installing"].includes(setup.status)) {
    return `<p>Setting up the training environment… <b>${escapeHtml(setup.step || "")}</b></p><p class="muted">This downloads Python, PyTorch and the training libraries (several GB) into NOVA's data folder.</p>`;
  }
  let html = setup.status === "failed" ? `<div class="setup-problem">${escapeHtml(setup.error || "Setup failed")}</div>` : "";
  if (!env.ready) {
    return `${html}<p class="muted">Training runs in a separate environment (Python, PyTorch with CUDA 12.6, transformers, PEFT). NOVA installs it from the official sources.</p><button class="primary-btn" id="training-setup">Set up training environment</button>`;
  }
  if (probe) {
    const gpu = probe.gpu ? `${escapeHtml(probe.gpu)} · ${escapeHtml(String(probe.vram_gb))} GB · compute ${escapeHtml(probe.compute_capability || "?")}` : "CPU only (very slow)";
    html += `<p><span class="tag ${probe.ready ? "measured" : "unmeasured"}">${probe.ready ? "ready" : "not ready"}</span> ${gpu} · PyTorch ${escapeHtml(probe.torch || "–")} · compute type ${escapeHtml(probe.compute_dtype || "–")}</p>`;
    html += (probe.notes || []).map((n) => `<div class="setup-warning">${escapeHtml(n)}</div>`).join("");
    html += (probe.problems || []).map((p) => `<div class="setup-problem">${escapeHtml(p)}</div>`).join("");
  }
  return `${html}<button class="ghost-btn" id="training-probe">Check hardware again</button>`;
}

export function renderDatasets(datasets) {
  const list = datasets.length
    ? datasets.map((d) => `<div class="model-sub"><code>${escapeHtml(d.name)}@${escapeHtml(d.version)}</code> · train ${d.counts.train} · val ${d.counts.val} · test ${d.counts.test}</div>`).join("")
    : `<p class="muted">No dataset yet.</p>`;
  return `${list}<p class="muted">A dataset is built from <b>supported</b> research findings only (disputed, unverified, outdated and terms-restricted sources are excluded). Each version is immutable.</p><button class="ghost-btn" id="training-dataset">Build dataset from research knowledge</button>`;
}

export function renderStart(status) {
  if (!status.env.ready || !status.datasets.length) return `<p class="muted">Needs a ready training environment and a dataset.</p>`;
  const datasets = status.datasets.map((d) => `<option value="${escapeHtml(`${d.name}@${d.version}`)}">${escapeHtml(`${d.name}@${d.version}`)}</option>`).join("");
  const choices = status.choices
    .map((c) => `<option value="${escapeHtml(c.id)}" ${c.id === status.installed_model ? "selected" : ""}>${escapeHtml(c.label)} – trains ${escapeHtml(c.repo)}</option>`)
    .join("");
  return `<form id="training-form" class="research-form">
<label>Dataset<select name="dataset">${datasets}</select></label>
<label>Base model<select name="choice">${choices}</select></label>
<label>Epochs<input name="epochs" type="number" min="0.5" max="20" step="0.5" value="2"></label>
<p class="muted">The base model must match the model NOVA runs, otherwise the result cannot be activated. During GPU training the chat pauses (the GPU is needed for training) and resumes automatically.</p>
<button class="primary-btn" type="submit">Start training</button></form>`;
}

export function renderJob(job, activeAdapter) {
  const t = job.training || {};
  const e = job.evaluation || {};
  const r = job.report;
  let state = escapeHtml(t.status || "queued");
  if (t.status === "training" && t.max_steps) state += ` · step ${t.step}/${t.max_steps} (${pct(t.step, t.max_steps)} %)${t.loss != null ? ` · loss ${t.loss}` : ""}${t.eta_s ? ` · ~${Math.ceil(t.eta_s / 60)} min left` : ""}`;
  if (t.status === "completed" && e.status === "evaluating") state = `evaluating · ${e.done || 0}/${e.total || "?"}`;
  if (t.error) state += ` – ${escapeHtml(t.error)}`;
  if (e.error) state += ` – evaluation: ${escapeHtml(e.error)}`;
  let html = `<div class="model-row research-run"><div class="research-run-body"><div class="model-name">${escapeHtml(job.id)}</div>
<div class="model-sub">${escapeHtml(job.dataset || "")} · ${escapeHtml(job.base_model || "")} · ${escapeHtml(job.method || "")}</div><div class="model-sub">${state}</div>`;
  if (r) {
    html += `<table class="eval-table"><tr><th></th><th>Base</th><th>Trained</th></tr>
<tr><td>Token-F1 vs. reference</td><td>${r.base.f1.toFixed(3)}</td><td>${r.tuned.f1.toFixed(3)}</td></tr>
<tr><td>Cites a source</td><td>${Math.round(r.base.cited_rate * 100)} %</td><td>${Math.round(r.tuned.cited_rate * 100)} %</td></tr>
<tr><td>General checks</td><td>${r.base.regression_passed}/10</td><td>${r.tuned.regression_passed}/10</td></tr></table>
<div class="${r.gate.eligible ? "setup-warning" : "setup-problem"}">Gate: ${r.gate.eligible ? "eligible for approval" : "not eligible"} – ${escapeHtml(r.gate.reasons.join("; "))}</div>`;
  }
  const actions = [];
  if (jobActive(job)) actions.push(`<button class="ghost-btn danger" data-training-cancel="${escapeHtml(job.id)}">Cancel</button>`);
  if (r && r.gate.eligible && !job.approval) actions.push(`<button class="primary-btn" data-training-approve="${escapeHtml(job.id)}">Approve and activate</button>`);
  if (job.approval) {
    const active = activeAdapter && activeAdapter.endsWith(job.id.slice(0, 15));
    actions.push(active ? `<span class="tag measured">active NOVA model</span><button class="ghost-btn" id="training-deactivate">Back to base model</button>` : `<span class="tag">approved ${escapeHtml(job.approval.approved_at)}</span>`);
  }
  html += `<div class="desktop-actions">${actions.join("")}</div></div></div>`;
  return html;
}

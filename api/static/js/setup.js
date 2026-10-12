// Ein-Klick-Einrichtung (llama.cpp + empfohlenes Modell) im Setup-Bereich.
import { escapeHtml } from "./escape.js";
import { formatBytes } from "./format.js";

export const SETUP_ACTIVE = new Set(["starting", "downloading", "extracting", "starting_runtime"]);
const STEP = {
  starting: "Preparing…",
  downloading: "Downloading",
  extracting: "Unpacking runtime…",
  starting_runtime: "Starting the model (this can take a minute)…",
};

export function hardwareLine(status) {
  const gpu = status.gpu;
  const ram = status.ram_gb ? ` · ${status.ram_gb} GB RAM` : "";
  if (gpu) return `GPU: <b>${escapeHtml(gpu.name)}</b> (${escapeHtml(String(gpu.vram_gb))} GB VRAM${gpu.compute_capability ? `, compute ${escapeHtml(gpu.compute_capability)}` : ""})${ram}`;
  return `No NVIDIA GPU detected – NOVA will run on the CPU (slower)${ram}`;
}

export function renderOneClick(status) {
  const setup = status.setup || {};
  if (!status.platform_supported) {
    return `<p class="muted">The one-click setup is available in the Windows app. On this system, install a runtime manually (see below).</p>`;
  }
  if (status.custom_config) {
    return `<p class="muted">You use your own <code>models.toml</code> – NOVA does not change it. Remove it to use the one-click setup.</p>`;
  }
  if (SETUP_ACTIVE.has(setup.status)) {
    const pct = setup.total ? Math.floor((setup.done / setup.total) * 100) : 0;
    const bytes = setup.total ? `${formatBytes(setup.done)} / ${formatBytes(setup.total)} (${pct} %)` : "";
    return `<p>${escapeHtml(STEP[setup.status] || setup.status)} ${setup.item ? `<code>${escapeHtml(setup.item)}</code>` : ""}</p>
<div class="progress"><div class="progress-bar" data-pct="${pct}"></div></div><div class="muted">${escapeHtml(bytes)}</div>`;
  }
  const failed = setup.status === "failed" ? `<div class="setup-problem">${escapeHtml(setup.error || "Setup failed")}</div>` : "";
  if (status.installed) {
    const state = status.running ? `<span class="tag measured">running</span>` : `<span class="tag unmeasured">stopped</span>`;
    return `${failed}<p>${state} <b>${escapeHtml(status.installed.label || status.installed.file)}</b> on llama.cpp ${escapeHtml(status.runtime_tag || "")}</p>
<p class="muted">${hardwareLine(status)}</p><button class="ghost-btn" id="runtime-restart">Restart model runtime</button>`;
  }
  const rec = status.recommended || {};
  return `${failed}<p class="muted">${hardwareLine(status)}</p>
<p>Recommended: <b>${escapeHtml(rec.label || "")}</b></p>
<p class="muted">NOVA downloads the llama.cpp runtime and the model from their official sources (GitHub, Hugging Face), checks every file against its published SHA-256 and starts it locally. Download size: about 6 GB for the 9B model, about 3 GB for the 4B model, less for the 2B model. Progress and exact size appear during the download.</p>
<label class="check"><input type="checkbox" id="runtime-accept"><span>I accept the model license (${escapeHtml(rec.base_license || "see model page")}) – <a href="https://huggingface.co/${escapeHtml(rec.repo || "")}" target="_blank" rel="noopener noreferrer">model page</a></span></label>
<button class="primary-btn" id="runtime-setup" data-choice="${escapeHtml(rec.id || "")}">Set up NOVA automatically</button>`;
}

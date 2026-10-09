// Integration mit der NOVA-Desktop-App (pywebview). Im normalen Browser inaktiv.
import { escapeHtml } from "./escape.js";

let bridge = null;

export function isDesktop() {
  return bridge !== null;
}

// Bridge erkennen: pywebview stellt window.pywebview.api bereit (ggf. erst nach "pywebviewready").
export function initDesktop(onReady, win = window) {
  const ready = () => {
    if (!win.pywebview || !win.pywebview.api || bridge) return;
    bridge = win.pywebview.api;
    win.document.documentElement.dataset.desktop = "1";
    if (onReady) onReady();
  };
  ready();
  win.addEventListener("pywebviewready", ready);
}

export function resetDesktopForTests() {
  bridge = null;
}

export async function desktopPanel() {
  if (!bridge) return "";
  let info;
  try {
    info = await bridge.info();
  } catch (err) {
    return `<div class="panel"><h3>Desktop app</h3><div class="error-card"><div class="err-title">${escapeHtml(String(err && err.message ? err.message : err))}</div></div></div>`;
  }
  const core = info.core || {};
  const rows = [
    ["App version", info.version],
    ["Core service", core.status],
    ["Core URL", core.url],
    ["Started by this window", core.started_by_desktop ? "yes" : "no"],
    ["Logs", core.log],
  ]
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`)
    .join("");
  return `<div class="panel" id="desktop-panel"><h3>Desktop app</h3><dl class="kv">${rows}</dl>
<label class="check spaced"><input type="checkbox" data-desktop-keep ${info.keep_core_running ? "checked" : ""}><span>Keep the core service running when this window closes (needed for integrations such as IC WARE HQ)</span></label>
<div class="desktop-actions">
<button class="ghost-btn" data-desktop-action="restart_core">Restart core</button>
<button class="ghost-btn" data-desktop-action="open_in_browser">Open in browser</button>
<button class="ghost-btn" data-desktop-action="open_data_folder">Data folder</button>
<button class="ghost-btn" data-desktop-action="open_logs">Logs</button>
<button class="ghost-btn danger" data-desktop-action="stop_core_and_quit">Stop core and quit</button>
</div></div>`;
}

const ACTIONS = new Set(["restart_core", "open_in_browser", "open_data_folder", "open_logs", "stop_core_and_quit"]);

// Klick/Change im Status-Panel; liefert true, wenn das Ereignis zur Desktop-Steuerung gehörte.
export async function handleDesktopEvent(e, refresh) {
  if (!bridge) return false;
  const keep = e.target.closest && e.target.closest("[data-desktop-keep]");
  if (keep && e.type === "change") {
    await bridge.set_keep_core_running(keep.checked);
    return true;
  }
  const btn = e.target.closest && e.target.closest("[data-desktop-action]");
  if (!btn || e.type !== "click") return false;
  const action = btn.dataset.desktopAction;
  if (!ACTIONS.has(action)) return false;
  if (action === "stop_core_and_quit" && !window.confirm("Stop the NOVA core service and close the app?")) return true;
  btn.disabled = true;
  try {
    await bridge[action]();
  } finally {
    btn.disabled = false;
  }
  if (action === "restart_core" && refresh) await refresh();
  return true;
}

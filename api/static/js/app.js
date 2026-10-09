// NOVA UI – Controller. Spricht ausschließlich mit der NOVA API (api.js).

import { api, ApiError, setToken } from "./api.js";
import { renderMarkdown, escapeHtml } from "./markdown.js";
import { responseStats, liveLine, dayGroup, formatBytes, formatDuration, categoryLabel } from "./format.js";

const $ = (id) => document.getElementById(id);
const MAX_FILE = 10 * 1024 * 1024;

const state = {
  conversations: [],
  currentId: null,
  messages: [],
  settings: null,
  models: [],
  modelStatus: null,
  system: null,
  attachments: [],
  running: null, // { runId, controller, el, state }
  search: "",
};

// ----------------------------------------------------------------- Hilfen

function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function toast(message, kind = "") {
  const t = el("div", `toast ${kind}`);
  t.textContent = message;
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "err" ? 7000 : 3500);
}

function newRunId() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function scrollToBottom(force = false) {
  const box = $("messages");
  const near = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
  if (force || near) box.scrollTop = box.scrollHeight;
}

function handleError(err) {
  if (err instanceof ApiError && err.status === 401) {
    $("token-modal").hidden = false;
    return;
  }
  toast(err.message || String(err), "err");
}

// ----------------------------------------------------------------- Status

function hasModel() {
  return Boolean(state.modelStatus && state.modelStatus.any_available);
}

function renderStatusIndicator() {
  const dot = $("status-dot");
  const label = $("status-label");
  const s = state.modelStatus;
  if (!s) {
    dot.dataset.state = "err";
    label.textContent = "API not reachable";
    return;
  }
  if (s.any_available) {
    dot.dataset.state = "ok";
    label.textContent = `${s.available_count} model${s.available_count === 1 ? "" : "s"} available`;
  } else {
    dot.dataset.state = "warn";
    label.textContent = "No local model available";
  }
  const banner = $("banner");
  if (state.system && state.system.dev_mode) {
    banner.hidden = false;
    banner.textContent = "Development mode" + (s.any_available ? "" : " – no local model available. Chat is disabled until a model runtime is configured.");
  } else {
    banner.hidden = true;
  }
  const noModel = $("no-model");
  noModel.hidden = s.any_available;
  if (!s.any_available) {
    const reasons = [];
    if (s.config_error) reasons.push(s.config_error);
    if (!s.models.length && !s.config_error) reasons.push("No models configured.");
    for (const m of s.models) reasons.push(`${m.name}: ${m.reason}`);
    $("no-model-text").textContent = reasons.join("\n") || "Start a local runtime (e.g. llama-server) and configure it in models.toml.";
  }
  updateComposer();
}

function renderModelChip() {
  const s = state.settings;
  $("model-chip-value").textContent = !s || s.model === "auto" ? "Auto" : s.model;
}

async function refreshStatus(refresh = false) {
  try {
    const [system, modelStatus] = await Promise.all([api.systemStatus(), api.modelsStatus(refresh)]);
    state.system = system;
    state.modelStatus = modelStatus;
  } catch (err) {
    state.modelStatus = null;
    if (err instanceof ApiError && err.status === 401) handleError(err);
  }
  renderStatusIndicator();
}

// ----------------------------------------------------------------- Sidebar

function renderConversations() {
  const nav = $("conversations");
  nav.innerHTML = "";
  const q = state.search.trim().toLowerCase();
  const items = state.conversations.filter((c) => !q || c.title.toLowerCase().includes(q));
  if (!items.length) {
    nav.append(el("div", "conv-empty", q ? "No matching chats" : "No chats yet"));
    return;
  }
  let group = null;
  for (const c of items) {
    const g = dayGroup(c.updated_at);
    if (g !== group) {
      group = g;
      nav.append(el("div", "conv-group", g));
    }
    const row = el("div", `conv${c.id === state.currentId ? " active" : ""}`);
    row.dataset.id = c.id;
    row.tabIndex = 0;
    const title = el("span", "conv-title");
    title.textContent = c.title;
    const actions = el(
      "span",
      "conv-actions",
      '<button data-act="rename" title="Rename" aria-label="Rename"><svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg></button>' +
        '<button data-act="delete" title="Delete" aria-label="Delete"><svg viewBox="0 0 24 24"><path d="M5 7h14M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button>',
    );
    row.append(title, actions);
    nav.append(row);
  }
}

async function loadConversations() {
  try {
    state.conversations = (await api.conversations()).conversations;
  } catch (err) {
    handleError(err);
  }
  renderConversations();
}

async function openConversation(id) {
  if (state.running) return toast("Stop the current generation first.");
  try {
    const conv = await api.conversation(id);
    state.currentId = id;
    state.messages = conv.messages;
    $("chat-title").textContent = conv.title;
    renderMessages();
    renderConversations();
    closeSidebar();
    scrollToBottom(true);
  } catch (err) {
    handleError(err);
  }
}

function newChat() {
  if (state.running) return toast("Stop the current generation first.");
  state.currentId = null;
  state.messages = [];
  state.attachments = [];
  $("chat-title").textContent = "New chat";
  renderAttachments();
  renderMessages();
  renderConversations();
  closeSidebar();
  $("input").focus();
}

// ----------------------------------------------------------------- Nachrichten

function statsHtml(meta) {
  const stats = responseStats(meta);
  if (!stats.length) return "";
  return `<div class="stats">${stats
    .map((s) => `<span class="stat ${s.tone || ""}">${escapeHtml(s.label)} <b>${escapeHtml(s.value)}</b></span>`)
    .join("")}</div>`;
}

function attachmentsHtml(list) {
  if (!list || !list.length) return "";
  return `<div class="att-list">${list
    .map((a) =>
      a.kind === "image" && a.file
        ? `<img class="att-thumb" src="/uploads/${encodeURIComponent(a.file)}" alt="${escapeHtml(a.name)}">`
        : `<span class="att-pill">${escapeHtml(a.name)} · ${formatBytes(a.size)}</span>`,
    )
    .join("")}</div>`;
}

function userNode(message) {
  const node = el("div", "msg msg-user");
  if (message.content) {
    const bubble = el("div", "bubble");
    bubble.textContent = message.content;
    node.append(bubble);
  }
  node.insertAdjacentHTML("beforeend", attachmentsHtml(message.attachments));
  return node;
}

function assistantBody(message) {
  const meta = message.meta || {};
  let html = "";
  if (meta.error) {
    const detail = typeof meta.error.detail === "string" ? meta.error.detail : meta.error.detail ? JSON.stringify(meta.error.detail, null, 2) : "";
    html += `<div class="error-card"><div class="err-title">${escapeHtml(meta.error.message)}</div>${
      detail ? `<div class="err-detail">${escapeHtml(detail)}</div>` : ""
    }</div>`;
  }
  if (message.content) html += `<div class="md">${renderMarkdown(message.content)}</div>`;
  if (meta.stopped) html += `<div class="stopped-tag">Generation stopped</div>`;
  if (meta.verification && meta.verification.summary) {
    html += `<div class="routing-reason">${escapeHtml(meta.verification.summary)}</div>`;
  }
  html += statsHtml(meta);
  if (state.settings && state.settings.show_routing && meta.routing && meta.routing.reason) {
    html += `<div class="routing-reason"><span class="reason-label">Routing reason</span>${escapeHtml(meta.routing.reason)}</div>`;
  }
  return html;
}

function assistantNode(message) {
  const node = el("div", "msg msg-assistant");
  node.innerHTML = `<div class="avatar" aria-hidden="true">N</div><div class="msg-body">${assistantBody(message)}</div>`;
  return node;
}

function renderMessages() {
  const box = $("messages");
  box.querySelectorAll(".msg").forEach((n) => n.remove());
  $("empty").hidden = state.messages.length > 0;
  for (const m of state.messages) box.append(m.role === "user" ? userNode(m) : assistantNode(m));
}

function liveHtml(run) {
  const parts = liveLine(run.state)
    .map((p) => (p.label === "Status" ? `<span><span class="pulse"></span>${p.label}: <b>${escapeHtml(p.value)}</b></span>` : `<span>${p.label}: <b>${escapeHtml(p.value)}</b></span>`))
    .join("");
  const text = run.state.text;
  return `<div class="live">${parts}</div>${text ? `<div class="md caret">${renderMarkdown(text)}</div>` : ""}`;
}

let renderScheduled = false;
function renderLive() {
  if (renderScheduled) return;
  renderScheduled = true;
  requestAnimationFrame(() => {
    renderScheduled = false;
    const run = state.running;
    if (!run) return;
    run.el.querySelector(".msg-body").innerHTML = liveHtml(run);
    scrollToBottom();
  });
}

// ----------------------------------------------------------------- Senden

function updateComposer() {
  const send = $("send");
  const running = Boolean(state.running);
  send.classList.toggle("stop", running);
  send.title = running ? "Stop generation" : "Send (Enter)";
  send.setAttribute("aria-label", running ? "Stop generation" : "Send");
  const text = $("input").value.trim();
  send.disabled = !running && (!hasModel() || (!text && !state.attachments.length));
  const agentAllowed = Boolean(state.settings && state.settings.agent_workspace);
  $("agent-toggle-wrap").classList.toggle("disabled", !agentAllowed);
  $("agent-toggle").disabled = !agentAllowed;
  if (!agentAllowed) $("agent-toggle").checked = false;
  $("agent-toggle-wrap").title = agentAllowed ? "Agent mode: plan, use tools, verify" : "Agent mode needs a workspace (Settings)";
  $("input").placeholder = hasModel() ? "Message NOVA…" : "No local model available";
}

async function send() {
  if (state.running) return stop();
  const input = $("input");
  const text = input.value.trim();
  if (!hasModel()) return toast("No local model available.", "err");
  if (!text && !state.attachments.length) return;
  const agent = $("agent-toggle").checked;
  if (agent && state.attachments.length) return toast("Agent mode does not take attachments yet.", "err");

  const runId = newRunId();
  const controller = new AbortController();
  const attachments = state.attachments.map(({ name, mime, data_b64 }) => ({ name, mime, data_b64 }));
  const pending = { role: "user", content: text, attachments: state.attachments.map((a) => ({ name: a.name, size: a.size, kind: a.kind })) };
  $("empty").hidden = true;
  const userEl = userNode(pending);
  $("messages").append(userEl);
  const node = assistantNode({ content: "", meta: {} });
  $("messages").append(node);
  state.running = { runId, controller, el: node, state: { status: agent ? "Starting agent…" : "Routing…", text: "" } };
  input.value = "";
  autosize();
  state.attachments = [];
  renderAttachments();
  updateComposer();
  renderLive();
  scrollToBottom(true);

  const body = agent
    ? { task: text, conversation_id: state.currentId, run_id: runId }
    : { message: text, conversation_id: state.currentId, attachments, run_id: runId };
  let finalMessage = null;
  try {
    const call = agent ? api.agentStream : api.chatStream;
    await call(body, (name, data) => onEvent(name, data, userEl, (m) => (finalMessage = m)), controller.signal);
  } catch (err) {
    if (err && err.name === "AbortError") {
      // Verbindung bewusst getrennt – Server speichert den Teiltext als abgebrochen
    } else {
      node.querySelector(".msg-body").innerHTML = assistantBody({ content: state.running?.state.text || "", meta: { error: { message: err.message, detail: err.detail } } });
      if (!(err instanceof ApiError && err.status === 401)) toast(err.message, "err");
      else handleError(err);
      if (err instanceof ApiError && err.code === "no_model") refreshStatus(true);
    }
  } finally {
    const run = state.running;
    state.running = null;
    updateComposer();
    if (finalMessage) {
      node.querySelector(".msg-body").innerHTML = assistantBody(finalMessage);
      state.messages.push(finalMessage);
    } else if (run && run.state.aborted) {
      node.querySelector(".msg-body").innerHTML = assistantBody({ content: run.state.text, meta: { stopped: true } });
    }
    await loadConversations();
    scrollToBottom();
  }
}

function onEvent(name, data, userEl, done) {
  const run = state.running;
  if (!run) return;
  switch (name) {
    case "run":
      if (!state.currentId || data.conversation_created) {
        state.currentId = data.conversation_id;
        $("chat-title").textContent = (data.user_message.content || "New chat").split("\n")[0].slice(0, 60);
      }
      state.messages.push(data.user_message);
      userEl.replaceWith(userNode(data.user_message));
      if (data.conversation_created) loadConversations();
      break;
    case "routing":
      run.state.model = data.model;
      run.state.routing = data.routing;
      run.state.status = "Waiting for model…";
      break;
    case "model":
      run.state.model = data.model;
      if (data.fallback) toast(`Fallback to ${data.model}`);
      break;
    case "status":
      run.state.status = data.state === "running" ? "Agent working…" : "Generating…";
      break;
    case "phase":
      run.state.status = `${data.phase}: ${data.message}`.slice(0, 120);
      break;
    case "token":
      run.state.text += data.delta;
      run.state.status = "Generating…";
      break;
    case "done":
    case "stopped":
      done(data.message);
      return;
    case "error":
      done({ role: "assistant", content: run.state.text, meta: { error: { message: data.message, detail: data.detail } } });
      if (data.code === "no_model") refreshStatus(true);
      return;
    default:
      return;
  }
  renderLive();
}

async function stop() {
  const run = state.running;
  if (!run) return;
  run.state.status = "Stopping…";
  run.state.aborted = true;
  renderLive();
  try {
    await api.stop(run.runId);
  } catch {
    run.controller.abort();
  }
  // Fallback: hört der Stream nicht binnen 3 s auf, Verbindung trennen
  setTimeout(() => state.running === run && run.controller.abort(), 3000);
}

// ----------------------------------------------------------------- Anhänge

function renderAttachments() {
  const box = $("attachments");
  box.innerHTML = "";
  box.hidden = !state.attachments.length;
  state.attachments.forEach((a, i) => {
    const chip = el("span", "chip");
    chip.innerHTML = `<span class="chip-name"></span><span class="chip-size">${formatBytes(a.size)}</span><button type="button" aria-label="Remove attachment"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>`;
    chip.querySelector(".chip-name").textContent = a.name;
    chip.querySelector("button").addEventListener("click", () => {
      state.attachments.splice(i, 1);
      renderAttachments();
      updateComposer();
    });
    box.append(chip);
  });
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function addFiles(files) {
  for (const file of files) {
    if (file.size > MAX_FILE) {
      toast(`${file.name}: larger than 10 MB`, "err");
      continue;
    }
    if (state.attachments.length >= 10) {
      toast("At most 10 attachments per message", "err");
      break;
    }
    try {
      const data = await readFile(file);
      state.attachments.push({ name: file.name, mime: file.type || "", size: file.size, data_b64: data, kind: file.type.startsWith("image/") ? "image" : "text" });
    } catch {
      toast(`${file.name}: could not be read`, "err");
    }
  }
  renderAttachments();
  updateComposer();
}

// ----------------------------------------------------------------- Status-Panel

function kv(rows) {
  return `<dl class="kv">${rows
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`)
    .join("")}</dl>`;
}

async function renderStatusPanel(refresh = false) {
  const body = $("status-body");
  body.innerHTML = '<div class="muted">Loading…</div>';
  try {
    const [system, modelStatus, models, router] = await Promise.all([api.systemStatus(), api.modelsStatus(refresh), api.models(), api.routerStatus()]);
    state.system = system;
    state.modelStatus = modelStatus;
    renderStatusIndicator();
    const res = system.resources || {};
    const gb = (v) => (typeof v === "number" ? `${v.toFixed(1)} GB` : "unknown");
    let html = `<div class="panel"><h3>System</h3>${kv([
      ["Version", system.version],
      ["Mode", system.dev_mode ? "Development" : "Production"],
      ["Uptime", formatDuration(system.uptime_s)],
      ["Hardware", system.hardware_summary || "unknown"],
      ["Free VRAM", gb(res.vram_free_gb)],
      ["Free RAM", gb(res.ram_free_gb)],
      ["Config", system.config_path || "none"],
      ["Config problem", system.config_error],
      ["Data", system.data_dir],
      ["Active runs", system.active_runs.length],
    ])}</div>`;
    const byName = Object.fromEntries(models.models.map((m) => [m.name, m]));
    html += `<div class="panel"><h3>Models</h3>`;
    if (!modelStatus.models.length) html += `<div class="muted">No local model available. No models configured.</div>`;
    for (const m of modelStatus.models) {
      const info = byName[m.name] || {};
      const measured = info.data_status === "MEASURED" || info.data_status === "PARTIAL";
      const caps = info.name ? `ctx ${info.context_length} · coding ${info.coding} · reasoning ${info.reasoning} · vision ${info.vision}${info.tool_calling ? " · tools" : ""}` : "";
      html += `<div class="model-row"><span class="status-dot" data-state="${m.available ? "ok" : "err"}"></span><div><div class="model-name">${escapeHtml(m.name)}<span class="tag ${measured ? "measured" : "unmeasured"}">${escapeHtml(info.data_status || "UNMEASURED")}</span></div><div class="model-sub">${escapeHtml(m.provider)} · ${escapeHtml(m.reason)}</div>${caps ? `<div class="model-sub">${escapeHtml(caps)}</div>` : ""}</div></div>`;
    }
    html += `</div>`;
    if (modelStatus.providers.length) {
      html += `<div class="panel"><h3>Runtimes</h3>${modelStatus.providers
        .map((p) => `<div class="model-row"><span class="status-dot" data-state="${p.ready ? "ok" : p.reachable ? "warn" : "err"}"></span><div><div class="model-name">${escapeHtml(p.name)}</div><div class="model-sub">${escapeHtml(p.detail || (p.ready ? "ready" : "not ready"))}</div></div></div>`)
        .join("")}</div>`;
    }
    html += `<div class="panel"><h3>Router</h3>${kv([
      ["Router", router.router],
      ["Classifier", router.classifier],
      ["Learned ranker", router.learned_ranker ? (router.learned_ranker.loaded ? "loaded" : "not loaded – rule ranking") : null],
      ["Routing log", router.log_path],
    ])}`;
    if (router.recent_decisions.length) {
      html += router.recent_decisions
        .slice(0, 8)
        .map((d) => `<div class="decision"><div class="decision-head"><span>${escapeHtml(categoryLabel(d.category) || "")} · ${escapeHtml(String(d.complexity || "").toLowerCase())}</span><span>${escapeHtml(d.selected_model || "")}</span></div><div class="decision-reason">${escapeHtml(d.reason || "")}</div></div>`)
        .join("");
    } else {
      html += `<div class="muted spaced">No routing decisions yet.</div>`;
    }
    html += `</div>`;
    body.innerHTML = html;
  } catch (err) {
    body.innerHTML = `<div class="error-card"><div class="err-title">${escapeHtml(err.message)}</div></div>`;
  }
}

function openStatus() {
  $("status-drawer").hidden = false;
  renderStatusPanel(true);
}

// ----------------------------------------------------------------- Einstellungen

function openSettings() {
  const s = state.settings;
  if (!s) return;
  const form = $("settings-form");
  const select = $("set-model");
  select.innerHTML = '<option value="auto">Auto (router decides)</option>';
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    select.append(opt);
  }
  for (const [key, value] of Object.entries(s)) {
    const field = form.elements.namedItem(key);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = Boolean(value);
    else field.value = value;
  }
  $("settings-error").hidden = true;
  $("settings-modal").hidden = false;
}

async function saveSettings(event) {
  event.preventDefault();
  const form = $("settings-form");
  const data = {
    model: form.model.value,
    temperature: Number(form.temperature.value),
    max_tokens: Number(form.max_tokens.value),
    history_messages: Number(form.history_messages.value),
    request_timeout_s: Number(form.request_timeout_s.value),
    system_prompt: form.system_prompt.value,
    agent_workspace: form.agent_workspace.value.trim(),
    show_routing: form.show_routing.checked,
  };
  try {
    state.settings = await api.saveSettings(data);
    $("settings-modal").hidden = true;
    renderModelChip();
    updateComposer();
    toast("Settings saved");
  } catch (err) {
    const box = $("settings-error");
    box.textContent = err.message;
    box.hidden = false;
  }
}

// ----------------------------------------------------------------- Layout

function openSidebar() {
  $("app").classList.add("sidebar-open");
}
function closeSidebar() {
  $("app").classList.remove("sidebar-open");
}

function autosize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
}

function bind() {
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!state.running) send();
    }
  });
  $("input").addEventListener("input", () => {
    autosize();
    updateComposer();
  });
  $("new-chat").addEventListener("click", newChat);
  $("search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderConversations();
  });
  $("conversations").addEventListener("click", async (e) => {
    const row = e.target.closest(".conv");
    if (!row) return;
    const act = e.target.closest("button")?.dataset.act;
    const id = row.dataset.id;
    if (act === "delete") {
      if (!confirm("Delete this chat?")) return;
      try {
        await api.remove(id);
        if (state.currentId === id) newChat();
        await loadConversations();
      } catch (err) {
        handleError(err);
      }
    } else if (act === "rename") {
      const current = state.conversations.find((c) => c.id === id);
      const title = prompt("Rename chat", current ? current.title : "");
      if (!title || !title.trim()) return;
      try {
        await api.rename(id, title.trim());
        if (state.currentId === id) $("chat-title").textContent = title.trim();
        await loadConversations();
      } catch (err) {
        handleError(err);
      }
    } else {
      openConversation(id);
    }
  });
  $("conversations").addEventListener("keydown", (e) => {
    const row = e.target.closest(".conv");
    if (row && e.key === "Enter") openConversation(row.dataset.id);
  });
  $("messages").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const code = btn.closest(".code-block").querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      const area = document.createElement("textarea");
      area.value = code;
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    btn.classList.add("copied");
    btn.querySelector("span").textContent = "Copied";
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.querySelector("span").textContent = "Copy";
    }, 1600);
  });
  $("attach").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    addFiles([...e.target.files]);
    e.target.value = "";
  });
  let dragDepth = 0;
  const main = $("main");
  main.addEventListener("dragenter", (e) => {
    if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
    dragDepth += 1;
    $("dropzone").hidden = false;
  });
  main.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) $("dropzone").hidden = true;
  });
  main.addEventListener("dragover", (e) => e.preventDefault());
  main.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    $("dropzone").hidden = true;
    if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]);
  });
  $("input").addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) {
      e.preventDefault();
      addFiles(files);
    }
  });
  $("open-settings").addEventListener("click", openSettings);
  $("model-chip").addEventListener("click", openSettings);
  $("close-settings").addEventListener("click", () => ($("settings-modal").hidden = true));
  $("cancel-settings").addEventListener("click", () => ($("settings-modal").hidden = true));
  $("settings-form").addEventListener("submit", saveSettings);
  $("open-status").addEventListener("click", openStatus);
  $("no-model-status").addEventListener("click", openStatus);
  $("close-status").addEventListener("click", () => ($("status-drawer").hidden = true));
  $("refresh-status").addEventListener("click", () => renderStatusPanel(true));
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("scrim").addEventListener("click", closeSidebar);
  $("token-form").addEventListener("submit", (e) => {
    e.preventDefault();
    setToken(e.target.token.value);
    $("token-modal").hidden = true;
    init();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("settings-modal").hidden = true;
      $("status-drawer").hidden = true;
      closeSidebar();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      newChat();
    }
  });
}

async function init() {
  try {
    const [settings, models] = await Promise.all([api.settings(), api.models()]);
    state.settings = settings;
    state.models = models.models;
  } catch (err) {
    handleError(err);
  }
  renderModelChip();
  await Promise.all([refreshStatus(), loadConversations()]);
  updateComposer();
}

bind();
init();
setInterval(() => !state.running && refreshStatus(), 30000);

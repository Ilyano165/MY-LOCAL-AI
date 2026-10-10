// Client für die NOVA API. Die UI spricht ausschließlich mit diesem Server.

const TOKEN_KEY = "nova.apiToken";

export class ApiError extends Error {
  constructor(status, code, message, detail) {
    super(message);
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

function token() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(value) {
  try {
    localStorage.setItem(TOKEN_KEY, value);
  } catch {
    /* Speicher nicht verfügbar – Token gilt nur für diese Sitzung */
  }
  sessionToken = value;
}

let sessionToken = null;

function headers(extra = {}) {
  const h = { "X-NOVA-Client": "ui", ...extra };
  const t = sessionToken || token();
  if (t) h["X-NOVA-Token"] = t;
  return h;
}

async function errorFrom(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* kein JSON */
  }
  const err = body && body.error;
  if (err) return new ApiError(response.status, err.code, err.message, err.detail);
  if (body && body.detail) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : String(body.detail);
    return new ApiError(response.status, "invalid_request", detail);
  }
  return new ApiError(response.status, "http_error", `HTTP ${response.status}`);
}

export async function request(method, path, body) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: headers(body !== undefined ? { "Content-Type": "application/json" } : {}),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(0, "offline", "NOVA API not reachable. Is the server running?");
  }
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}

// Inkrementeller Parser für Server-Sent Events (auch über Chunk-Grenzen hinweg).
export function createSSEParser(onEvent) {
  let buffer = "";
  return function feed(text) {
    buffer += text.replace(/\r\n?/g, "\n");
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      let name = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      let payload;
      try {
        payload = JSON.parse(data.join("\n"));
      } catch {
        payload = { raw: data.join("\n") };
      }
      onEvent(name, payload);
    }
  };
}

// POST mit SSE-Antwort; abbrechbar über AbortSignal.
export async function stream(path, body, onEvent, signal) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e && e.name === "AbortError") throw e;
    throw new ApiError(0, "offline", "NOVA API not reachable. Is the server running?");
  }
  if (!response.ok) throw await errorFrom(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const feed = createSSEParser(onEvent);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    feed(decoder.decode(value, { stream: true }));
  }
  feed(decoder.decode() + "\n\n");
}

export const api = {
  systemStatus: () => request("GET", "/system/status"),
  models: () => request("GET", "/models"),
  modelsStatus: (refresh = false) => request("GET", `/models/status${refresh ? "?refresh=true" : ""}`),
  routerStatus: () => request("GET", "/router/status"),
  settings: () => request("GET", "/settings"),
  saveSettings: (changes) => request("PUT", "/settings", changes),
  conversations: () => request("GET", "/conversations"),
  conversation: (id) => request("GET", `/conversations/${encodeURIComponent(id)}`),
  rename: (id, title) => request("PATCH", `/conversations/${encodeURIComponent(id)}`, { title }),
  remove: (id) => request("DELETE", `/conversations/${encodeURIComponent(id)}`),
  stop: (runId) => request("POST", "/agent/stop", { run_id: runId }),
  modelCatalog: () => request("GET", "/models/catalog"),
  startDownload: (id, acceptLicense) => request("POST", "/models/downloads", { id, accept_license: acceptLicense }),
  researchStatus: () => request("GET", "/research/status"),
  researchConfig: (config) => request("PUT", "/research/config", config),
  researchRuns: () => request("GET", "/research/runs"),
  researchStart: (body) => request("POST", "/research/runs", body),
  researchReport: (id) => request("GET", `/research/runs/${encodeURIComponent(id)}/report`),
  researchCancel: (id) => request("POST", `/research/runs/${encodeURIComponent(id)}/cancel`),
  researchResume: (id) => request("POST", `/research/runs/${encodeURIComponent(id)}/resume`),
  chatStream: (body, onEvent, signal) => stream("/chat/stream", body, onEvent, signal),
  agentStream: (body, onEvent, signal) => stream("/agent/run", body, onEvent, signal),
};

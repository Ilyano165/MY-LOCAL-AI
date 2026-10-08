// Formatierung von Messwerten. Grundsatz: nur anzeigen, was tatsächlich vorliegt.

export function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatDuration(seconds) {
  if (!isNum(seconds)) return null;
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

export function formatBytes(bytes) {
  if (!isNum(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const CATEGORY = {
  fast: "Fast",
  general: "General",
  coding: "Coding",
  reasoning: "Reasoning",
  vision: "Vision",
  long_context: "Long context",
};

export function categoryLabel(value) {
  return value ? CATEGORY[value] ?? value : null;
}

// Kennzahlen nach einer Antwort. Fehlende Werte werden weggelassen – nie geschätzt.
export function responseStats(meta) {
  if (!meta || typeof meta !== "object") return [];
  const stats = [];
  if (meta.model) stats.push({ label: "Model used", value: meta.model });
  const time = formatDuration(meta.generation_time_s);
  if (time) stats.push({ label: "Generation time", value: time });
  if (isNum(meta.completion_tokens)) {
    const prompt = isNum(meta.prompt_tokens) ? ` · ${meta.prompt_tokens} in` : "";
    stats.push({ label: "Tokens", value: `${meta.completion_tokens} out${prompt}` });
  }
  if (isNum(meta.tokens_per_second)) {
    stats.push({ label: "Tokens/sec", value: meta.tokens_per_second.toFixed(1) });
  } else if (isNum(meta.runtime_tokens_per_second)) {
    stats.push({ label: "Tokens/sec", value: `${meta.runtime_tokens_per_second.toFixed(1)} (runtime)` });
  }
  const routing = meta.routing;
  if (routing && routing.mode === "auto" && routing.category) {
    const complexity = routing.complexity ? ` · ${String(routing.complexity).toLowerCase()}` : "";
    stats.push({ label: "Routing", value: `${categoryLabel(routing.category)}${complexity}` });
  } else if (routing && routing.mode === "manual") {
    stats.push({ label: "Routing", value: "manual" });
  }
  if (meta.verification && meta.verification.status) {
    const v = meta.verification;
    const tone = v.verified ? "ok" : v.status === "failed" ? "err" : "warn";
    stats.push({ label: "Verification", value: v.status, tone });
  }
  if (meta.streamed === false) stats.push({ label: "Streaming", value: "not supported by runtime" });
  return stats;
}

// Live-Zeile während der Generierung.
export function liveLine(state) {
  const parts = [];
  if (state.model) parts.push({ label: "Model", value: state.model });
  if (state.routing && state.routing.mode === "auto" && state.routing.category) {
    const c = state.routing.complexity ? ` · ${String(state.routing.complexity).toLowerCase()}` : "";
    parts.push({ label: "Routing", value: `${categoryLabel(state.routing.category)}${c}` });
  } else if (state.routing && state.routing.mode === "manual") {
    parts.push({ label: "Routing", value: "manual selection" });
  }
  parts.push({ label: "Status", value: state.status || "Generating…" });
  return parts;
}

export function dayGroup(iso, now = new Date()) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Older";
  const start = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((start(now) - start(d)) / 86400000);
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return "Older";
}

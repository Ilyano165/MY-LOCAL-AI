// Quellenblock unter Antworten, die Erkenntnisse aus NOVA-Research nutzen.
import { escapeHtml } from "./escape.js";

const LABEL = {
  supported: "supported",
  contested: "disputed",
  unverified: "unverified",
  hypothesis: "hypothesis",
  opinion: "opinion",
};

function safeUrl(url) {
  return /^https?:\/\//i.test(String(url || "")) ? String(url) : null;
}

function refItem(r) {
  const sources = (r.sources || [])
    .map((s) => {
      const url = safeUrl(s.url);
      const title = escapeHtml(s.title || s.url || "source");
      return url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title;
    })
    .join(", ");
  const label = LABEL[r.classification] || r.classification;
  return `<li><b>[${escapeHtml(r.ref)}]</b> ${escapeHtml(r.statement)} <span class="tag ${r.classification === "supported" ? "measured" : "unmeasured"}">${escapeHtml(label)}</span>${sources ? `<div class="model-sub">${sources}</div>` : ""}</li>`;
}

export function renderKnowledge(refs) {
  if (!Array.isArray(refs) || !refs.length) return "";
  const cited = refs.filter((r) => r.cited);
  const other = refs.filter((r) => !r.cited);
  let html = `<div class="knowledge">`;
  if (cited.length) html += `<div class="reason-label">Sources (NOVA research)</div><ul>${cited.map(refItem).join("")}</ul>`;
  if (other.length) {
    html += `<details><summary>${other.length} research finding${other.length > 1 ? "s" : ""} provided but not cited</summary><ul>${other.map(refItem).join("")}</ul></details>`;
  }
  return `${html}</div>`;
}

// HTML-Escaping (gemeinsam für Markdown und Syntaxhervorhebung).
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ESC[c]);
}

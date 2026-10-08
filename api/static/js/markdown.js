// Kleiner, sicherer Markdown-Renderer (kein HTML aus dem Modell wird ausgeführt).
// Unterstützt: Absätze, Überschriften, Listen (verschachtelt), Zitate, Tabellen, Trennlinien,
// Code-Blöcke (auch unvollständige während des Streamings), Inline-Code, fett/kursiv,
// durchgestrichen, Links (nur http/https/mailto) und Autolinks.

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ESC[c]);
}

const COPY_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';

export function safeUrl(url) {
  const trimmed = url.trim();
  return /^(https?:\/\/|mailto:)/i.test(trimmed) ? trimmed : null;
}

function inline(text) {
  // text ist bereits HTML-escaped. Code-Spans zuerst schützen.
  const slots = [];
  const keep = (html) => `\u0000${slots.push(html) - 1}\u0000`;
  let out = text.replace(/`([^`\n]+)`/g, (_, code) => keep(`<code>${code}</code>`));
  out = out.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
    const url = safeUrl(href.replace(/&amp;/g, "&"));
    if (!url) return match;
    return keep(`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  });
  out = out.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, (_, pre, href) => {
    const url = href.replace(/&amp;/g, "&");
    return pre + keep(`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${href}</a>`);
  });
  out = out
    .replace(/\*\*(?=\S)([^*]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/__(?=\S)([^_]*?\S)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_(?=\S)([^_\n]*?\S)_(?![_\w])/g, "$1<em>$2</em>")
    .replace(/~~(?=\S)([^~]*?\S)~~/g, "<del>$1</del>");
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => slots[Number(i)]);
}

function codeBlock(lang, lines, open) {
  const language = lang ? escapeHtml(lang.toLowerCase().slice(0, 30)) : "";
  const label = language || "code";
  const cls = language ? ` class="language-${language}"` : "";
  return (
    `<div class="code-block${open ? " streaming" : ""}"><div class="code-head">` +
    `<span class="code-lang">${label}</span>` +
    `<button type="button" class="copy-btn" data-copy aria-label="Copy code">${COPY_ICON}<span>Copy</span></button>` +
    `</div><pre><code${cls}>${escapeHtml(lines.join("\n"))}</code></pre></div>`
  );
}

const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const HR_RE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;

function splitRow(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|")) row = row.slice(0, -1);
  return row.split("|").map((c) => c.trim());
}

function renderList(items) {
  // items: [{indent, ordered, text}] – verschachtelt nach Einrückung
  let html = "";
  let i = 0;
  const base = items[0].indent;
  const ordered = items[0].ordered;
  html += ordered ? "<ol>" : "<ul>";
  while (i < items.length) {
    const item = items[i];
    if (item.indent < base) break;
    let j = i + 1;
    const children = [];
    while (j < items.length && items[j].indent > base) {
      children.push(items[j]);
      j += 1;
    }
    html += `<li>${inline(escapeHtml(item.text))}${children.length ? renderList(children) : ""}</li>`;
    i = j;
  }
  return html + (ordered ? "</ol>" : "</ul>");
}

export function renderMarkdown(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)\s*$/);
    if (fence) {
      const marker = fence[1][0];
      const body = [];
      i += 1;
      let closed = false;
      while (i < lines.length) {
        if (new RegExp(`^\\s*${marker === "`" ? "`" : "~"}{3,}\\s*$`).test(lines[i])) {
          closed = true;
          i += 1;
          break;
        }
        body.push(lines[i]);
        i += 1;
      }
      out.push(codeBlock(fence[2], body, !closed));
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${inline(escapeHtml(heading[2]))}</h${level}>`);
      i += 1;
      continue;
    }
    if (HR_RE.test(line)) {
      out.push("<hr>");
      i += 1;
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        quote.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      out.push(`<blockquote>${renderMarkdown(quote.join("\n"))}</blockquote>`);
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      const th = head.map((c) => `<th>${inline(escapeHtml(c))}</th>`).join("");
      const tb = rows
        .map((r) => `<tr>${head.map((_, k) => `<td>${inline(escapeHtml(r[k] ?? ""))}</td>`).join("")}</tr>`)
        .join("");
      out.push(`<table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`);
      continue;
    }
    if (LIST_RE.test(line)) {
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(LIST_RE);
        if (m) {
          items.push({ indent: m[1].replace(/\t/g, "    ").length, ordered: /\d/.test(m[2]), text: m[3] });
          i += 1;
        } else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1].text += " " + lines[i].trim();
          i += 1;
        } else {
          break;
        }
      }
      out.push(renderList(items));
      continue;
    }
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\s*(```|~~~)/.test(lines[i]) &&
      !/^\s{0,3}#{1,6}\s/.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !LIST_RE.test(lines[i]) &&
      !HR_RE.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(`<p>${para.map((p) => inline(escapeHtml(p))).join("<br>")}</p>`);
  }
  return out.join("");
}

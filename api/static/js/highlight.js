// Leichte, sichere Syntaxhervorhebung für Code-Blöcke (ohne externe Bibliothek).
// Arbeitet auf dem Rohtext: jedes Token wird einzeln escaped – Modellausgabe kann kein HTML
// einschleusen. Der Textinhalt bleibt exakt erhalten (Copy kopiert den Originalcode).

import { escapeHtml } from "./escape.js";

const C_LIKE_COMMENTS = { line: "//", block: ["/*", "*/"] };

const LANGS = {
  python: {
    aliases: ["py", "python3"],
    line: "#",
    strings: ['"""', "'''", '"', "'"],
    kw: "and as assert async await break class continue def del elif else except finally for from global if import in is lambda match case nonlocal not or pass raise return try while with yield",
    lit: "True False None self cls",
  },
  javascript: {
    aliases: ["js", "jsx", "mjs", "cjs", "ts", "tsx", "typescript", "json", "jsonc"],
    ...C_LIKE_COMMENTS,
    strings: ["`", '"', "'"],
    kw: "async await break case catch class const continue debugger default delete do else export extends finally for from function if implements import in instanceof interface let new of return static switch throw try type typeof var void while with yield enum readonly private public protected as",
    lit: "true false null undefined this NaN Infinity",
  },
  bash: {
    aliases: ["sh", "shell", "zsh", "console"],
    line: "#",
    strings: ['"', "'"],
    kw: "if then else elif fi for while until do done case esac function in select return export local readonly unset",
    lit: "true false",
  },
  rust: {
    aliases: ["rs"],
    ...C_LIKE_COMMENTS,
    strings: ['"'],
    kw: "as async await break const continue crate dyn else enum extern fn for if impl in let loop match mod move mut pub ref return static struct super trait type unsafe use where while",
    lit: "true false self Self None Some Ok Err",
  },
  go: {
    aliases: ["golang"],
    ...C_LIKE_COMMENTS,
    strings: ["`", '"', "'"],
    kw: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
    lit: "true false nil iota",
  },
  c: {
    aliases: ["cpp", "c++", "h", "hpp", "cc", "java", "kotlin", "kt", "cs", "csharp", "swift", "php", "scala", "dart"],
    ...C_LIKE_COMMENTS,
    strings: ['"', "'"],
    kw: "auto break case catch char class const continue default delete do double else enum extends final finally float for fun func goto if implements import include int interface long namespace new override package private protected public return short signed sizeof static struct switch template this throw throws try typedef typename union unsigned using val var virtual void volatile while",
    lit: "true false null nullptr NULL nil",
  },
  sql: {
    aliases: ["postgres", "postgresql", "mysql", "sqlite"],
    line: "--",
    block: ["/*", "*/"],
    strings: ["'", '"'],
    kw: "select from where and or not insert into values update set delete create table index view drop alter add join left right inner outer full on group by order having limit offset as distinct union all case when then else end primary key foreign references unique default exists in between like is with returning",
    lit: "null true false",
    caseInsensitive: true,
  },
  css: {
    aliases: ["scss", "less"],
    block: ["/*", "*/"],
    strings: ['"', "'"],
    kw: "important media import keyframes from to",
    lit: "",
  },
  toml: {
    aliases: ["yaml", "yml", "ini", "cfg", "conf", "dockerfile", "make", "makefile"],
    line: "#",
    strings: ['"""', '"', "'"],
    kw: "FROM RUN CMD COPY ADD ENV WORKDIR EXPOSE ENTRYPOINT ARG USER VOLUME LABEL",
    lit: "true false null yes no on off",
  },
};

const BY_NAME = new Map();
for (const [name, spec] of Object.entries(LANGS)) {
  BY_NAME.set(name, spec);
  for (const alias of spec.aliases) BY_NAME.set(alias, spec);
}

const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const CACHE = new Map();

function compile(spec) {
  if (CACHE.has(spec)) return CACHE.get(spec);
  const parts = [];
  if (spec.block) parts.push(`(?<com>${reEsc(spec.block[0])}[\\s\\S]*?(?:${reEsc(spec.block[1])}|$))`);
  if (spec.line) parts.push(`(?<line>${reEsc(spec.line)}[^\\n]*)`);
  const strings = spec.strings.map((d) => {
    const q = reEsc(d);
    if (d.length === 3) return `${q}[\\s\\S]*?(?:${q}|$)`;
    if (d === "`") return "`(?:[^`\\\\]|\\\\[\\s\\S])*(?:`|$)";
    return `${q}(?:[^${q}\\\\\\n]|\\\\.)*(?:${q}|$)`;
  });
  parts.push(`(?<str>${strings.join("|")})`);
  parts.push("(?<num>\\b(?:0[xob][\\da-fA-F_]+|\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b)");
  parts.push("(?<word>[A-Za-z_$][\\w$]*)");
  const flags = "g";
  const fold = (s) => (spec.caseInsensitive ? s.toLowerCase() : s);
  const compiled = {
    re: new RegExp(parts.join("|"), flags),
    kw: new Set(spec.kw.split(/\s+/).filter(Boolean).map(fold)),
    lit: new Set(spec.lit.split(/\s+/).filter(Boolean).map(fold)),
    fold,
  };
  CACHE.set(spec, compiled);
  return compiled;
}

export function languageSupported(lang) {
  return BY_NAME.has(String(lang || "").toLowerCase());
}

export function highlight(code, lang) {
  const spec = BY_NAME.get(String(lang || "").toLowerCase());
  if (!spec || code.length > 100_000) return escapeHtml(code);
  const { re, kw, lit, fold } = compile(spec);
  re.lastIndex = 0;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(code)) !== null) {
    if (m[0] === "") {
      re.lastIndex += 1;
      continue;
    }
    out += escapeHtml(code.slice(last, m.index));
    const text = escapeHtml(m[0]);
    const g = m.groups;
    let cls = null;
    if (g.com !== undefined || g.line !== undefined) cls = "com";
    else if (g.str !== undefined) cls = "str";
    else if (g.num !== undefined) cls = "num";
    else if (g.word !== undefined) {
      const w = fold(m[0]);
      if (kw.has(w)) cls = "kw";
      else if (lit.has(w)) cls = "lit";
      else if (/^\s*\(/.test(code.slice(re.lastIndex, re.lastIndex + 40))) cls = "fn";
    }
    out += cls ? `<span class="tok-${cls}">${text}</span>` : text;
    last = re.lastIndex;
  }
  return out + escapeHtml(code.slice(last));
}

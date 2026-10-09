import test from "node:test";
import assert from "node:assert/strict";
import { highlight, languageSupported } from "../../api/static/js/highlight.js";
import { renderMarkdown } from "../../api/static/js/markdown.js";

const strip = (html) =>
  html
    .replace(/<[^>]+>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");

test("python keywords, strings, comments, numbers, functions", () => {
  const html = highlight('def add(a, b):\n    # sum\n    return a + b + 42  # "x"\nprint("hi")', "python");
  assert.match(html, /<span class="tok-kw">def<\/span> <span class="tok-fn">add<\/span>/);
  assert.match(html, /<span class="tok-com"># sum<\/span>/);
  assert.match(html, /<span class="tok-num">42<\/span>/);
  assert.match(html, /<span class="tok-str">&quot;hi&quot;<\/span>/);
  assert.match(html, /<span class="tok-kw">return<\/span>/);
});

test("text content is preserved exactly (copy stays correct)", () => {
  const samples = [
    ["python", 's = """multi\nline <b>"""\nx = \'it\\\'s\''],
    ["javascript", "const t = `a ${b}`; // <script>\n/* block */ if (x < 2) {}"],
    ["sql", "SELECT * FROM users WHERE name = 'O''Brien' -- note"],
    ["bash", 'echo "$HOME" # comment & more'],
  ];
  for (const [lang, code] of samples) assert.equal(strip(highlight(code, lang)), code, lang);
});

test("no HTML injection through code", () => {
  const html = highlight('x = "<img src=x onerror=alert(1)>" # <script>', "python");
  assert.ok(!html.includes("<img") && !html.includes("<script"));
});

test("unknown languages are only escaped", () => {
  assert.equal(highlight("a < b", "brainfuck"), "a &lt; b");
  assert.equal(highlight("a < b", ""), "a &lt; b");
  assert.ok(languageSupported("TS") && !languageSupported("cobol"));
});

test("SQL keywords are case-insensitive", () => {
  assert.match(highlight("select 1 from t", "sql"), /<span class="tok-kw">select<\/span>/);
});

test("markdown code blocks use highlighting", () => {
  const html = renderMarkdown("```js\nconst x = 1;\n```");
  assert.match(html, /<span class="tok-kw">const<\/span>/);
});

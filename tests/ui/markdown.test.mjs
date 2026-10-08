import test from "node:test";
import assert from "node:assert/strict";
import { renderMarkdown, escapeHtml, safeUrl } from "../../api/static/js/markdown.js";

test("escapes raw HTML (no XSS)", () => {
  const html = renderMarkdown('<img src=x onerror="alert(1)"> <script>alert(1)</script>');
  assert.ok(!html.includes("<img"));
  assert.ok(!html.includes("<script"));
  assert.ok(html.includes("&lt;script&gt;"));
});

test("rejects javascript: links but renders http links safely", () => {
  const bad = renderMarkdown("[click](javascript:alert(1))");
  assert.ok(!bad.includes("href="));
  const good = renderMarkdown("[NOVA](https://example.org/a?b=1&c=2)");
  assert.match(good, /<a href="https:\/\/example.org\/a\?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">NOVA<\/a>/);
  assert.equal(safeUrl(" data:text/html,x"), null);
});

test("code blocks with language, escaping and copy button", () => {
  const html = renderMarkdown("```python\nif a < b:\n    print('<tag>')\n```");
  assert.match(html, /class="code-block"/);
  assert.match(html, /<span class="code-lang">python<\/span>/);
  assert.match(html, /data-copy/);
  assert.ok(html.includes("if a &lt; b:"));
  assert.ok(html.includes("print(&#39;&lt;tag&gt;&#39;)"));
});

test("unclosed fence while streaming still renders as code", () => {
  const html = renderMarkdown("Here:\n```js\nconst x = 1;");
  assert.match(html, /code-block streaming/);
  assert.ok(html.includes("const x = 1;"));
});

test("markdown inside code is not formatted", () => {
  const html = renderMarkdown("```\n**not bold** and `tick`\n```");
  assert.ok(!html.includes("<strong>"));
  const inline = renderMarkdown("Use `**raw**` here");
  assert.ok(inline.includes("<code>**raw**</code>"));
});

test("headings, emphasis, strike, lists, nesting", () => {
  const html = renderMarkdown("## Title\n\n**bold** and *it* and ~~old~~\n\n- a\n  - b\n- c\n\n1. one\n2. two");
  assert.ok(html.includes("<h2>Title</h2>"));
  assert.ok(html.includes("<strong>bold</strong>"));
  assert.ok(html.includes("<em>it</em>"));
  assert.ok(html.includes("<del>old</del>"));
  assert.match(html, /<ul><li>a<ul><li>b<\/li><\/ul><\/li><li>c<\/li><\/ul>/);
  assert.match(html, /<ol><li>one<\/li><li>two<\/li><\/ol>/);
});

test("tables and blockquotes", () => {
  const html = renderMarkdown("| a | b |\n|---|:-:|\n| 1 | <x> |\n\n> quoted **text**");
  assert.match(html, /<table><thead><tr><th>a<\/th><th>b<\/th><\/tr><\/thead><tbody><tr><td>1<\/td><td>&lt;x&gt;<\/td><\/tr><\/tbody><\/table>/);
  assert.match(html, /<blockquote><p>quoted <strong>text<\/strong><\/p><\/blockquote>/);
});

test("snake_case words are not italicised", () => {
  assert.ok(!renderMarkdown("call my_var_name now").includes("<em>"));
});

test("escapeHtml", () => {
  assert.equal(escapeHtml(`<a href="x">'&'</a>`), "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
});

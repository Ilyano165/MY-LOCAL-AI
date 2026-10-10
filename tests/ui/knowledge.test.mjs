import test from "node:test";
import assert from "node:assert/strict";
import { renderKnowledge } from "../../api/static/js/knowledge.js";

const refs = [
  { ref: "K1", statement: "A <b>", classification: "supported", cited: true, sources: [{ url: "https://a.org/x", title: "A" }] },
  { ref: "K2", statement: "B", classification: "contested", cited: false, sources: [{ url: "javascript:alert(1)", title: "evil" }] },
];

test("cited and uncited findings are shown separately, escaped, safe links only", () => {
  const html = renderKnowledge(refs);
  assert.match(html, /Sources \(NOVA research\)/);
  assert.match(html, /A &lt;b&gt;/);
  assert.match(html, /href="https:\/\/a.org\/x"/);
  assert.doesNotMatch(html, /href="javascript/);
  assert.match(html, /1 research finding provided but not cited/);
  assert.match(html, /disputed/);
});

test("nothing rendered without knowledge", () => {
  assert.equal(renderKnowledge(undefined), "");
  assert.equal(renderKnowledge([]), "");
});

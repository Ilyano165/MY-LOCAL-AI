import test from "node:test";
import assert from "node:assert/strict";
import { formatRun, hasActiveRun, parseSeedUrls, renderResearchForm, renderRuns } from "../../api/static/js/research.js";

test("seed URLs: only http(s), max 50", () => {
  assert.deepEqual(parseSeedUrls("https://a.org\n javascript:alert(1)  http://b.org/x ftp://c"), ["https://a.org", "http://b.org/x"]);
  assert.equal(parseSeedUrls(Array.from({ length: 60 }, (_, i) => `https://x.org/${i}`).join(" ")).length, 50);
  assert.deepEqual(parseSeedUrls(null), []);
});

test("run list shows status-dependent actions and escapes text", () => {
  const html = renderRuns([
    { id: "r1", objective: "<b>x</b>", status: "running", counts: { sources: 2 }, active_seconds: 90, budget: { max_duration_s: 600 } },
    { id: "r2", objective: "y", status: "paused", stop_reason: "paused", counts: {} },
    { id: "r3", objective: "z", status: "completed", counts: { findings: 4 }, model: "m" },
  ]);
  assert.match(html, /&lt;b&gt;x&lt;\/b&gt;/);
  assert.match(html, /data-research-cancel="r1"/);
  assert.match(html, /data-research-resume="r2"/);
  assert.doesNotMatch(html, /data-research-cancel="r3"|data-research-resume="r3"/);
  assert.match(html, /4 findings/);
  assert.equal(renderRuns([]).includes("No research runs yet"), true);
});

test("active detection and run summary", () => {
  assert.equal(hasActiveRun([{ status: "completed" }, { status: "queued" }]), true);
  assert.equal(hasActiveRun([{ status: "paused" }]), false);
  assert.equal(formatRun({ counts: { sources: 3, claims: 5, findings: 2, errors: 1 }, active_seconds: 120, budget: { max_duration_s: 3600 } }), "3 sources · 5 statements · 2 findings · 1 errors · 2 min of 60 min");
});

test("form explains missing provider and missing model honestly", () => {
  const none = renderResearchForm({ search_provider: null }, false);
  assert.match(none, /No web search provider/);
  assert.match(none, /No local model is available/);
  const ok = renderResearchForm({ search_provider: "searxng" }, true);
  assert.match(ok, /Web search: <b>searxng<\/b>/);
  assert.doesNotMatch(ok, /No local model/);
});

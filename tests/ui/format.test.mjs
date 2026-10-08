import test from "node:test";
import assert from "node:assert/strict";
import { responseStats, liveLine, formatDuration, dayGroup } from "../../api/static/js/format.js";
import { createSSEParser } from "../../api/static/js/api.js";

test("responseStats shows only values that exist", () => {
  assert.deepEqual(responseStats({}), []);
  assert.deepEqual(responseStats({ generation_time_s: 1.5 }), [{ label: "Generation time", value: "1.50 s" }]);
  const full = responseStats({
    model: "m",
    generation_time_s: 0.25,
    completion_tokens: 12,
    prompt_tokens: 30,
    tokens_per_second: 41.234,
    routing: { mode: "auto", category: "coding", complexity: "HIGH" },
    verification: { status: "success", verified: true },
  });
  assert.deepEqual(
    full.map((s) => [s.label, s.value]),
    [
      ["Model used", "m"],
      ["Generation time", "250 ms"],
      ["Tokens", "12 out · 30 in"],
      ["Tokens/sec", "41.2"],
      ["Routing", "Coding · high"],
      ["Verification", "success"],
    ],
  );
  assert.equal(full.at(-1).tone, "ok");
});

test("no tokens/sec without measurement, runtime value labelled", () => {
  assert.ok(!responseStats({ completion_tokens: 3 }).some((s) => s.label === "Tokens/sec"));
  const rt = responseStats({ runtime_tokens_per_second: 20 });
  assert.equal(rt[0].value, "20.0 (runtime)");
});

test("non-streaming runtimes are labelled honestly", () => {
  assert.ok(responseStats({ streamed: false }).some((s) => s.label === "Streaming"));
});

test("liveLine always has status, model/routing only if known", () => {
  assert.deepEqual(liveLine({ status: "Routing…" }), [{ label: "Status", value: "Routing…" }]);
  const parts = liveLine({ model: "m", routing: { mode: "auto", category: "reasoning", complexity: "MEDIUM" } });
  assert.deepEqual(parts.map((p) => p.label), ["Model", "Routing", "Status"]);
  assert.equal(parts[2].value, "Generating…");
});

test("formatDuration and dayGroup", () => {
  assert.equal(formatDuration(undefined), null);
  assert.equal(formatDuration(75), "1 min 15 s");
  const now = new Date("2026-10-08T12:00:00");
  assert.equal(dayGroup("2026-10-08T08:00:00", now), "Today");
  assert.equal(dayGroup("2026-10-07T23:00:00", now), "Yesterday");
  assert.equal(dayGroup("2026-01-01T00:00:00", now), "Older");
  assert.equal(dayGroup("garbage", now), "Older");
});

test("SSE parser handles chunk boundaries and multiple events", () => {
  const events = [];
  const feed = createSSEParser((name, data) => events.push([name, data]));
  feed('event: token\ndata: {"delta":"He');
  feed('llo"}\n\nevent: done\r\ndata: {"ok":true}\r\n\r\n: comment\n\n');
  assert.deepEqual(events, [["token", { delta: "Hello" }], ["done", { ok: true }]]);
});

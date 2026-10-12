import test from "node:test";
import assert from "node:assert/strict";
import { jobActive, renderDatasets, renderEnv, renderJob, renderStart } from "../../api/static/js/training.js";

const probe = { ready: true, gpu: "GTX <1080 Ti>", vram_gb: 11, compute_capability: "6.1", torch: "2.x+cu126", compute_dtype: "float32", notes: ["Pascal: fp32"], problems: [] };

test("env: setup offer, progress, failure, ready probe", () => {
  assert.match(renderEnv({ ready: false, setup: { status: "idle" } }), /id="training-setup"/);
  assert.match(renderEnv({ ready: false, setup: { status: "installing", step: "torch" } }), /<b>torch<\/b>/);
  assert.match(renderEnv({ ready: false, setup: { status: "failed", error: "uv <x>" } }), /uv &lt;x&gt;/);
  const html = renderEnv({ ready: true, setup: { status: "completed" }, probe });
  assert.match(html, /GTX &lt;1080 Ti&gt;/);
  assert.match(html, /Pascal: fp32/);
  assert.match(html, /id="training-probe"/);
  assert.match(renderEnv({ ready: true, setup: {}, probe: { ...probe, ready: false, problems: ["no CUDA"] } }), /no CUDA/);
});

test("datasets and start form", () => {
  assert.match(renderDatasets([]), /No dataset yet/);
  const ds = [{ name: "nova-knowledge", version: "1.0.0", counts: { train: 30, val: 5, test: 12 } }];
  assert.match(renderDatasets(ds), /nova-knowledge@1.0.0/);
  assert.match(renderStart({ env: { ready: false }, datasets: ds }), /Needs a ready/);
  const form = renderStart({ env: { ready: true }, datasets: ds, installed_model: "general-9b", choices: [{ id: "general-4b", label: "4B", repo: "a/b" }, { id: "general-9b", label: "9B", repo: "c/d" }] });
  assert.match(form, /value="general-9b" selected/);
  assert.match(form, /id="training-form"/);
});

test("job: progress, report, gate, approve, active", () => {
  const running = { id: "20261012-120000-abcd", running: true, training: { status: "training", step: 5, max_steps: 10, loss: 1.2, eta_s: 120 } };
  assert.ok(jobActive(running));
  const h = renderJob(running, null);
  assert.match(h, /step 5\/10 \(50 %\)/);
  assert.match(h, /data-training-cancel/);
  const report = { base: { f1: 0.1, cited_rate: 0, regression_passed: 9 }, tuned: { f1: 0.3, cited_rate: 0.5, regression_passed: 9 }, gate: { eligible: true, reasons: ["F1 gain +0.200"] } };
  const done = { id: "20261012-120000-abcd", running: false, training: { status: "completed" }, evaluation: { status: "completed" }, report };
  assert.ok(!jobActive(done));
  assert.match(renderJob(done, null), /data-training-approve/);
  const blocked = { ...done, report: { ...report, gate: { eligible: false, reasons: ["only 3 <test>"] } } };
  assert.doesNotMatch(renderJob(blocked, null), /data-training-approve/);
  assert.match(renderJob(blocked, null), /only 3 &lt;test&gt;/);
  const approved = { ...done, approval: { approved_at: "2026-10-12" } };
  assert.match(renderJob(approved, "nova-general-9b-20261012-120000"), /active NOVA model/);
  assert.match(renderJob(approved, null), /approved 2026-10-12/);
});

import test from "node:test";
import assert from "node:assert/strict";
import { hardwareLine, renderOneClick } from "../../api/static/js/setup.js";

const base = { platform_supported: true, custom_config: false, setup: { status: "idle" }, gpu: { name: "GTX <1080>", vram_gb: 11, compute_capability: "6.1" }, ram_gb: 32, recommended: { id: "general-9b", label: "Qwen 9B", repo: "x/y", base_license: "apache-2.0" } };

test("offer: hardware, recommendation, license acceptance, button", () => {
  const html = renderOneClick(base);
  assert.match(html, /GTX &lt;1080&gt;/);
  assert.match(html, /Recommended: <b>Qwen 9B<\/b>/);
  assert.match(html, /id="runtime-accept"/);
  assert.match(html, /data-choice="general-9b"/);
  assert.match(hardwareLine({ gpu: null, ram_gb: 16 }), /CPU/);
});

test("progress, installed, failed, custom config and unsupported platform", () => {
  assert.match(renderOneClick({ ...base, setup: { status: "downloading", item: "m.gguf", done: 50, total: 100 } }), /50 %/);
  assert.match(renderOneClick({ ...base, installed: { label: "Qwen 9B" }, running: true, runtime_tag: "b1" }), /running/);
  assert.match(renderOneClick({ ...base, setup: { status: "failed", error: "SHA <bad>" } }), /SHA &lt;bad&gt;/);
  assert.match(renderOneClick({ ...base, custom_config: true }), /own <code>models.toml<\/code>/);
  assert.match(renderOneClick({ ...base, platform_supported: false }), /Windows app/);
});

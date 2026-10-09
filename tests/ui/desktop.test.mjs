import test from "node:test";
import assert from "node:assert/strict";
import { initDesktop, isDesktop, desktopPanel, handleDesktopEvent, resetDesktopForTests } from "../../api/static/js/desktop.js";

function fakeWindow(api) {
  const listeners = {};
  return {
    pywebview: api ? { api } : undefined,
    document: { documentElement: { dataset: {} } },
    addEventListener: (name, fn) => (listeners[name] = fn),
    fire: (name) => listeners[name] && listeners[name](),
  };
}

function fakeBridge(overrides = {}) {
  const calls = [];
  const api = {
    info: async () => ({ version: "0.1.0", keep_core_running: false, core: { status: "running", url: "http://127.0.0.1:8765/", started_by_desktop: true, log: "C:/x/<log>" } }),
    restart_core: async () => calls.push("restart_core"),
    open_logs: async () => calls.push("open_logs"),
    set_keep_core_running: async (v) => calls.push(["keep", v]),
    ...overrides,
  };
  return { api, calls };
}

test("browser without pywebview stays inactive and renders nothing", async () => {
  resetDesktopForTests();
  const win = fakeWindow(null);
  initDesktop(null, win);
  assert.equal(isDesktop(), false);
  assert.equal(win.document.documentElement.dataset.desktop, undefined);
  assert.equal(await desktopPanel(), "");
  assert.equal(await handleDesktopEvent({ type: "click", target: { closest: () => null } }), false);
});

test("bridge that arrives later via pywebviewready is detected once", () => {
  resetDesktopForTests();
  const win = fakeWindow(null);
  let ready = 0;
  initDesktop(() => ready++, win);
  assert.equal(isDesktop(), false);
  win.pywebview = { api: fakeBridge().api };
  win.fire("pywebviewready");
  win.fire("pywebviewready");
  assert.equal(isDesktop(), true);
  assert.equal(ready, 1);
  assert.equal(win.document.documentElement.dataset.desktop, "1");
});

test("panel shows core state, escapes values and offers the controls", async () => {
  resetDesktopForTests();
  initDesktop(null, fakeWindow(fakeBridge().api));
  const html = await desktopPanel();
  assert.match(html, /Desktop app/);
  assert.match(html, /running/);
  assert.match(html, /C:\/x\/&lt;log&gt;/);
  for (const action of ["restart_core", "open_in_browser", "open_data_folder", "open_logs", "stop_core_and_quit"]) {
    assert.match(html, new RegExp(`data-desktop-action="${action}"`));
  }
});

test("actions call only whitelisted bridge methods; checkbox persists preference", async () => {
  resetDesktopForTests();
  const { api, calls } = fakeBridge({ evil: async () => calls.push("evil") });
  initDesktop(null, fakeWindow(api));
  const button = (action) => ({ dataset: { desktopAction: action }, disabled: false });
  const ev = (type, el, sel) => ({ type, target: { closest: (s) => (s === sel ? el : null) } });
  let refreshed = 0;
  assert.equal(await handleDesktopEvent(ev("click", button("restart_core"), "[data-desktop-action]"), async () => refreshed++), true);
  assert.equal(refreshed, 1);
  assert.equal(await handleDesktopEvent(ev("click", button("evil"), "[data-desktop-action]")), false);
  await handleDesktopEvent(ev("change", { checked: true }, "[data-desktop-keep]"));
  assert.deepEqual(calls, ["restart_core", ["keep", true]]);
});

test("bridge errors are shown, not thrown", async () => {
  resetDesktopForTests();
  initDesktop(null, fakeWindow(fakeBridge({ info: async () => { throw new Error("core <down>"); } }).api));
  assert.match(await desktopPanel(), /core &lt;down&gt;/);
});

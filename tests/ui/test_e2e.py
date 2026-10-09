"""Browser-Tests der Oberfläche gegen echte Server (NOVA API + Test-Runtime mit SSE).

Laufen mit Playwright und dem vorinstallierten Chromium; ohne beides werden sie übersprungen.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

playwright_api = pytest.importorskip("playwright.sync_api")
CHROMIUM = next(
    (
        str(p)
        for p in sorted(
            Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")).glob(
                "chromium-*/chrome-linux/chrome"
            )
        )
    ),
    None,
)
if CHROMIUM is None:
    pytest.skip("Chromium nicht gefunden", allow_module_level=True)

import uvicorn  # noqa: E402

from api.app import create_app  # noqa: E402
from api.config import ApiConfig  # noqa: E402
from tests.api.fake_runtime import create_runtime  # noqa: E402

CONFIG = """
[[providers]]
name = "local"
type = "openai_compatible"
base_url = "http://127.0.0.1:{port}"
health_path = "/health"

[[models]]
name = "test-model"
provider = "local"
parameter_count = "8B"
context_length = 8192
reasoning_capability = "good"
coding_capability = "good"
vision_capability = "none"
tool_calling = true
speed = "medium"
memory_requirement = 6
quantization = "Q4_K_M"
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    def __init__(self, app: object, port: int) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> _Server:
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("Server startet nicht")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def servers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    base = tmp_path_factory.mktemp("e2e")
    runtime_port, nova_port, dev_port = _free_port(), _free_port(), _free_port()
    config = base / "models.toml"
    config.write_text(CONFIG.format(port=runtime_port))
    nova = create_app(ApiConfig(models_config=config, data_dir=base / "nova"))
    dev = create_app(ApiConfig(models_config=None, dev_mode=True, data_dir=base / "dev"))
    with (
        _Server(create_runtime("test-model", delay=0.02), runtime_port),
        _Server(nova, nova_port),
        _Server(dev, dev_port),
    ):
        yield {
            "nova": f"http://127.0.0.1:{nova_port}",
            "dev": f"http://127.0.0.1:{dev_port}",
            "base": str(base),
        }


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM)
        yield b
        b.close()


@pytest.fixture
def page(browser: object, servers: dict[str, str]) -> Iterator[object]:
    context = browser.new_context(
        viewport={"width": 1280, "height": 860},  # type: ignore[attr-defined]
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    yield page
    context.close()
    assert not errors, errors


def test_dev_mode_shows_no_model_and_blocks_sending(page: object, servers: dict[str, str]) -> None:
    page.goto(servers["dev"])  # type: ignore[attr-defined]
    expect = playwright_api.expect
    expect(page.locator(".no-model-title")).to_have_text("No local model available.")  # type: ignore[attr-defined]
    expect(page.locator("#status-label")).to_have_text("No local model available")  # type: ignore[attr-defined]
    expect(page.locator("#banner")).to_contain_text("Development mode")  # type: ignore[attr-defined]
    page.fill("#input", "Hello?")  # type: ignore[attr-defined]
    expect(page.locator("#send")).to_be_disabled()  # type: ignore[attr-defined]
    expect(page.locator(".msg")).to_have_count(0)  # type: ignore[attr-defined]


def test_chat_streams_and_shows_real_stats(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    expect(page.locator("#status-dot")).to_have_attribute("data-state", "ok")  # type: ignore[attr-defined]
    page.fill("#input", "Show me a Python function")  # type: ignore[attr-defined]
    page.keyboard.press("Enter")  # type: ignore[attr-defined]
    live = page.locator(".live")  # type: ignore[attr-defined]
    expect(live).to_contain_text("Model: test-model")
    expect(live).to_contain_text("Routing:")
    expect(live).to_contain_text("Status:")
    stats = page.locator(".msg-assistant .stats")  # type: ignore[attr-defined]
    expect(stats).to_contain_text("Model used test-model", timeout=15000)
    for label in ("Generation time", "Tokens", "Tokens/sec", "Routing"):
        expect(stats).to_contain_text(label)
    expect(stats).not_to_contain_text("Verification")  # Chat ohne Verifikation
    expect(page.locator(".code-block .code-lang")).to_have_text("python")  # type: ignore[attr-defined]
    expect(page.locator(".md strong").first).to_have_text("scripted test reply")  # type: ignore[attr-defined]
    expect(page.locator(".conv")).to_have_count(1)  # type: ignore[attr-defined]
    # Kopieren
    page.click(".copy-btn")  # type: ignore[attr-defined]
    expect(page.locator(".copy-btn")).to_contain_text("Copied")  # type: ignore[attr-defined]
    copied = page.evaluate("navigator.clipboard.readText()")  # type: ignore[attr-defined]
    assert copied == "def add(a, b):\n    return a + b"


def test_stop_generation(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    page.fill("#input", "Please write a long text")  # type: ignore[attr-defined]
    page.keyboard.press("Enter")  # type: ignore[attr-defined]
    expect(page.locator(".msg-assistant .md")).to_contain_text("word5")  # type: ignore[attr-defined]
    expect(page.locator("#send")).to_have_class("send-btn stop")  # type: ignore[attr-defined]
    page.click("#send")  # type: ignore[attr-defined]
    expect(page.locator(".stopped-tag")).to_have_text("Generation stopped", timeout=10000)  # type: ignore[attr-defined]
    expect(page.locator("#send")).not_to_have_class("send-btn stop")  # type: ignore[attr-defined]
    text = page.locator(".msg-assistant .md").last.inner_text()  # type: ignore[attr-defined]
    assert "word399" not in text  # wirklich abgebrochen


def test_new_chat_history_and_reopen(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    count = page.locator(".conv").count()  # type: ignore[attr-defined]
    page.fill("#input", "First chat message")  # type: ignore[attr-defined]
    page.keyboard.press("Enter")  # type: ignore[attr-defined]
    expect(page.locator(".msg-assistant .stats")).to_be_visible(timeout=15000)  # type: ignore[attr-defined]
    page.click("#new-chat")  # type: ignore[attr-defined]
    expect(page.locator(".msg")).to_have_count(0)  # type: ignore[attr-defined]
    expect(page.locator("#chat-title")).to_have_text("New chat")  # type: ignore[attr-defined]
    expect(page.locator(".conv")).to_have_count(count + 1)  # type: ignore[attr-defined]
    page.locator(".conv", has_text="First chat message").click()  # type: ignore[attr-defined]
    expect(page.locator(".bubble")).to_have_text("First chat message")  # type: ignore[attr-defined]
    expect(page.locator(".msg-assistant .stats")).to_contain_text("Model used test-model")  # type: ignore[attr-defined]


def test_attachment_via_drop(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    page.evaluate(  # type: ignore[attr-defined]
        """() => {
        const dt = new DataTransfer();
        dt.items.add(new File(['hello from file'], 'notes.txt', {type: 'text/plain'}));
        const main = document.getElementById('main');
        main.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true}));
        main.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true}));
    }"""
    )
    expect(page.locator(".chip-name")).to_have_text("notes.txt")  # type: ignore[attr-defined]
    expect(page.locator("#dropzone")).to_be_hidden()  # type: ignore[attr-defined]
    page.fill("#input", "Read the file")  # type: ignore[attr-defined]
    page.keyboard.press("Enter")  # type: ignore[attr-defined]
    expect(page.locator(".att-pill")).to_contain_text("notes.txt")  # type: ignore[attr-defined]
    expect(page.locator(".msg-assistant .stats")).to_be_visible(timeout=15000)  # type: ignore[attr-defined]
    expect(page.locator("#attachments")).to_be_hidden()  # type: ignore[attr-defined]


def test_settings_and_status_panels(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    page.click("#open-settings")  # type: ignore[attr-defined]
    page.select_option("#set-model", "test-model")  # type: ignore[attr-defined]
    page.fill("input[name=temperature]", "0.3")  # type: ignore[attr-defined]
    page.click("#settings-form button[type=submit]")  # type: ignore[attr-defined]
    expect(page.locator("#model-chip-value")).to_have_text("test-model")  # type: ignore[attr-defined]
    page.click("#open-settings")  # type: ignore[attr-defined]
    page.fill("input[name=agent_workspace]", "/does/not/exist")  # type: ignore[attr-defined]
    page.click("#settings-form button[type=submit]")  # type: ignore[attr-defined]
    error = page.locator("#settings-error")  # type: ignore[attr-defined]
    expect(error).to_contain_text("does not exist")  # Server-Validierung sichtbar
    expect(page.locator("#settings-modal")).to_be_visible()  # type: ignore[attr-defined]
    page.click("#cancel-settings")  # type: ignore[attr-defined]
    page.click("#open-status")  # type: ignore[attr-defined]
    body = page.locator("#status-body")  # type: ignore[attr-defined]
    expect(body).to_contain_text("test-model")
    expect(body).to_contain_text("UNMEASURED")
    expect(body).to_contain_text("RuleBasedRouter")
    page.keyboard.press("Escape")  # type: ignore[attr-defined]
    page.click("#open-settings")  # type: ignore[attr-defined]
    page.select_option("#set-model", "auto")  # type: ignore[attr-defined]
    page.fill("input[name=agent_workspace]", "")  # type: ignore[attr-defined]
    page.click("#settings-form button[type=submit]")  # type: ignore[attr-defined]
    expect(page.locator("#model-chip-value")).to_have_text("Auto")  # type: ignore[attr-defined]


def test_mobile_layout(browser: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    context = browser.new_context(viewport={"width": 390, "height": 800})  # type: ignore[attr-defined]
    page = context.new_page()
    page.goto(servers["nova"])
    expect(page.locator("#sidebar")).not_to_be_in_viewport()
    page.click("#open-sidebar")
    expect(page.locator("#sidebar")).to_be_in_viewport()
    page.click("#scrim", position={"x": 370, "y": 400})
    expect(page.locator("#sidebar")).not_to_be_in_viewport()
    context.close()


def test_no_inline_script_violations(page: object, servers: dict[str, str]) -> None:
    messages: list[str] = []
    page.on("console", lambda m: messages.append(m.text))  # type: ignore[attr-defined]
    page.goto(servers["nova"])  # type: ignore[attr-defined]
    page.wait_for_timeout(500)  # type: ignore[attr-defined]
    assert not [m for m in messages if "Content Security Policy" in m]


def test_setup_screen_without_model(page: object, servers: dict[str, str]) -> None:
    expect = playwright_api.expect
    page.goto(servers["dev"])  # type: ignore[attr-defined]
    page.click("#open-setup")  # type: ignore[attr-defined]
    body = page.locator("#setup-body")  # type: ignore[attr-defined]
    expect(body).to_contain_text("Local model runtime")
    expect(body).to_contain_text("No model catalog yet")
    expect(body).to_contain_text("model-catalog.json")
    expect(body).to_contain_text("NOVA never invents these values")
    page.keyboard.press("Escape")  # type: ignore[attr-defined]
    expect(page.locator("#setup-drawer")).to_be_hidden()  # type: ignore[attr-defined]

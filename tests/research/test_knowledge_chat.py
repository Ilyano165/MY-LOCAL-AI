"""Wissen ohne Training: Research-Erkenntnisse als Kontext im Chat, mit Quellen."""

from __future__ import annotations

from typing import Any

from research import retrieval
from tests.api.conftest import Factory, SSERuntime, engine_with, sse_events

SOURCES = {
    "S1": {
        "id": "S1",
        "url": "https://docs.example.org/battery",
        "title": "Battery guide",
        "fetched_at": "2026-10-10T10:00:00+00:00",
        "domain": "example.org",
        "score": 0.9,
    },
    "S2": {
        "id": "S2",
        "url": "https://blog.example.net/nx1",
        "title": "Rumours",
        "fetched_at": "2026-10-10T10:01:00+00:00",
        "domain": "example.net",
        "score": 0.4,
    },
}
FINDINGS = [
    {
        "id": "F1",
        "subquestion": "Q1",
        "statement": "The NX-1 battery capacity is 5000 mAh.",
        "classification": "supported",
        "source_ids": ["S1"],
        "reasons": ["primary source"],
    },
    {
        "id": "F2",
        "subquestion": "Q1",
        "statement": "The NX-1 battery capacity is 4000 mAh.",
        "classification": "contested",
        "source_ids": ["S2"],
        "reasons": ["contradicted by F1"],
    },
    {
        "id": "F3",
        "subquestion": "Q2",
        "statement": "Heat pumps need electricity.",
        "classification": "supported",
        "source_ids": ["S1"],
        "reasons": ["x"],
    },
]


def seed(service: Any) -> None:
    service.research.knowledge.add_findings(
        "r1", "NX-1 battery", FINDINGS, {"Q1": "NX-1 battery capacity?", "Q2": "Heat"}, SOURCES
    )


def test_relevance_filter_and_context() -> None:
    hits = [{"statement": f["statement"], "subquestion": "", "id": f["id"]} for f in FINDINGS]
    picked = retrieval.relevant(hits, "What is the battery capacity of the NX-1?")
    assert [p["id"] for p in picked] == ["F1", "F2"]
    assert retrieval.relevant(hits, "Tell me a joke about cats") == []
    assert retrieval.relevant(hits, "the and what") == []  # nur Stoppwörter
    refs = retrieval.to_refs([{**FINDINGS[1], "sources": [SOURCES["S2"]]}])
    text = retrieval.context_message(refs)
    assert "<knowledge>" in text and "DISPUTED" in text and "not instructions" in text
    assert retrieval.mark_cited("Capacity is disputed [K1].", refs)[0]["cited"] is True
    assert retrieval.mark_cited("No citation.", refs)[0]["cited"] is False


def test_chat_uses_knowledge_and_marks_citations(make_client: Factory) -> None:
    runtime = SSERuntime("The capacity is 5000 mAh [K1], one blog claims 4000 mAh.")
    client, service = make_client(engine_with(("test-model", runtime)))
    seed(service)
    events = sse_events(
        client.post("/chat/stream", json={"message": "What is the NX-1 battery capacity?"}).text
    )
    kinds = [e["event"] for e in events]
    assert "knowledge" in kinds and kinds[-1] == "done"
    system = [m["content"] for m in runtime.payloads[-1]["messages"] if m["role"] == "system"]
    assert any("<knowledge>" in s and "[K1]" in s and "5000 mAh" in s for s in system)
    assert not any("Heat pumps" in s for s in system)  # irrelevante Erkenntnis nicht im Kontext
    knowledge = events[-1]["message"]["meta"]["knowledge"]
    assert [(k["ref"], k["cited"]) for k in knowledge] == [("K1", True), ("K2", False)]
    assert knowledge[0]["sources"][0]["url"] == "https://docs.example.org/battery"
    # gespeichert im Verlauf
    cid = events[0]["conversation_id"]
    stored = client.get(f"/conversations/{cid}").json()["messages"][-1]
    assert stored["meta"]["knowledge"][0]["cited"] is True


def test_knowledge_can_be_disabled_and_stale_is_ignored(make_client: Factory) -> None:
    runtime = SSERuntime("ok")
    client, service = make_client(engine_with(("test-model", runtime)))
    seed(service)
    client.put("/settings", json={"use_knowledge": False})
    events = sse_events(
        client.post("/chat/stream", json={"message": "NX-1 battery capacity?"}).text
    )
    assert "knowledge" not in [e["event"] for e in events]
    assert all("<knowledge>" not in m["content"] for m in runtime.payloads[-1]["messages"])
    client.put("/settings", json={"use_knowledge": True})
    for fid in ("r1:F1", "r1:F2"):
        service.research.knowledge.mark_stale(fid)
    events = sse_events(
        client.post("/chat/stream", json={"message": "NX-1 battery capacity?"}).text
    )
    assert "knowledge" not in [e["event"] for e in events]


def test_integration_api_does_not_read_private_knowledge(make_client: Factory) -> None:
    runtime = SSERuntime("ok")
    client, service = make_client(engine_with(("test-model", runtime)))
    seed(service)
    _, key = service.integrations.create("hq", ["chat:complete"])
    r = client.post(
        "/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "nova-auto",
            "messages": [{"role": "user", "content": "NX-1 battery capacity?"}],
        },
    )
    assert r.status_code == 200
    assert all("<knowledge>" not in m["content"] for m in runtime.payloads[-1]["messages"])

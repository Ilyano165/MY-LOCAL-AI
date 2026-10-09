"""Beispiel: serverseitige Integration von NOVA in IC WARE HQ.

Das HQ-Frontend ruft **nur** diesen HQ-Endpunkt auf (mit der normalen HQ-Anmeldung). Der
NOVA-Schlüssel liegt ausschließlich auf dem HQ-Server (Umgebungsvariable) – nie im Browser.

    Browser (HQ-Frontend) → HQ-Backend (/hq/assistant) → NOVA (/api/v1) → Router → Modell

Start (Beispiel):
    NOVA_BASE_URL=http://127.0.0.1:8765 NOVA_API_KEY=… uvicorn hq_backend:app --port 9000

Die HQ-eigene Benutzeranmeldung ist hier nur als Platzhalter (``require_hq_user``) angedeutet,
weil das HQ-Authentifizierungssystem nicht Teil von NOVA ist.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from nova_client import (
    NovaAuthError,
    NovaClient,
    NovaConnectionError,
    NovaError,
    NovaPermissionError,
    NovaRateLimitError,
    NovaTimeoutError,
    NovaUnavailableError,
)
from pydantic import BaseModel, Field

#: Metadaten, die HQ an seine Oberfläche weitergeben darf (alles andere bleibt im Backend)
ALLOWED_METADATA = ("model", "routing_category", "request_id")


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    context: str | None = Field(default=None, max_length=50_000)


class AssistantResponse(BaseModel):
    answer: str
    metadata: dict[str, Any]


def require_hq_user(authorization: str | None = Header(default=None)) -> str:
    """Platzhalter für die HQ-eigene Anmeldung (Session/SSO). Hier: Bearer muss vorhanden sein."""
    if not authorization:
        raise HTTPException(401, "HQ login required")
    return "hq-user"


def create_app(client: NovaClient | None = None) -> FastAPI:
    app = FastAPI(title="IC WARE HQ – NOVA integration example")
    app.state.nova = client

    def nova() -> NovaClient:
        if app.state.nova is None:
            app.state.nova = NovaClient()  # liest NOVA_BASE_URL/NOVA_API_KEY aus der Umgebung
        return app.state.nova  # type: ignore[no-any-return]

    @app.post("/hq/assistant", response_model=AssistantResponse)
    def assistant(body: AssistantRequest, _user: str = Depends(require_hq_user)) -> Any:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are the IC WARE HQ assistant. Answer concisely."}
        ]
        if body.context:
            messages.append({"role": "user", "content": f"Context:\n{body.context}"})
        messages.append({"role": "user", "content": body.question})
        try:
            result = nova().chat(messages)
        except NovaUnavailableError as exc:
            raise HTTPException(
                503, "The local AI service has no model available right now."
            ) from exc
        except NovaTimeoutError as exc:
            raise HTTPException(504, "The AI service took too long – please try again.") from exc
        except NovaRateLimitError as exc:
            raise HTTPException(429, "Too many AI requests – please wait a moment.") from exc
        except (NovaAuthError, NovaPermissionError) as exc:
            # Konfigurationsfehler auf HQ-Seite: Details ins Server-Log, nicht zum Benutzer
            raise HTTPException(
                502, f"AI integration misconfigured (ref {exc.request_id})"
            ) from exc
        except NovaConnectionError as exc:
            raise HTTPException(503, "The AI service is not reachable.") from exc
        except NovaError as exc:
            raise HTTPException(502, f"AI request failed (ref {exc.request_id})") from exc
        metadata = {
            "model": result.model,
            "routing_category": result.routing.get("category"),
            "request_id": result.request_id,
        }
        return AssistantResponse(
            answer=result.content, metadata={k: metadata[k] for k in ALLOWED_METADATA}
        )

    return app


app = create_app()

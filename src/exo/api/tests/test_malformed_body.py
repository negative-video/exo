"""A request body that doesn't parse is answered with a 400 that doesn't quote it, and
isn't logged: it is request content."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from exo.api.main import API

MARKER = "marker-61b0d4"


def test_a_malformed_ollama_chat_body_is_a_400_that_does_not_quote_it() -> None:
    api = object.__new__(API)
    app = FastAPI()
    app.post("/ollama/api/chat", response_model=None)(api.ollama_chat)
    logged: list[str] = []
    handler_id = logger.add(logged.append, format="{message}")
    try:
        response = TestClient(app).post(
            "/ollama/api/chat",
            content=f'{{"model": "m", "messages": "{MARKER}"}}',
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 400
    assert MARKER not in response.text
    assert all(MARKER not in line for line in logged)

"""Tests that other machines can only list models and chat, with the key, and that this
machine can still use everything."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.access_gate import AccessGate

KEY = "test-key"
REMOTE = ("100.84.78.85", 50000)
LOCAL = ("127.0.0.1", 50000)


def _client(client: tuple[str, int], api_key: str | None = KEY) -> TestClient:
    app = FastAPI()
    app.get("/node_id")(lambda: "node-a")
    app.get("/v1/models")(lambda: {"data": "none"})
    app.post("/v1/chat/completions")(lambda: {"ok": True})
    app.get("/state")(lambda: {"state": "secret"})
    app.post("/instance")(lambda: {"ok": True})

    def _delete_download(node_id: str, model_id: str) -> dict[str, bool]:
        return {"ok": True}

    app.delete("/download/{node_id}/{model_id:path}")(_delete_download)
    app.get("/")(lambda: "dashboard")
    app.add_middleware(AccessGate, api_key=api_key)
    return TestClient(app, client=client)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_this_machine_can_use_everything_without_a_key() -> None:
    client = _client(LOCAL)
    assert client.get("/state").status_code == 200
    assert client.post("/instance").status_code == 200
    assert client.delete("/download/node-b/org/model").status_code == 200
    assert client.get("/").status_code == 200
    assert client.post("/v1/chat/completions").status_code == 200


def test_ipv6_loopback_counts_as_this_machine() -> None:
    assert _client(("::1", 50000)).get("/state").status_code == 200


def test_other_machines_can_chat_with_the_key() -> None:
    client = _client(REMOTE)
    assert client.get("/v1/models", headers=_auth(KEY)).status_code == 200
    assert client.post("/v1/chat/completions", headers=_auth(KEY)).status_code == 200


def test_other_machines_cannot_chat_without_the_right_key() -> None:
    client = _client(REMOTE)
    assert client.post("/v1/chat/completions").status_code == 401
    assert (
        client.post("/v1/chat/completions", headers=_auth("wrong")).status_code == 401
    )
    response = client.get("/v1/models", headers=_auth("wrong"))
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Missing or wrong API key"


def test_with_no_key_set_other_machines_cannot_chat_at_all() -> None:
    client = _client(REMOTE, api_key=None)
    assert client.post("/v1/chat/completions").status_code == 401
    assert client.post("/v1/chat/completions", headers=_auth("")).status_code == 401
    assert client.post("/v1/chat/completions", headers=_auth("None")).status_code == 401


def test_other_machines_are_refused_everything_else_even_with_the_key() -> None:
    client = _client(REMOTE)
    for method, path in [
        ("GET", "/state"),
        ("POST", "/instance"),
        ("DELETE", "/download/node-b/org/model"),
        ("GET", "/"),
        ("GET", "/v1/models/"),
        ("GET", "/v1/chat/completions"),
        ("OPTIONS", "/v1/chat/completions"),
    ]:
        response = client.request(method, path, headers=_auth(KEY))
        assert response.status_code == 403, (method, path)
    assert "secret" not in client.get("/state", headers=_auth(KEY)).text


def test_peers_can_still_ask_for_the_node_id_without_a_key() -> None:
    assert _client(REMOTE, api_key=None).get("/node_id").status_code == 200

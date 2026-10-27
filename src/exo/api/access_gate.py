import hmac
import ipaddress
import json
from typing import cast, final

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send

# What another machine may use, and only with the API key: listing the models and chatting.
# That is all Open WebUI needs.
KEYED_ROUTES = {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}

# What another machine may use without a key. Every node calls GET /node_id on its peers,
# on each of their addresses, to find which links it can reach them over (net_profile.py);
# placement is built on those links. It answers with the node's ID and nothing else.
OPEN_ROUTES = {("GET", "/node_id")}


@final
class AccessGate:
    """Decides which requests from other machines the API answers.

    A request from this machine can use every route: the dashboard (over `ssh -L`), the
    autoloader and launch.py all run here. Another machine may only call the routes above,
    and gets 403 for everything else, such as creating or deleting instances, downloads and
    their deletes, /models/add, /state, traces, images and the dashboard itself. Without an
    API key set, the keyed routes are closed to other machines as well.

    This trusts the connection's source address, so nothing on this machine may forward
    outside requests to the API (a reverse proxy, `tailscale serve`): they would all look
    local.
    """

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key.encode() if api_key else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or _is_from_this_machine(scope):
            await self.app(scope, receive, send)
            return

        host = _client_host(scope)
        if scope["type"] != "http":
            # No route takes a websocket, so one from another machine is always turned away
            logger.info(f"API: refused a websocket from {host}")
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return

        route = (cast(str, scope["method"]), cast(str, scope["path"]))
        if route in OPEN_ROUTES:
            await self.app(scope, receive, send)
            return
        if route in KEYED_ROUTES:
            if self.api_key is not None and hmac.compare_digest(
                _bearer_token(scope), self.api_key
            ):
                await self.app(scope, receive, send)
                return
            logger.info(
                f"API: refused {route[0]} {route[1]} from {host}: missing or wrong API key"
            )
            await _send_error(send, 401, "Missing or wrong API key")
            return

        logger.info(
            f"API: refused {route[0]} {route[1]} from {host}: only open to this machine"
        )
        await _send_error(
            send, 403, "This route is only open to the machine exo runs on"
        )


def _client_host(scope: Scope) -> str:
    client = cast(tuple[str, int] | None, scope.get("client"))
    return client[0] if client else ""


def _is_from_this_machine(scope: Scope) -> bool:
    """Whether the request came over loopback.

    A client address that isn't an IP address (no client at all, or a Unix socket) raises
    ValueError; such a request is treated as coming from another machine, so it is refused.
    """
    try:
        return ipaddress.ip_address(_client_host(scope)).is_loopback
    except ValueError:
        return False


def _bearer_token(scope: Scope) -> bytes:
    for name, value in cast(list[tuple[bytes, bytes]], scope["headers"]):
        if name.lower() == b"authorization" and value[:7].lower() == b"bearer ":
            return value[7:].strip()
    return b""


async def _send_error(send: Send, status: int, message: str) -> None:
    # Same shape as the API's other errors (OpenAI style), so clients can show the message
    body = json.dumps(
        {
            "error": {
                "message": message,
                "type": "access_denied",
                "param": None,
                "code": status,
            }
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})

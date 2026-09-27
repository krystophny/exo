import asyncio

import pytest

from exo.api.private_auth import PrivateAPIAuth


def test_authenticated_dispatch_and_identity_discovery(tmp_path):
    key = tmp_path / "key"
    key.write_text("x" * 48)
    key.chmod(0o600)
    dispatched = []

    async def app(scope, receive, send):
        dispatched.append(scope["path"])
        await send({"type": "http.response.start", "status": 200})

    middleware = PrivateAPIAuth(app, str(key))

    async def request(path, headers=(), method="GET", kind="http"):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            messages.append(message)

        await middleware(
            {"type": kind, "path": path, "method": method, "headers": headers},
            receive,
            send,
        )
        return messages

    for path in ["/v1/chat/completions", "/state", "/instance"]:
        assert asyncio.run(request(path))[0]["status"] == 401
    assert asyncio.run(request("/node_id", method="POST"))[0]["status"] == 401
    assert asyncio.run(request("/node_id"))[0]["status"] == 200
    correct = (b"authorization", b"Bearer " + b"x" * 48)
    assert asyncio.run(request("/state", [correct]))[0]["status"] == 200
    assert asyncio.run(request("/state", [correct, correct]))[0]["status"] == 401
    assert asyncio.run(request("/ws", kind="websocket"))[0]["code"] == 1008
    assert dispatched == ["/node_id", "/state"]


def test_exposed_key_rejected(tmp_path):
    key = tmp_path / "key"
    key.write_text("x" * 48)
    key.chmod(0o644)
    with pytest.raises(ValueError):
        PrivateAPIAuth(None, str(key))


@pytest.mark.parametrize("contents", ["", "short", "x" * 40 + "\nsecond-line"])
def test_invalid_private_key_fails_closed(tmp_path, contents):
    key = tmp_path / "key"
    key.write_text(contents)
    key.chmod(0o600)
    with pytest.raises(ValueError):
        PrivateAPIAuth(None, str(key))

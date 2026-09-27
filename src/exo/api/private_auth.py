"""Optional private transport authentication without changing topology discovery."""

import hmac
import os
import stat
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send


class PrivateAPIAuth:
    def __init__(self, app: ASGIApp, credential_file: str) -> None:
        self.app = app
        path = Path(credential_file)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("private API credential must be a regular mode0600 file")
        key = path.read_bytes().strip()
        if len(key) < 32 or b"\n" in key or b"\r" in key:
            raise ValueError("invalid private API credential")
        self.authorization = b"Bearer " + key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = cast(str, scope["type"])
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        discovery = (
            kind == "http"
            and scope.get("method") == "GET"
            and scope.get("path") == "/node_id"
        )
        values = [
            v
            for k, v in cast(list[tuple[bytes, bytes]], scope.get("headers", []))
            if k.lower() == b"authorization"
        ]
        authorized = len(values) == 1 and hmac.compare_digest(
            values[0], self.authorization
        )
        if discovery or authorized:
            return await self.app(scope, receive, send)
        if kind == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        body = b'{"error":"private_backend_unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install_private_auth(app: FastAPI) -> None:
    credential_file = os.environ.get("EXO_API_KEY_FILE")
    if credential_file:
        app.add_middleware(PrivateAPIAuth, credential_file=credential_file)

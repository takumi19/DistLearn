"""Minimal HTTP config server: bootstrap serves run config, followers fetch it.

The server runs on grpc_port + 1 and responds to GET /run-config with JSON.
This removes the need for any token distribution — followers just need the
bootstrap address and their own node id.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


_CONFIG_PATH = "/run-config"


class RunConfigServer:
    """Thread-safe HTTP server that serves a single JSON payload."""

    def __init__(self, host: str, port: int, payload: dict[str, Any]) -> None:
        self._payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        self._server = HTTPServer((host, port), self._make_handler())
        self._thread: threading.Thread | None = None

    def _make_handler(self):
        payload_bytes = self._payload_bytes

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == _CONFIG_PATH:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass  # suppress stdlib HTTP logging

        return _Handler

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def address(self) -> str:
        host, port = self._server.server_address
        return f"{host}:{port}"


def fetch_run_config(
    bootstrap_host: str,
    config_port: int,
    *,
    timeout_s: float = 30.0,
    retry_interval_s: float = 1.5,
) -> dict[str, Any]:
    """Fetch run config from the bootstrap HTTP config server, with retry."""
    url = f"http://{bootstrap_host}:{config_port}{_CONFIG_PATH}"
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            last_exc = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(retry_interval_s, remaining))

    raise RuntimeError(
        f"Could not fetch run config from {url} within {timeout_s}s: {last_exc}"
    )


def config_port_for(grpc_port: int) -> int:
    """Config HTTP port is always grpc_port + 1."""
    return grpc_port + 1

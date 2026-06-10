"""Minimal HTTP config server: bootstrap serves run config, followers fetch it.

The server runs on grpc_port + 1 and responds to GET /run-config with JSON.
Also handles peer registration for soft-barrier consensus:
  POST /sync/ready  {"node_id": "...", "reachable": [...]}
  GET  /sync/participants  → {"participants": [...]} once agreed, or 202 while pending
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
_SYNC_READY_PATH = "/sync/ready"
_SYNC_PARTICIPANTS_PATH = "/sync/participants"

# Module-level singleton so sync_barrier can access the server without
# passing it through the call stack.
_current_server: "RunConfigServer | None" = None


def get_current_server() -> "RunConfigServer | None":
    return _current_server


class RunConfigServer:
    """Thread-safe HTTP server that serves run config and coordinates soft barrier."""

    def __init__(self, host: str, port: int, payload: dict[str, Any]) -> None:
        self._payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        # Peer registration state (used for soft barrier consensus).
        # _registrations: {node_id: set_of_reachable_node_ids}
        self._registrations: dict[str, set[str]] = {}
        self._participants: list[str] | None = None  # None = not yet decided
        self._lock = threading.Lock()
        self._server = HTTPServer((host, port), self._make_handler())
        self._thread: threading.Thread | None = None

    def _make_handler(self):
        payload_bytes = self._payload_bytes
        srv = self  # capture server instance for handler callbacks

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == _CONFIG_PATH:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                elif self.path == _SYNC_PARTICIPANTS_PATH:
                    with srv._lock:
                        participants = srv._participants
                    if participants is None:
                        # Not decided yet — tell caller to retry
                        self.send_response(202)
                        self.end_headers()
                    else:
                        body = json.dumps({"participants": participants}).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == _SYNC_READY_PATH:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        node_id = data["node_id"]
                        reachable = set(data.get("reachable", []))
                    except Exception:
                        self.send_error(400)
                        return
                    with srv._lock:
                        srv._registrations[node_id] = reachable
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass  # suppress stdlib HTTP logging

        return _Handler

    def decide_participants(
        self,
        expected_node_ids: list[str],
        *,
        min_fraction: float = 0.7,
        timeout_s: float = 120.0,
    ) -> list[str]:
        """
        Wait until all expected nodes register or timeout, then compute the agreed
        participant set and store it for GET /sync/participants responses.

        A node is included if:
          - it registered itself (i.e., is alive), AND
          - at least min_fraction of all registrants reported it as reachable.

        Called once by bootstrap after starting the config server.
        Returns the agreed participant list.
        """
        expected = set(expected_node_ids)
        deadline = time.monotonic() + timeout_s

        # Wait for all nodes to register (or deadline)
        while time.monotonic() < deadline:
            with self._lock:
                registered = set(self._registrations)
            if registered >= expected:
                break
            time.sleep(1.0)

        with self._lock:
            registrations = dict(self._registrations)

        # Nodes that registered themselves are candidates
        candidates = set(registrations)

        # A candidate is accepted if ≥ min_fraction of registrants say it's reachable
        n_registrants = len(registrations)
        if n_registrants == 0:
            participants: list[str] = []
        else:
            threshold = max(1, int(n_registrants * min_fraction))
            participants = []
            for node_id in candidates:
                votes = sum(
                    1 for reporter_id, reachable in registrations.items()
                    if reporter_id != node_id and node_id in reachable
                )
                # Also count self-reported reachability of others toward node_id's votes:
                # if at least `threshold` peers could ping node_id, include it.
                if votes >= threshold - 1:  # -1 because node doesn't vote for itself
                    participants.append(node_id)

        participants.sort()
        with self._lock:
            self._participants = participants

        return participants

    def start(self) -> None:
        global _current_server
        _current_server = self
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


def post_sync_ready(
    bootstrap_host: str,
    config_port: int,
    self_node_id: str,
    reachable_node_ids: list[str],
    *,
    timeout_s: float = 10.0,
) -> None:
    """Report this node's reachable peers to bootstrap for soft-barrier consensus."""
    url = f"http://{bootstrap_host}:{config_port}{_SYNC_READY_PATH}"
    body = json.dumps({"node_id": self_node_id, "reachable": reachable_node_ids}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Content-Length", str(len(body)))
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp.read()


def fetch_sync_participants(
    bootstrap_host: str,
    config_port: int,
    *,
    timeout_s: float = 120.0,
    retry_interval_s: float = 2.0,
) -> list[str]:
    """Poll bootstrap until the agreed participant list is available."""
    url = f"http://{bootstrap_host}:{config_port}{_SYNC_PARTICIPANTS_PATH}"
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read())["participants"]
                # 202 = not decided yet, keep polling
        except Exception as exc:
            last_exc = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(retry_interval_s, remaining))

    raise RuntimeError(
        f"Could not fetch sync participants from {url} within {timeout_s}s: {last_exc}"
    )


def config_port_for(grpc_port: int) -> int:
    """Config HTTP port is always grpc_port + 1."""
    return grpc_port + 1

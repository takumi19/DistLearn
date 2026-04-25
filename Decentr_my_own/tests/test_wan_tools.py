from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.comm.server import PeerServer
from decentr_my_own.config.loader import load_resolved_config, load_yaml
from decentr_my_own.deployment.wan import build_launch_plan, build_wan_preflight, probe_neighbors


class WanToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = PROJECT_ROOT / "configs" / "cluster.wan-2node.example.yaml"
        self.training_sync = PROJECT_ROOT / "configs" / "training.wan-sync.example.yaml"

    def test_launch_plan_uses_platform_specific_commands(self) -> None:
        resolved = load_resolved_config(self.cluster, self.training_sync, "node-mac")
        plan = build_launch_plan(
            cluster_path=self.cluster,
            training_path=self.training_sync,
            cluster=resolved.cluster,
            training=resolved.training,
        )

        commands = {item["node_id"]: item["run_command"] for item in plan["per_node"]}
        self.assertIn("PYTHONPATH=Decentr_my_own python3", commands["node-mac"])
        self.assertIn("$env:PYTHONPATH='Decentr_my_own'; python -m", commands["node-win"])

    def test_preflight_warns_for_loopback_host_in_tailscale_mode(self) -> None:
        payload = load_yaml(self.cluster)
        payload["nodes"][0]["host"] = "127.0.0.1"

        with tempfile.TemporaryDirectory() as tmp_dir:
            cluster_path = Path(tmp_dir) / "cluster.loopback.yaml"
            with cluster_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            resolved = load_resolved_config(cluster_path, self.training_sync, "node-mac")
            report = build_wan_preflight(resolved, check_dns=False)

        warning_blob = "\n".join(report["warnings"])
        self.assertIn("loopback", warning_blob)

    def test_probe_neighbors_reaches_running_peer(self) -> None:
        port_a = _find_free_port()
        port_b = _find_free_port()
        cluster_payload = {
            "cluster_name": "probe-test",
            "transport": "grpc",
            "overlay_network": "none",
            "tls_enabled": False,
            "bootstrap_node_id": "node-a",
            "nodes": [
                {
                    "id": "node-a",
                    "host": "127.0.0.1",
                    "bind_host": "127.0.0.1",
                    "port": port_a,
                    "platform": "macos",
                    "neighbors": ["node-b"],
                },
                {
                    "id": "node-b",
                    "host": "127.0.0.1",
                    "bind_host": "127.0.0.1",
                    "port": port_b,
                    "platform": "windows",
                    "neighbors": ["node-a"],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            cluster_path = Path(tmp_dir) / "cluster.yaml"
            with cluster_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cluster_payload, handle, sort_keys=False)

            resolved = load_resolved_config(cluster_path, self.training_sync, "node-a")
            server = PeerServer(node_id="node-b", host="127.0.0.1", port=port_b)
            server.start()
            try:
                result = probe_neighbors(resolved, timeout_s=2.0, include_state=True)
            finally:
                server.stop(grace=0.0)

        self.assertEqual(result["reachable_count"], 1)
        self.assertTrue(result["results"][0]["reachable"])
        self.assertIn("remote_state", result["results"][0])


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()

"""Tests for create-run / join-run UX: token encoding, inventory loading, CLI output."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.cli import build_parser, main
from decentr_my_own.config.loader import load_inventory
from decentr_my_own.deployment.wan import (
    build_create_run_output,
    decode_run_token,
    encode_run_token,
)


_INVENTORY_YAML = """\
cluster:
  name: test-cluster
  overlay: none
  bootstrap_node: node-a
nodes:
  node-a:
    host: 127.0.0.1
    port: 59010
    bind_host: 0.0.0.0
    platform: linux
    weight: 1.0
    resources:
      cpu_cores: 4
      relative_speed: 1.0
  node-b:
    host: 127.0.0.1
    port: 59011
    bind_host: 0.0.0.0
    platform: linux
    weight: 0.5
    resources:
      cpu_cores: 2
      relative_speed: 0.5
training:
  mode: async
  device_preference: [cpu]
  epochs: 10
  batch_size: 32
  lr: 0.01
  scheduler_mode: static
  storage_mode: replicated
"""


class TestInventoryLoading(unittest.TestCase):
    def test_load_inventory_creates_cluster_and_training(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            cluster, training = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(cluster.cluster_name, "test-cluster")
        self.assertEqual(len(cluster.nodes), 2)
        self.assertEqual(cluster.bootstrap_node_id, "node-a")

        node_ids = {node.id for node in cluster.nodes}
        self.assertIn("node-a", node_ids)
        self.assertIn("node-b", node_ids)

        node_a = cluster.get_node("node-a")
        self.assertIn("node-b", node_a.neighbors)
        node_b = cluster.get_node("node-b")
        self.assertIn("node-a", node_b.neighbors)

        self.assertEqual(training.mode, "async")
        self.assertEqual(training.device_preference, ["cpu"])
        self.assertTrue(training.async_config.enabled)
        self.assertFalse(training.sync.enabled)
        self.assertEqual(training.optimization.epochs, 10)
        self.assertEqual(training.optimization.batch_size, 32)

    def test_inventory_sets_relative_speed(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            cluster, _ = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        node_a = cluster.get_node("node-a")
        node_b = cluster.get_node("node-b")
        self.assertAlmostEqual(node_a.resources.relative_speed, 1.0)
        self.assertAlmostEqual(node_b.resources.relative_speed, 0.5)

    def test_inventory_missing_nodes_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("cluster:\n  name: empty\n")
            tmp_path = Path(fh.name)
        try:
            with self.assertRaises(ValueError):
                load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestRunTokenEncoding(unittest.TestCase):
    def _make_cluster_training(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)
        try:
            return load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_encode_decode_roundtrip(self) -> None:
        cluster, training = self._make_cluster_training()
        token = encode_run_token(
            cluster=cluster,
            training=training,
            run_name="demo-001",
            epochs=10,
        )
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 10)

        decoded = decode_run_token(token)
        self.assertIn("cluster", decoded)
        self.assertIn("training", decoded)
        self.assertEqual(decoded["run_name"], "demo-001")
        self.assertEqual(decoded["epochs"], 10)
        self.assertEqual(decoded["cluster"]["cluster_name"], "test-cluster")
        self.assertEqual(decoded["training"]["mode"], "async")

    def test_decoded_token_is_valid_config(self) -> None:
        from decentr_my_own.config.loader import load_cluster_config_inline, load_training_config_inline
        import base64

        cluster, training = self._make_cluster_training()
        token = encode_run_token(
            cluster=cluster, training=training, run_name="x", epochs=5
        )
        decoded = decode_run_token(token)
        cluster_b64 = base64.urlsafe_b64encode(
            json.dumps(decoded["cluster"], separators=(",", ":")).encode()
        ).decode()
        training_b64 = base64.urlsafe_b64encode(
            json.dumps(decoded["training"], separators=(",", ":")).encode()
        ).decode()
        cluster2 = load_cluster_config_inline(cluster_b64)
        training2 = load_training_config_inline(training_b64)
        self.assertEqual(cluster2.cluster_name, "test-cluster")
        self.assertEqual(training2.mode, "async")


class TestBuildCreateRunOutput(unittest.TestCase):
    def _make_cluster_training(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)
        try:
            return load_inventory(tmp_path), tmp_path
        finally:
            pass  # don't unlink here, caller does it

    def test_create_run_output_structure(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            cluster, training = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        result = build_create_run_output(
            cluster_path=None,
            training_path=None,
            cluster=cluster,
            training=training,
            run_name="run-001",
            epochs=5,
        )

        self.assertEqual(result["run_name"], "run-001")
        self.assertEqual(result["epochs"], 5)
        self.assertEqual(result["bootstrap_node_id"], "node-a")
        self.assertIn("run_token", result)
        self.assertIn("bootstrap_command", result)
        self.assertIn("join_commands", result)
        self.assertIn("report_command", result)

        join_commands = result["join_commands"]
        self.assertIn("node-b", join_commands)
        self.assertNotIn("node-a", join_commands)

        self.assertIn("--self-node node-b", join_commands["node-b"])
        self.assertIn("join-run", join_commands["node-b"])
        self.assertIn("--run-token", join_commands["node-b"])

    def test_bootstrap_command_contains_self_node(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            cluster, training = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        result = build_create_run_output(
            cluster_path=None,
            training_path=None,
            cluster=cluster,
            training=training,
            run_name="r",
            epochs=3,
        )

        self.assertIn("--self-node node-a", result["bootstrap_command"])
        self.assertIn("run-async-node", result["bootstrap_command"])

    def test_node_subset_selection(self) -> None:
        inventory_yaml = """\
cluster:
  name: three-node
  overlay: none
  bootstrap_node: n1
nodes:
  n1:
    host: 127.0.0.1
    port: 59020
    bind_host: 0.0.0.0
    platform: linux
  n2:
    host: 127.0.0.1
    port: 59021
    bind_host: 0.0.0.0
    platform: linux
  n3:
    host: 127.0.0.1
    port: 59022
    bind_host: 0.0.0.0
    platform: linux
training:
  mode: async
  epochs: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(inventory_yaml)
            tmp_path = Path(fh.name)

        try:
            cluster, training = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        result = build_create_run_output(
            cluster_path=None,
            training_path=None,
            cluster=cluster,
            training=training,
            run_name="r",
            epochs=5,
            selected_node_ids=["n1", "n2"],
        )

        self.assertEqual(set(result["selected_nodes"]), {"n1", "n2"})
        self.assertIn("n2", result["join_commands"])
        self.assertNotIn("n3", result["join_commands"])

    def test_report_command_includes_run_name(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            cluster, training = load_inventory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        result = build_create_run_output(
            cluster_path=None,
            training_path=None,
            cluster=cluster,
            training=training,
            run_name="my-run",
            epochs=3,
        )
        self.assertIn("my-run", result["report_command"])


class TestCreateRunCLI(unittest.TestCase):
    def test_create_run_cli_with_inventory(self) -> None:
        import io
        from contextlib import redirect_stdout

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_INVENTORY_YAML)
            tmp_path = Path(fh.name)

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main([
                    "create-run",
                    "--inventory", str(tmp_path),
                    "--run-name", "cli-test",
                    "--epochs", "3",
                ])
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(rc, 0)
        result = json.loads(buf.getvalue())
        self.assertEqual(result["run_name"], "cli-test")
        self.assertEqual(result["epochs"], 3)
        self.assertIn("run_token", result)
        self.assertIn("bootstrap_command", result)
        self.assertIn("join_commands", result)

    def test_create_run_cli_node_subset(self) -> None:
        import io
        from contextlib import redirect_stdout

        inventory_yaml = """\
cluster:
  name: big-cluster
  overlay: none
  bootstrap_node: a
nodes:
  a:
    host: 127.0.0.1
    port: 59030
    bind_host: 0.0.0.0
    platform: linux
  b:
    host: 127.0.0.1
    port: 59031
    bind_host: 0.0.0.0
    platform: linux
  c:
    host: 127.0.0.1
    port: 59032
    bind_host: 0.0.0.0
    platform: linux
training:
  mode: async
  epochs: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(inventory_yaml)
            tmp_path = Path(fh.name)

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main([
                    "create-run",
                    "--inventory", str(tmp_path),
                    "--run-name", "subset-run",
                    "--nodes", "a,b",
                ])
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(rc, 0)
        result = json.loads(buf.getvalue())
        self.assertEqual(set(result["selected_nodes"]), {"a", "b"})
        self.assertNotIn("c", result["join_commands"])


if __name__ == "__main__":
    unittest.main()

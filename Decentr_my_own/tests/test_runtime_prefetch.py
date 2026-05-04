from __future__ import annotations

import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.comm.client import PeerClient
from decentr_my_own.comm.server import PeerServer
from decentr_my_own.config.loader import load_resolved_config, load_training_config, load_yaml
from decentr_my_own.data.runtime import AdaptiveMicroShardRuntime
from decentr_my_own.data.shards import build_dataset_shards


class RuntimePrefetchTests(unittest.TestCase):
    def test_bootstrap_builds_missing_manifest_for_adaptive_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = self._write_runtime_configs(
                tmp_path,
                prefetch_shards=0,
            )
            resolved = load_resolved_config(cluster_path, training_paths[node_ids[0]], node_ids[0])
            manifest_path = Path(resolved.training.dataset.manifest_path)
            self.assertFalse(manifest_path.exists())

            server = PeerServer(
                node_id=node_ids[0],
                host="127.0.0.1",
                port=resolved.self_node.port,
                transfer_chunk_bytes=256,
            )
            runtime = AdaptiveMicroShardRuntime(
                resolved=resolved,
                server=server,
                timeout_s=10.0,
            )
            try:
                server.start()
                shard_ids = runtime.get_window_shard_ids(0)
                self.assertTrue(shard_ids)
                self.assertTrue(manifest_path.exists())
                self.assertTrue(server.shard_store.list_local_shards("train"))
            finally:
                runtime.close()
                server.stop(grace=0.0)

    def test_adaptive_prefetch_reduces_wait_for_next_window(self) -> None:
        cold_elapsed, cold_stats = self._measure_window_wait(prefetch_shards=0)
        warm_elapsed, warm_stats = self._measure_window_wait(prefetch_shards=3)

        self.assertGreaterEqual(cold_elapsed, 0.0)
        self.assertGreaterEqual(warm_elapsed, 0.0)
        self.assertEqual(cold_stats["prefetch_hits"], 0)
        self.assertEqual(warm_stats["prefetch_hits"], 1)
        self.assertGreater(warm_stats["prefetch_wait_s"], 0.0)
        self.assertGreaterEqual(warm_stats["shards_pulled"], 2)
        self.assertIn("cache_hit_rate", warm_stats)

    def test_adaptive_runtime_retries_transient_pull_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = self._write_runtime_configs(
                tmp_path,
                prefetch_shards=0,
            )

            bootstrap_training = load_training_config(training_paths[node_ids[0]])
            build_dataset_shards(bootstrap_training)

            servers: dict[str, PeerServer] = {}
            runtimes: dict[str, AdaptiveMicroShardRuntime] = {}
            try:
                for node_id in node_ids:
                    resolved = load_resolved_config(cluster_path, training_paths[node_id], node_id)
                    manifest_path = resolved.training.dataset.manifest_path
                    server = PeerServer(
                        node_id=node_id,
                        host="127.0.0.1",
                        port=resolved.self_node.port,
                        shard_manifest_path=(
                            manifest_path
                            if manifest_path is not None and Path(manifest_path).exists()
                            else None
                        ),
                        transfer_chunk_bytes=256,
                    )
                    server.start()
                    servers[node_id] = server
                    runtimes[node_id] = AdaptiveMicroShardRuntime(
                        resolved=resolved,
                        server=server,
                        timeout_s=10.0,
                    )

                original_pull_shard = PeerClient.pull_shard
                failure_count = {"remaining": 1}

                def flaky_pull_shard(client_self, *args, **kwargs):
                    if failure_count["remaining"] > 0:
                        failure_count["remaining"] -= 1
                        raise RuntimeError("transient shard pull failure")
                    return original_pull_shard(client_self, *args, **kwargs)

                with mock.patch.object(PeerClient, "pull_shard", new=flaky_pull_shard):
                    bootstrap_shards = runtimes["node-1"].get_window_shard_ids(0)
                    self.assertTrue(bootstrap_shards)
                    shard_ids = runtimes["node-2"].get_window_shard_ids(0)
                    self.assertTrue(shard_ids)
                    stats = runtimes["node-2"].stats()

                self.assertGreaterEqual(stats["transfer_retry_count"], 1)
                self.assertGreater(stats["shards_pulled"], 0)
            finally:
                for runtime in runtimes.values():
                    runtime.close()
                for server in servers.values():
                    server.stop(grace=0.0)

    def _measure_window_wait(self, *, prefetch_shards: int) -> tuple[float, dict[str, float | int]]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = self._write_runtime_configs(
                tmp_path,
                prefetch_shards=prefetch_shards,
            )

            bootstrap_training = load_training_config(training_paths[node_ids[0]])
            build_dataset_shards(bootstrap_training)

            servers: dict[str, PeerServer] = {}
            runtimes: dict[str, AdaptiveMicroShardRuntime] = {}
            try:
                for node_id in node_ids:
                    resolved = load_resolved_config(cluster_path, training_paths[node_id], node_id)
                    manifest_path = resolved.training.dataset.manifest_path
                    server = PeerServer(
                        node_id=node_id,
                        host="127.0.0.1",
                        port=resolved.self_node.port,
                        shard_manifest_path=(
                            manifest_path
                            if manifest_path is not None and Path(manifest_path).exists()
                            else None
                        ),
                        transfer_chunk_bytes=256,
                    )
                    server.start()
                    servers[node_id] = server
                    runtimes[node_id] = AdaptiveMicroShardRuntime(
                        resolved=resolved,
                        server=server,
                        timeout_s=10.0,
                    )

                original_pull_shard = PeerClient.pull_shard

                def delayed_pull_shard(client_self, *args, **kwargs):
                    time.sleep(0.05)
                    return original_pull_shard(client_self, *args, **kwargs)

                with mock.patch.object(PeerClient, "pull_shard", new=delayed_pull_shard):
                    for node_id in node_ids:
                        shard_ids = runtimes[node_id].get_window_shard_ids(0)
                        self.assertTrue(shard_ids)

                    if prefetch_shards > 0:
                        for node_id in node_ids:
                            runtimes[node_id].schedule_prefetch(1)

                    for node_id in node_ids:
                        runtimes[node_id].report_window(
                            window_id=0,
                            samples_processed=4,
                            duration_s=1.0,
                        )

                    if prefetch_shards > 0:
                        time.sleep(0.35)
                    else:
                        bootstrap_shards = runtimes["node-1"].get_window_shard_ids(1)
                        self.assertTrue(bootstrap_shards)

                    started = time.perf_counter()
                    shard_ids = runtimes["node-2"].get_window_shard_ids(1)
                    elapsed = time.perf_counter() - started
                    self.assertTrue(shard_ids)
                    stats = runtimes["node-2"].stats()
                return elapsed, stats
            finally:
                for runtime in runtimes.values():
                    runtime.close()
                for server in servers.values():
                    server.stop(grace=0.0)

    def _write_runtime_configs(
        self,
        tmp_path: Path,
        *,
        prefetch_shards: int,
    ) -> tuple[Path, dict[str, Path], list[str]]:
        node_ids = ["node-1", "node-2", "node-3"]
        ports = [_find_free_port() for _ in node_ids]

        cluster_payload = {
            "cluster_name": "runtime-prefetch",
            "transport": "grpc",
            "overlay_network": "none",
            "tls_enabled": False,
            "bootstrap_node_id": node_ids[0],
            "nodes": [],
        }
        for index, node_id in enumerate(node_ids):
            cluster_payload["nodes"].append(
                {
                    "id": node_id,
                    "host": "127.0.0.1",
                    "port": ports[index],
                    "platform": "linux",
                    "neighbors": [item for item in node_ids if item != node_id],
                    "weight": 1.0,
                    "resources": {
                        "cpu_cores": 2,
                        "accelerator": "cpu",
                        "relative_speed": 1.0,
                    },
                }
            )

        cluster_path = tmp_path / "cluster.runtime-prefetch.yaml"
        with cluster_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cluster_payload, handle, sort_keys=False)

        training_payload = load_yaml(PROJECT_ROOT / "configs" / "training.local-smoke.yaml")
        training_payload["mode"] = "async"
        training_payload["async"]["enabled"] = True
        training_payload["sync"]["enabled"] = False
        training_payload["device_preference"] = ["cpu"]
        training_payload["dataset"]["storage_mode"] = "micro_shards"
        training_payload["dataset"]["scheduler_mode"] = "adaptive"
        training_payload["dataset"]["partitioning"] = "homogeneous"
        training_payload["dataset"]["shard_samples"] = 4
        training_payload["dataset"]["prefetch_shards"] = prefetch_shards
        training_payload["dataset"]["rebalance_window_batches"] = 1
        training_payload["dataset"]["throughput_ema"] = 0.0
        training_payload["dataset"]["warmup_windows"] = 0
        training_payload["dataset"]["min_local_shards"] = 1
        training_payload["dataset"]["max_cache_bytes"] = 1_000_000
        training_payload["dataset"]["fake_train_size"] = 24
        training_payload["dataset"]["fake_val_size"] = 8
        training_payload["dataset"]["fake_test_size"] = 8
        training_payload["optimization"]["batch_size"] = 16
        training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
        training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")

        training_paths: dict[str, Path] = {}
        for node_id in node_ids:
            node_dir = tmp_path / node_id
            node_dir.mkdir(parents=True, exist_ok=True)
            node_payload = dict(training_payload)
            node_payload["dataset"] = dict(training_payload["dataset"])
            node_payload["logging"] = dict(training_payload["logging"])
            node_payload["dataset"]["manifest_path"] = str(node_dir / "manifest.json")
            node_payload["dataset"]["cache_dir"] = str(
                node_dir if node_id == node_ids[0] else node_dir / "cache"
            )
            training_path = node_dir / "training.runtime-prefetch.yaml"
            with training_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(node_payload, handle, sort_keys=False)
            training_paths[node_id] = training_path

        return cluster_path, training_paths, node_ids


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()

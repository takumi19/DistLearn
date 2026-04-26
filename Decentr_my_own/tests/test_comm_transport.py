from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.comm.client import PeerClient
from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload
from decentr_my_own.comm.server import PeerServer
from decentr_my_own.config.loader import load_training_config, load_yaml
from decentr_my_own.data.manifest import load_manifest
from decentr_my_own.data.scheduler_state import LeasePlanRecord, ThroughputReportRecord
from decentr_my_own.data.shards import build_dataset_shards


class CommTransportTests(unittest.TestCase):
    def test_ping_push_and_state(self) -> None:
        server = PeerServer(node_id="receiver-node", host="127.0.0.1", port=0)
        server.start()
        self.addCleanup(server.stop)

        client = PeerClient(server.address)
        self.addCleanup(client.close)

        ping = client.ping(sender_node_id="sender-node")
        self.assertEqual(ping["receiver_node_id"], "receiver-node")

        payload = PeerPayload(
            metadata=PayloadMetadata(
                sender_node_id="sender-node",
                payload_id="payload-transport",
                payload_kind="model_delta",
                model_version=2,
                step=7,
                sample_count=64,
            ),
            tensors={
                "weights": torch.ones((2, 2), dtype=torch.float32),
                "bias": torch.zeros((2,), dtype=torch.float32),
            },
        )
        push = client.push_payload(payload)
        self.assertEqual(push.receiver_node_id, "receiver-node")
        self.assertEqual(push.sender_node_id, "sender-node")
        self.assertEqual(push.tensor_count, 2)

        state = client.get_peer_state()
        self.assertEqual(state["received_payload_count"], 1)
        self.assertEqual(state["payloads"][0]["payload_id"], "payload-transport")
        self.assertEqual(state["payloads"][0]["payload_kind"], "model_delta")
        self.assertEqual(state["payloads"][0]["tensor_names"], ["weights", "bias"])

    def test_peer_state_counts_total_payloads_while_bounding_history(self) -> None:
        server = PeerServer(node_id="receiver-node", host="127.0.0.1", port=0)
        server.start()
        self.addCleanup(server.stop)

        client = PeerClient(server.address)
        self.addCleanup(client.close)

        for step in range(6):
            payload = PeerPayload(
                metadata=PayloadMetadata(
                    sender_node_id="sender-node",
                    payload_id=f"payload-{step}",
                    payload_kind="async_weights",
                    model_version=step,
                    step=step,
                    sample_count=8,
                ),
                tensors={
                    "weights": torch.full((2, 2), float(step), dtype=torch.float32),
                },
            )
            client.push_payload(payload)

        state = client.get_peer_state()
        self.assertEqual(state["received_payload_count"], 6)
        self.assertEqual(len(state["payloads"]), 1)
        self.assertEqual(state["payloads"][0]["payload_id"], "payload-5")
        self.assertIsNone(server.get_payload("sender-node", payload_id="payload-0"))
        self.assertIsNotNone(server.get_payload("sender-node", payload_id="payload-5"))

    def test_manifest_inventory_pull_and_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_payload = load_yaml(
                PROJECT_ROOT / "configs" / "training.local-smoke.yaml"
            )
            training_payload["dataset"]["storage_mode"] = "micro_shards"
            training_payload["dataset"]["manifest_path"] = str(tmp_path / "manifest.json")
            training_payload["dataset"]["shard_samples"] = 4

            training_path = tmp_path / "training.micro.yaml"
            with training_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(training_payload, handle, sort_keys=False)

            training = load_training_config(training_path)
            build_dataset_shards(training)
            manifest = load_manifest(training.dataset.manifest_path)
            first_shard = manifest.shards[0]

            server = PeerServer(
                node_id="receiver-node",
                host="127.0.0.1",
                port=0,
                shard_manifest_path=training.dataset.manifest_path,
                transfer_chunk_bytes=64,
            )
            server.set_lease_plan(
                LeasePlanRecord(
                    window_id=3,
                    assignments={
                        "node-a": (first_shard.shard_id,),
                    },
                )
            )
            server.start()
            self.addCleanup(server.stop)

            client = PeerClient(server.address)
            self.addCleanup(client.close)

            remote_manifest = client.get_manifest()
            self.assertEqual(remote_manifest.model_dump(), manifest.model_dump())

            remote_shards = client.list_local_shards()
            self.assertEqual(
                sorted((item.model_dump() for item in remote_shards), key=lambda item: item["shard_id"]),
                sorted((item.model_dump() for item in manifest.shards), key=lambda item: item["shard_id"]),
            )

            pulled_path = tmp_path / "pulled" / f"{first_shard.shard_id}.pt"
            pull_result = client.pull_shard(
                shard_id=first_shard.shard_id,
                destination_path=pulled_path,
                expected_meta=first_shard,
            )
            self.assertEqual(pull_result.shard_id, first_shard.shard_id)
            self.assertTrue(pull_result.path.exists())

            report_reply = client.report_throughput(
                ThroughputReportRecord(
                    node_id="worker-1",
                    window_id=7,
                    samples_processed=32,
                    window_seconds=1.5,
                    effective_throughput=21.33,
                    local_inventory=(first_shard.shard_id,),
                )
            )
            self.assertEqual(report_reply["receiver_node_id"], "receiver-node")
            self.assertEqual(report_reply["node_id"], "worker-1")
            self.assertEqual(report_reply["window_id"], 7)

            control_snapshot = server.control_snapshot()
            self.assertEqual(len(control_snapshot.throughput_reports), 1)
            self.assertEqual(control_snapshot.throughput_reports[0].node_id, "worker-1")

            lease_plan = client.get_lease_plan(window_id=3)
            self.assertEqual(lease_plan.window_id, 3)
            self.assertEqual(lease_plan.shards_for_node("node-a"), (first_shard.shard_id,))


if __name__ == "__main__":
    unittest.main()

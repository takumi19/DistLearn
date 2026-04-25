from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload
from decentr_my_own.comm.serialization import messages_to_payload, payload_to_messages


class CommSerializationTests(unittest.TestCase):
    def test_payload_round_trip(self) -> None:
        payload = PeerPayload(
            metadata=PayloadMetadata(
                sender_node_id="node-1",
                payload_id="payload-1",
                payload_kind="model_state",
                model_version=3,
                step=12,
                sample_count=128,
            ),
            tensors={
                "weights": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "bias": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            },
        )

        encoded = list(payload_to_messages(payload, chunk_size=8))
        decoded, num_bytes, digest = messages_to_payload(encoded)

        self.assertEqual(decoded.metadata.sender_node_id, "node-1")
        self.assertEqual(decoded.metadata.payload_id, "payload-1")
        self.assertEqual(num_bytes, 60)
        self.assertTrue(digest)
        self.assertTrue(torch.equal(decoded.tensors["weights"], payload.tensors["weights"]))
        self.assertTrue(torch.equal(decoded.tensors["bias"], payload.tensors["bias"]))


if __name__ == "__main__":
    unittest.main()

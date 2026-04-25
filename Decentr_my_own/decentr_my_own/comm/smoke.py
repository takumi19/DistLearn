from __future__ import annotations

import multiprocessing as mp
import socket
import time

import torch

from decentr_my_own.comm.client import PeerClient
from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload
from decentr_my_own.comm.server import PeerServer


def run_peer_smoke(peer_count: int = 3, timeout_s: float = 10.0) -> dict:
    if peer_count < 2:
        raise ValueError("peer_count must be at least 2")

    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_events = []
    processes = []
    ports = [_find_free_port() for _ in range(peer_count)]
    node_ids = [f"node-{idx + 1}" for idx in range(peer_count)]

    try:
        for node_id, port in zip(node_ids, ports):
            stop_event = ctx.Event()
            process = ctx.Process(
                target=_serve_process,
                args=(node_id, port, ready_queue, stop_event),
            )
            process.start()
            stop_events.append(stop_event)
            processes.append(process)

        ready = {}
        deadline = time.time() + timeout_s
        while len(ready) < peer_count and time.time() < deadline:
            node_id, address = ready_queue.get(timeout=max(0.1, deadline - time.time()))
            ready[node_id] = address

        if len(ready) != peer_count:
            raise RuntimeError("Not all peer servers became ready in time")

        ping_results = {}
        push_results = []
        state_results = {}
        for sender_idx, sender_node_id in enumerate(node_ids):
            for receiver_node_id in node_ids:
                if sender_node_id == receiver_node_id:
                    continue
                client = PeerClient(ready[receiver_node_id])
                try:
                    ping_results[f"{sender_node_id}->{receiver_node_id}"] = client.ping(
                        sender_node_id=sender_node_id,
                        timeout_s=timeout_s,
                    )
                    push_results.append(
                        client.push_payload(
                            _build_smoke_payload(sender_idx, sender_node_id, receiver_node_id),
                            timeout_s=timeout_s,
                        ).to_dict()
                    )
                finally:
                    client.close()

        for node_id in node_ids:
            client = PeerClient(ready[node_id])
            try:
                state_results[node_id] = client.get_peer_state(timeout_s=timeout_s)
            finally:
                client.close()

        return {
            "peer_count": peer_count,
            "addresses": ready,
            "ping_results": ping_results,
            "push_results": push_results,
            "state_results": state_results,
        }
    finally:
        for stop_event in stop_events:
            stop_event.set()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def _serve_process(
    node_id: str, port: int, ready_queue: mp.Queue, stop_event: mp.Event
) -> None:
    server = PeerServer(node_id=node_id, host="127.0.0.1", port=port)
    server.start()
    ready_queue.put((node_id, server.address))
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        server.stop(grace=0.0)


def _build_smoke_payload(sender_idx: int, sender_node_id: str, receiver_node_id: str) -> PeerPayload:
    tensor = torch.full((2, 3), fill_value=float(sender_idx + 1), dtype=torch.float32)
    bias = torch.arange(3, dtype=torch.float32) + sender_idx
    return PeerPayload(
        metadata=PayloadMetadata(
            sender_node_id=sender_node_id,
            payload_id=f"{sender_node_id}-to-{receiver_node_id}",
            payload_kind="model_state",
            model_version=1,
            step=sender_idx + 1,
            sample_count=16 * (sender_idx + 1),
        ),
        tensors={"weights": tensor, "bias": bias},
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

from __future__ import annotations

from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from decentr_my_own.comm.messages import PeerPayload, StoredPayload, StoredPayloadSummary
from decentr_my_own.data.scheduler_state import (
    LeasePlanRecord,
    RunCompletionRecord,
    ThroughputReportRecord,
)


@dataclass
class PeerStateSnapshot:
    node_id: str
    payloads: list[StoredPayloadSummary]
    received_payload_count: int

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "received_payload_count": self.received_payload_count,
            "latest_sender_count": len(self.payloads),
            "payloads": [payload.to_dict() for payload in self.payloads],
        }


@dataclass
class ControlPlaneSnapshot:
    node_id: str
    throughput_reports: list[ThroughputReportRecord]
    lease_plan: LeasePlanRecord
    run_completions: list[RunCompletionRecord]

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "throughput_reports": [item.to_dict() for item in self.throughput_reports],
            "lease_plan": self.lease_plan.to_dict(),
            "run_completions": [item.to_dict() for item in self.run_completions],
        }


class PeerStateStore:
    def __init__(
        self,
        receiver_node_id: str,
        *,
        max_payload_history_per_sender: int = 1,
    ):
        self.receiver_node_id = receiver_node_id
        self.max_payload_history_per_sender = max(max_payload_history_per_sender, 1)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest_payloads: dict[str, StoredPayload] = {}
        self._payload_history: dict[tuple[str, str], StoredPayload] = {}
        self._payload_order: dict[str, deque[str]] = defaultdict(deque)
        self._received_payload_count = 0

    def store(self, payload: PeerPayload, num_bytes: int, digest: str) -> StoredPayloadSummary:
        summary = StoredPayloadSummary(
            receiver_node_id=self.receiver_node_id,
            sender_node_id=payload.metadata.sender_node_id,
            payload_id=payload.metadata.payload_id,
            payload_kind=payload.metadata.payload_kind,
            model_version=payload.metadata.model_version,
            step=payload.metadata.step,
            sample_count=payload.metadata.sample_count,
            tensor_count=len(payload.tensors),
            num_bytes=num_bytes,
            digest=digest,
            received_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            tensor_names=tuple(payload.tensors.keys()),
        )
        with self._condition:
            stored_payload = StoredPayload(
                summary=summary,
                payload=payload,
            )
            self._latest_payloads[payload.metadata.sender_node_id] = stored_payload
            self._payload_history[
                (payload.metadata.sender_node_id, payload.metadata.payload_id)
            ] = stored_payload
            self._payload_order[payload.metadata.sender_node_id].append(payload.metadata.payload_id)
            self._received_payload_count += 1
            self._trim_history_for_sender(payload.metadata.sender_node_id)
            self._condition.notify_all()
        return summary

    def snapshot(self) -> PeerStateSnapshot:
        with self._lock:
            payloads = [item.summary for item in self._latest_payloads.values()]
            received_payload_count = self._received_payload_count
        payloads.sort(key=lambda item: item.sender_node_id)
        return PeerStateSnapshot(
            node_id=self.receiver_node_id,
            payloads=payloads,
            received_payload_count=received_payload_count,
        )

    def get_payload(self, sender_node_id: str, payload_id: str | None = None) -> PeerPayload | None:
        with self._lock:
            if payload_id is None:
                stored = self._latest_payloads.get(sender_node_id)
            else:
                stored = self._payload_history.get((sender_node_id, payload_id))
            return None if stored is None else stored.payload

    def wait_for_payloads(self, min_count: int, timeout_s: float) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self._latest_payloads) >= min_count,
                timeout=timeout_s,
            )

    def wait_for_payload_ids(
        self, sender_node_ids: list[str], payload_id: str, timeout_s: float
    ) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: all(
                    (sender_node_id, payload_id) in self._payload_history
                    for sender_node_id in sender_node_ids
                ),
                timeout=timeout_s,
            )

    def _trim_history_for_sender(self, sender_node_id: str) -> None:
        payload_ids = self._payload_order[sender_node_id]
        while len(payload_ids) > self.max_payload_history_per_sender:
            payload_id = payload_ids.popleft()
            self._payload_history.pop((sender_node_id, payload_id), None)


class ControlPlaneStateStore:
    def __init__(self, receiver_node_id: str):
        self.receiver_node_id = receiver_node_id
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._throughput_reports: dict[str, ThroughputReportRecord] = {}
        self._throughput_history: dict[tuple[str, int], ThroughputReportRecord] = {}
        self._lease_plan = LeasePlanRecord()
        self._lease_plan_history: dict[int, LeasePlanRecord] = {}
        self._run_completions: dict[str, RunCompletionRecord] = {}

    def store_throughput_report(
        self, report: ThroughputReportRecord
    ) -> ThroughputReportRecord:
        with self._condition:
            self._throughput_reports[report.node_id] = report
            self._throughput_history[(report.node_id, report.window_id)] = report
            self._condition.notify_all()
        return report

    def set_lease_plan(self, lease_plan: LeasePlanRecord) -> None:
        with self._condition:
            self._lease_plan = lease_plan
            self._lease_plan_history[lease_plan.window_id] = lease_plan
            self._condition.notify_all()

    def store_run_completion(
        self, completion: RunCompletionRecord
    ) -> RunCompletionRecord:
        with self._condition:
            self._run_completions[completion.node_id] = completion
            self._condition.notify_all()
        return completion

    def get_lease_plan(self, window_id: int | None = None) -> LeasePlanRecord:
        with self._lock:
            if window_id is None:
                return self._lease_plan
            if window_id in self._lease_plan_history:
                return self._lease_plan_history[window_id]
            return LeasePlanRecord(window_id=window_id, assignments={})

    def get_throughput_reports(
        self,
        *,
        window_id: int | None = None,
    ) -> list[ThroughputReportRecord]:
        with self._lock:
            if window_id is None:
                reports = list(self._throughput_reports.values())
            else:
                reports = [
                    report
                    for (_, current_window_id), report in self._throughput_history.items()
                    if current_window_id == window_id
                ]
        reports.sort(key=lambda item: item.node_id)
        return reports

    def wait_for_throughput_reports(
        self,
        *,
        node_ids: list[str],
        window_id: int,
        timeout_s: float,
    ) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: all(
                    (node_id, window_id) in self._throughput_history for node_id in node_ids
                ),
                timeout=timeout_s,
            )

    def get_run_completions(self) -> list[RunCompletionRecord]:
        with self._lock:
            completions = list(self._run_completions.values())
        completions.sort(key=lambda item: item.node_id)
        return completions

    def wait_for_run_completions(
        self,
        *,
        node_ids: list[str],
        timeout_s: float,
    ) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: all(node_id in self._run_completions for node_id in node_ids),
                timeout=timeout_s,
            )

    def snapshot(self) -> ControlPlaneSnapshot:
        with self._lock:
            reports = list(self._throughput_reports.values())
            lease_plan = self._lease_plan
            run_completions = list(self._run_completions.values())
        reports.sort(key=lambda item: item.node_id)
        run_completions.sort(key=lambda item: item.node_id)
        return ControlPlaneSnapshot(
            node_id=self.receiver_node_id,
            throughput_reports=reports,
            lease_plan=lease_plan,
            run_completions=run_completions,
        )

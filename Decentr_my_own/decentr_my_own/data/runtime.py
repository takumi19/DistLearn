from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from decentr_my_own.comm.client import PeerClient
from decentr_my_own.comm.server import PeerServer
from decentr_my_own.config.models import ResolvedConfig
from decentr_my_own.data.lease_planner import AdaptiveLeasePlanner, build_static_lease_plan
from decentr_my_own.data.manifest import load_manifest, save_manifest
from decentr_my_own.data.scheduler_state import LeasePlanRecord, ThroughputReportRecord

_INITIAL_TRANSFER_BACKOFF_S = 0.1
_MAX_TRANSFER_BACKOFF_S = 1.0
_MAX_IN_FLIGHT_TRANSFERS = 1


@dataclass(frozen=True)
class AdaptiveWindowAssignment:
    window_id: int
    epoch_id: int
    epoch_window_index: int
    epoch_window_count: int
    shard_ids: list[str]

    @property
    def is_last_window_for_epoch(self) -> bool:
        return self.epoch_window_index + 1 >= self.epoch_window_count


def prepare_static_micro_shards(
    resolved: ResolvedConfig,
    server: PeerServer,
    *,
    timeout_s: float,
    window_id: int = 0,
) -> list[str]:
    config = resolved.training
    if config.dataset.storage_mode != "micro_shards":
        return []
    if config.dataset.manifest_path is None:
        raise ValueError("dataset.manifest_path is required for storage_mode=micro_shards")

    manifest_path = Path(config.dataset.manifest_path)
    cache_dir = _cache_dir(config.dataset.manifest_path, config.dataset.cache_dir)
    bootstrap_node_id = resolved.cluster.bootstrap_node_id or resolved.self_node_id

    if resolved.self_node_id == bootstrap_node_id:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Bootstrap node '{resolved.self_node_id}' is missing manifest at {manifest_path}"
            )
        server.configure_shard_store(manifest_path, base_dir=cache_dir)
        if not server.shard_store.list_local_shards("train"):
            raise RuntimeError(
                f"Bootstrap node '{resolved.self_node_id}' has no local train shards under {cache_dir}"
            )
        lease_plan = build_static_lease_plan(
            server.shard_store.manifest,
            resolved.cluster,
            resolved.training,
            split="train",
            window_id=window_id,
        )
        server.set_lease_plan(lease_plan)
    else:
        bootstrap = resolved.cluster.get_node(bootstrap_node_id)
        client = PeerClient(f"{bootstrap.host}:{bootstrap.port}")
        try:
            if not manifest_path.exists():
                manifest = _fetch_manifest_with_retry(client, timeout_s=timeout_s)
                save_manifest(manifest, manifest_path)
            server.configure_shard_store(manifest_path, base_dir=cache_dir)
            lease_plan = _fetch_lease_plan_with_retry(
                client,
                window_id=window_id,
                timeout_s=timeout_s,
            )
            _pull_missing_shards(
                client,
                server,
                lease_plan,
                node_id=resolved.self_node_id,
                timeout_s=timeout_s,
            )
        finally:
            client.close()

    if server.shard_store is None:
        server.configure_shard_store(manifest_path, base_dir=cache_dir)
    lease_plan = server.control_store.get_lease_plan(window_id)
    if not lease_plan.assignments:
        bootstrap = resolved.cluster.get_node(bootstrap_node_id)
        client = PeerClient(f"{bootstrap.host}:{bootstrap.port}")
        try:
            lease_plan = _fetch_lease_plan_with_retry(client, window_id=window_id, timeout_s=timeout_s)
        finally:
            client.close()
    return list(lease_plan.shards_for_node(resolved.self_node_id))


def _cache_dir(manifest_path: str, cache_dir: str | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(manifest_path).parent


def _fetch_manifest_with_retry(client: PeerClient, *, timeout_s: float):
    deadline = time.time() + timeout_s
    while True:
        try:
            return client.get_manifest(timeout_s=min(2.0, max(0.5, deadline - time.time())))
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(0.1)


def _fetch_lease_plan_with_retry(
    client: PeerClient,
    *,
    window_id: int,
    timeout_s: float,
) -> LeasePlanRecord:
    deadline = time.time() + timeout_s
    while True:
        try:
            lease_plan = client.get_lease_plan(
                window_id=window_id,
                timeout_s=min(2.0, max(0.5, deadline - time.time())),
            )
            if lease_plan.window_id == window_id and lease_plan.assignments:
                return lease_plan
        except Exception:
            pass
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for lease plan window_id={window_id}")
        time.sleep(0.1)


def _pull_shard_with_retry(
    client: PeerClient,
    *,
    shard_meta,
    destination_path,
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    backoff_s = _INITIAL_TRANSFER_BACKOFF_S
    last_error: Exception | None = None
    while True:
        try:
            client.pull_shard(
                shard_id=shard_meta.shard_id,
                destination_path=destination_path,
                expected_meta=shard_meta,
                timeout_s=min(2.0, max(0.5, deadline - time.time())),
            )
            return
        except Exception as exc:
            last_error = exc
            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            delay_s = min(backoff_s, remaining_s)
            if delay_s > 0:
                time.sleep(delay_s)
            backoff_s = min(backoff_s * 2.0, _MAX_TRANSFER_BACKOFF_S)
    if last_error is not None:
        raise last_error


def _pull_missing_shards(
    client: PeerClient,
    server: PeerServer,
    lease_plan: LeasePlanRecord,
    *,
    node_id: str,
    timeout_s: float,
) -> None:
    if server.shard_store is None:
        raise RuntimeError("Shard store must be configured before pulling shards")
    for shard_id in lease_plan.shards_for_node(node_id):
        if server.shard_store.has_local_shard(shard_id):
            continue
        shard_meta = server.shard_store.get_shard_meta(shard_id)
        _pull_shard_with_retry(
            client,
            shard_meta=shard_meta,
            destination_path=server.shard_store.resolve_path(shard_meta),
            timeout_s=timeout_s,
        )
    for split_name in ("val", "test"):
        for shard_meta in server.shard_store.manifest.shards_for_split(split_name):
            if server.shard_store.has_local_shard(shard_meta):
                continue
            _pull_shard_with_retry(
                client,
                shard_meta=shard_meta,
                destination_path=server.shard_store.resolve_path(shard_meta),
                timeout_s=timeout_s,
            )
    server.set_lease_plan(lease_plan)


@dataclass
class AdaptiveMicroShardRuntime:
    resolved: ResolvedConfig
    server: PeerServer
    timeout_s: float
    _initialized: bool = False
    _planner: AdaptiveLeasePlanner | None = None
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _prefetch_task: _PrefetchTask | None = field(default=None, init=False, repr=False)
    _inventory_cache: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _active_train_shard_ids: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _transfer_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(_MAX_IN_FLIGHT_TRANSFERS),
        init=False,
        repr=False,
    )
    _scheduler_history_rows: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _stats: dict[str, float | int] = field(
        default_factory=lambda: {
            "bytes_transferred": 0,
            "shards_pulled": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_failures": 0,
            "prefetch_wait_s": 0.0,
            "window_wait_s": 0.0,
            "evicted_shards": 0,
            "evicted_bytes": 0,
            "train_shard_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "transfer_retry_count": 0,
            "transfer_retry_backoff_s": 0.0,
        },
        init=False,
        repr=False,
    )

    def get_window_assignment(self, window_id: int) -> AdaptiveWindowAssignment:
        self._initialize()
        started_wait = time.perf_counter()
        prefetched = self._wait_for_prefetch(window_id)
        lease_plan = self._materialize_window(window_id, prefetch_limit=None)
        shard_ids = list(lease_plan.shards_for_node(self.resolved.self_node_id))
        elapsed_s = time.perf_counter() - started_wait

        with self._state_lock:
            self._active_train_shard_ids = tuple(shard_ids)
            if prefetched:
                self._stats["prefetch_hits"] += 1
                self._stats["prefetch_wait_s"] += elapsed_s
            else:
                self._stats["prefetch_misses"] += 1
                self._stats["window_wait_s"] += elapsed_s

        self._trim_cache(extra_protected_shard_ids=shard_ids)
        return AdaptiveWindowAssignment(
            window_id=lease_plan.window_id,
            epoch_id=lease_plan.epoch_id,
            epoch_window_index=lease_plan.epoch_window_index,
            epoch_window_count=lease_plan.epoch_window_count,
            shard_ids=shard_ids,
        )

    def get_window_shard_ids(self, window_id: int) -> list[str]:
        return self.get_window_assignment(window_id).shard_ids

    def schedule_prefetch(self, window_id: int) -> None:
        self._initialize()
        if self.resolved.training.dataset.prefetch_shards <= 0:
            return

        with self._state_lock:
            task = self._prefetch_task
            if task is not None:
                if task.window_id == window_id:
                    return
                if task.thread is not None and task.thread.is_alive():
                    return
            task = _PrefetchTask(window_id=window_id)
            thread = threading.Thread(
                target=self._run_prefetch_task,
                args=(task,),
                name=f"prefetch-{self.resolved.self_node_id}-{window_id}",
                daemon=True,
            )
            task.thread = thread
            self._prefetch_task = task

        thread.start()

    def report_window(
        self,
        *,
        window_id: int,
        samples_processed: int,
        duration_s: float,
    ) -> None:
        self._initialize()
        inventory = ()
        if self.server.shard_store is not None:
            inventory = tuple(
                shard.shard_id for shard in self.server.shard_store.list_local_shards("train")
            )
        report = ThroughputReportRecord(
            node_id=self.resolved.self_node_id,
            window_id=window_id,
            samples_processed=samples_processed,
            window_seconds=duration_s,
            effective_throughput=samples_processed / max(duration_s, 1e-12),
            local_inventory=inventory,
        )
        if self._is_bootstrap:
            self.server.control_store.store_throughput_report(report)
            if self.resolved.training.dataset.prefetch_shards > 0:
                next_window_id = window_id + 1
                next_plan = self.server.control_store.get_lease_plan(next_window_id)
                if not next_plan.assignments:
                    self._ensure_leader_plan(next_window_id)
        else:
            client = self._bootstrap_client()
            try:
                client.report_throughput(report, timeout_s=self.timeout_s)
            finally:
                client.close()
        self._record_scheduler_window(window_id=window_id, report=report)

    def stats(self) -> dict[str, float | int]:
        with self._state_lock:
            stats = dict(self._stats)
        request_count = int(stats.get("train_shard_requests", 0))
        cache_hits = int(stats.get("cache_hits", 0))
        stats["cache_hit_rate"] = (
            cache_hits / float(request_count) if request_count > 0 else 0.0
        )
        return stats

    def scheduler_history(self) -> list[dict[str, object]]:
        with self._state_lock:
            return [dict(row) for row in self._scheduler_history_rows]

    def close(self) -> None:
        with self._state_lock:
            task = self._prefetch_task
        if task is not None and task.thread is not None and task.thread.is_alive():
            task.thread.join(timeout=self.timeout_s)

    def _initialize(self) -> None:
        if self._initialized:
            return
        config = self.resolved.training
        if config.dataset.storage_mode != "micro_shards":
            self._initialized = True
            return
        if config.dataset.manifest_path is None:
            raise ValueError("dataset.manifest_path is required for storage_mode=micro_shards")

        manifest_path = Path(config.dataset.manifest_path)
        cache_dir = _cache_dir(config.dataset.manifest_path, config.dataset.cache_dir)

        if self._is_bootstrap:
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Bootstrap node '{self.resolved.self_node_id}' is missing manifest at {manifest_path}"
                )
            self.server.configure_shard_store(manifest_path, base_dir=cache_dir)
            if not self.server.shard_store.list_local_shards("train"):
                raise RuntimeError(
                    f"Bootstrap node '{self.resolved.self_node_id}' has no local train shards under {cache_dir}"
                )
            self._planner = AdaptiveLeasePlanner(
                self.server.shard_store.manifest,
                self.resolved.cluster,
                self.resolved.training,
                split="train",
            )
            initial_plan = self._planner.plan_window(window_id=0)
            self.server.set_lease_plan(initial_plan)
        else:
            client = self._bootstrap_client()
            try:
                if not manifest_path.exists():
                    manifest = _fetch_manifest_with_retry(client, timeout_s=self.timeout_s)
                    save_manifest(manifest, manifest_path)
                self.server.configure_shard_store(manifest_path, base_dir=cache_dir)
                self._pull_eval_shards(client)
            finally:
                client.close()

        self._initialized = True

    @property
    def _is_bootstrap(self) -> bool:
        return (
            self.resolved.cluster.bootstrap_node_id or self.resolved.self_node_id
        ) == self.resolved.self_node_id

    def _bootstrap_client(self) -> PeerClient:
        bootstrap_node_id = self.resolved.cluster.bootstrap_node_id or self.resolved.self_node_id
        bootstrap = self.resolved.cluster.get_node(bootstrap_node_id)
        return PeerClient(f"{bootstrap.host}:{bootstrap.port}")

    def _fetch_remote_lease_plan(self, window_id: int) -> LeasePlanRecord:
        client = self._bootstrap_client()
        try:
            return _fetch_lease_plan_with_retry(client, window_id=window_id, timeout_s=self.timeout_s)
        finally:
            client.close()

    def _wait_for_prefetch(self, window_id: int) -> bool:
        with self._state_lock:
            task = self._prefetch_task if self._prefetch_task is not None else None
            if task is None or task.window_id != window_id:
                return False

        if not task.completed.wait(timeout=self.timeout_s):
            raise TimeoutError(f"Timed out waiting for prefetch of window {window_id}")
        if task.error is not None:
            with self._state_lock:
                self._stats["prefetch_failures"] += 1
            return False
        return True

    def _run_prefetch_task(self, task: "_PrefetchTask") -> None:
        try:
            self._materialize_window(
                task.window_id,
                prefetch_limit=self.resolved.training.dataset.prefetch_shards,
            )
        except Exception as exc:
            task.error = exc
        finally:
            task.completed.set()

    def _materialize_window(
        self,
        window_id: int,
        *,
        prefetch_limit: int | None,
    ) -> LeasePlanRecord:
        if self.server.shard_store is None:
            raise RuntimeError("Shard store is not configured")

        if self._is_bootstrap:
            self._ensure_leader_plan(window_id)
            return self.server.control_store.get_lease_plan(window_id)

        lease_plan = self.server.control_store.get_lease_plan(window_id)
        if not lease_plan.assignments:
            lease_plan = self._fetch_remote_lease_plan(window_id)
        self._ensure_train_shards(lease_plan, prefetch_limit=prefetch_limit)
        return lease_plan

    def _pull_eval_shards(self, client: PeerClient) -> None:
        if self.server.shard_store is None:
            raise RuntimeError("Shard store is not configured")
        for split_name in ("val", "test"):
            for shard_meta in self.server.shard_store.manifest.shards_for_split(split_name):
                if self.server.shard_store.has_local_shard(shard_meta):
                    continue
                self._pull_shard_with_retry(client, shard_meta)

    def _ensure_train_shards(
        self,
        lease_plan: LeasePlanRecord,
        *,
        prefetch_limit: int | None,
    ) -> None:
        if self.server.shard_store is None:
            raise RuntimeError("Shard store is not configured")

        assigned_shard_ids = list(lease_plan.shards_for_node(self.resolved.self_node_id))
        with self._state_lock:
            self._stats["train_shard_requests"] += len(assigned_shard_ids)
        missing_shard_ids = [
            shard_id
            for shard_id in assigned_shard_ids
            if not self.server.shard_store.has_local_shard(shard_id)
        ]
        with self._state_lock:
            self._stats["cache_hits"] += len(assigned_shard_ids) - len(missing_shard_ids)
            self._stats["cache_misses"] += len(missing_shard_ids)
        if prefetch_limit is not None:
            missing_shard_ids = missing_shard_ids[:prefetch_limit]

        if missing_shard_ids:
            self._refresh_inventory_cache()
            for shard_id in missing_shard_ids:
                self._pull_shard_from_best_source(shard_id)

        self.server.set_lease_plan(lease_plan)
        self._trim_cache(extra_protected_shard_ids=assigned_shard_ids)

    def _refresh_inventory_cache(self) -> None:
        bootstrap_node_id = self.resolved.cluster.bootstrap_node_id or self.resolved.self_node_id
        inventories: dict[str, set[str]] = {}

        for node in self.resolved.cluster.nodes:
            if node.id in {self.resolved.self_node_id, bootstrap_node_id}:
                continue
            client = PeerClient(f"{node.host}:{node.port}")
            try:
                inventories[node.id] = {
                    shard.shard_id
                    for shard in client.list_local_shards(timeout_s=min(2.0, self.timeout_s))
                }
            except Exception:
                inventories[node.id] = set()
            finally:
                client.close()

        with self._state_lock:
            self._inventory_cache = inventories

    def _pull_shard_from_best_source(self, shard_id: str) -> None:
        if self.server.shard_store is None:
            raise RuntimeError("Shard store is not configured")

        shard_meta = self.server.shard_store.get_shard_meta(shard_id)
        last_error: Exception | None = None
        for source_node_id in self._source_candidates_for_shard(shard_id):
            node = self.resolved.cluster.get_node(source_node_id)
            client = PeerClient(f"{node.host}:{node.port}")
            try:
                self._pull_shard_with_retry(client, shard_meta)
                return
            except Exception as exc:
                last_error = exc
            finally:
                client.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No available source for shard '{shard_id}'")

    def _source_candidates_for_shard(self, shard_id: str) -> list[str]:
        bootstrap_node_id = self.resolved.cluster.bootstrap_node_id or self.resolved.self_node_id
        with self._state_lock:
            inventory_cache = {node_id: set(shards) for node_id, shards in self._inventory_cache.items()}

        candidates = [
            node.id
            for node in self.resolved.cluster.nodes
            if node.id not in {self.resolved.self_node_id, bootstrap_node_id}
            and shard_id in inventory_cache.get(node.id, set())
        ]
        if bootstrap_node_id != self.resolved.self_node_id:
            candidates.append(bootstrap_node_id)
        return candidates

    def _trim_cache(self, *, extra_protected_shard_ids: list[str] | tuple[str, ...]) -> None:
        if self._is_bootstrap or self.server.shard_store is None:
            return

        protected = set(extra_protected_shard_ids)
        with self._state_lock:
            protected.update(self._active_train_shard_ids)
        for split_name in ("val", "test"):
            protected.update(
                shard.shard_id for shard in self.server.shard_store.manifest.shards_for_split(split_name)
            )

        evicted = self.server.shard_store.trim_cache(
            max_cache_bytes=self.resolved.training.dataset.max_cache_bytes,
            protected_shard_ids=protected,
            split="train",
        )
        if not evicted:
            return

        evicted_bytes = sum(
            self.server.shard_store.get_shard_meta(shard_id).byte_size for shard_id in evicted
        )
        with self._state_lock:
            self._stats["evicted_shards"] += len(evicted)
            self._stats["evicted_bytes"] += evicted_bytes

    def _record_transfer(self, byte_size: int) -> None:
        with self._state_lock:
            self._stats["shards_pulled"] += 1
            self._stats["bytes_transferred"] += byte_size

    def _record_retry(self, backoff_s: float) -> None:
        with self._state_lock:
            self._stats["transfer_retry_count"] += 1
            self._stats["transfer_retry_backoff_s"] += backoff_s

    def _pull_shard_with_retry(
        self,
        client: PeerClient,
        shard_meta,
    ) -> None:
        if self.server.shard_store is None:
            raise RuntimeError("Shard store is not configured")

        deadline = time.time() + self.timeout_s
        backoff_s = _INITIAL_TRANSFER_BACKOFF_S
        last_error: Exception | None = None
        while True:
            try:
                with self._transfer_semaphore:
                    result = client.pull_shard(
                        shard_id=shard_meta.shard_id,
                        destination_path=self.server.shard_store.resolve_path(shard_meta),
                        expected_meta=shard_meta,
                        timeout_s=min(2.0, max(0.5, deadline - time.time())),
                    )
                self._record_transfer(result.byte_size)
                return
            except Exception as exc:
                last_error = exc
                remaining_s = deadline - time.time()
                if remaining_s <= 0:
                    break
                delay_s = min(backoff_s, remaining_s)
                self._record_retry(delay_s)
                if delay_s > 0:
                    time.sleep(delay_s)
                backoff_s = min(backoff_s * 2.0, _MAX_TRANSFER_BACKOFF_S)

        if last_error is not None:
            raise last_error

    def _ensure_leader_plan(self, window_id: int) -> None:
        existing = self.server.control_store.get_lease_plan(window_id)
        if existing.assignments:
            return
        if self._planner is None:
            raise RuntimeError("Adaptive planner is not initialized on bootstrap node")
        if window_id == 0:
            initial_plan = self._planner.plan_window(window_id=0)
            self.server.set_lease_plan(initial_plan)
            self._record_scheduler_window(window_id=0, report=None)
            return

        # Adaptive epoch planning must not stall on every slow follower. Use whatever
        # reports arrived for the previous epoch and keep the existing EMA for nodes
        # that have not reported yet.
        reports = self.server.get_throughput_reports(window_id=window_id - 1)
        lease_plan = self._planner.plan_window(window_id=window_id, reports=reports)
        self.server.set_lease_plan(lease_plan)
        self._record_scheduler_window(window_id=window_id, report=None)

    def _record_scheduler_window(
        self,
        *,
        window_id: int,
        report: ThroughputReportRecord | None,
    ) -> None:
        if self.server.shard_store is None:
            return

        lease_plan = self.server.control_store.get_lease_plan(window_id)
        assigned_shard_ids = list(lease_plan.shards_for_node(self.resolved.self_node_id))
        assigned_samples = sum(
            self.server.shard_store.get_shard_meta(shard_id).sample_count
            for shard_id in assigned_shard_ids
        )
        stats = self.stats()
        row = {
            "window_id": window_id,
            "epoch_id": lease_plan.epoch_id,
            "epoch_window_index": lease_plan.epoch_window_index,
            "epoch_window_count": lease_plan.epoch_window_count,
            "node_id": self.resolved.self_node_id,
            "assigned_shard_count": len(assigned_shard_ids),
            "assigned_sample_count": assigned_samples,
            "samples_processed": None if report is None else report.samples_processed,
            "effective_throughput": None if report is None else report.effective_throughput,
            "local_inventory_count": len(self.server.shard_store.list_local_shards("train")),
            "cache_hit_rate": stats["cache_hit_rate"],
            "shards_pulled_total": stats["shards_pulled"],
            "bytes_transferred_total": stats["bytes_transferred"],
            "prefetch_wait_s_total": stats["prefetch_wait_s"],
            "transfer_retry_count": stats["transfer_retry_count"],
        }
        with self._state_lock:
            self._scheduler_history_rows.append(row)


@dataclass
class _PrefetchTask:
    window_id: int
    thread: threading.Thread | None = None
    completed: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None

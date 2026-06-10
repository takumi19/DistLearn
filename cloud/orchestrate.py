#!/usr/bin/env python3
"""
orchestrate.py — Decentr campaign orchestrator.

Usage:
  python3 orchestrate.py check   --pool pool.yaml
  python3 orchestrate.py run     --pool pool.yaml --suite suites/cifar10_wan_campaign.yaml
  python3 orchestrate.py run     --pool pool.yaml --suite suites/... --only smoke
  python3 orchestrate.py status  --suite suites/cifar10_wan_campaign.yaml
  python3 orchestrate.py reset   --suite suites/... --only <exp_name>
  python3 orchestrate.py shell   --pool pool.yaml --node 0
  python3 orchestrate.py collect --pool pool.yaml --run-id <run_id> [--results-dir ./results]
  python3 orchestrate.py compare [--results-dir ./results] [--run-ids id1 id2 ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

CAMPAIGN_STATE_DIR = Path(__file__).parent / "campaign-state"
BOOTSTRAP_DONE     = ".bootstrap.done"

# Sentinel files written by run-async-node / run-sync-node when training completes.
# The training code writes async_run_summary.json (or sync_run_summary.json for sync mode).
# We check for all variants so a single constant can't silently break completion detection.
# Path on node: {project_dir}/artifacts/logs/{run_name}/{node_id}/<filename>
DONE_FILENAMES = [
    "async_run_summary.json",
    "run_summary.json",
    "sync_run_summary.json",
]
DONE_FILENAME = DONE_FILENAMES[0]   # kept for backward compatibility

GRPC_PORT   = 50051
CONFIG_PORT = 50052   # bootstrap serves HTTP config on GRPC_PORT + 1

# Default training parameters (can be overridden per-experiment in suite YAML).
#
# load_inventory() from the DistLearn codebase expects:
#   - Flat keys for optimization: epochs, batch_size, lr  (extracted directly)
#   - Nested "dataset": {...}  — used as-is; if present, flat dataset keys are ignored
#   - Nested "model": {...}    — used as-is
#   - Nested "async": {...}    — used as-is (async gossip params MUST be here)
#
# micro_shards is required for the adaptive workload scheduler to actually work.
# On the bootstrap node the shard manifest is built automatically at run start
# (~2 min for CIFAR-10 / 512-sample shards).  Followers fetch the manifest and
# pull their assigned shards from the bootstrap over gRPC before training begins.
# Subsequent experiment runs reuse the already-built manifest.
# ── ROADMAP FIXED BASE (01_EXPERIMENT_ROADMAP.ru.md) ─────────────────────────
# Every experiment shares this base; experiments change exactly one factor.
# Re-based 2026-06: lr 0.03→0.005, push_interval 10→100, mixing_alpha 0.5→0.2,
# added gradient_clip_norm=1.0. Old-base results archived under results/_archive_lr0.03/.
# Re-based 2026-06 (v3): lr 0.01→0.03, push_interval 100→10.
# Empirical: lr=0.005→47.7%, lr=0.01 (untested to completion). lr=0.03 +
# push_interval=10 are the only proven settings (56-61% in 50ep).
# lr=0.005/0.01 results archived under results/_archive_lr0.005/.
DEFAULT_TRAINING: dict[str, Any] = {
    # ── Flat optimization keys (extracted by load_inventory directly) ─────────
    "mode":          "async",
    "seed":          42,
    "epochs":        50,
    "batch_size":    32,
    "lr":            0.03,
    "momentum":      0.9,
    "weight_decay":  0.0005,
    "gradient_clip_norm": 1.0,
    # ── Dataset (nested dict required so flat overrides are not silently used) ─
    "dataset": {
        "name":                  "CIFAR10",
        # Absolute path so the cwd of start-run/join-run doesn't matter.
        "root":                  "/opt/decentr/DistLearn/Decentr_my_own/data",
        # micro_shards: data is split into small files; bootstrap builds them,
        # followers pull their assigned subset.  Required for adaptive scheduling.
        "storage_mode":          "micro_shards",
        "shard_samples":         512,
        "manifest_path":         "/opt/decentr/data/manifest.json",
        "cache_dir":             "/opt/decentr/data",
        "partitioning":          "heterogeneous",
        # static by default; async_adaptive_default overrides this to "adaptive".
        "scheduler_mode":        "static",
        "rebalance_window_batches": 50,
        # throughput_ema: how fast the adaptive scheduler reacts to node speed.
        # The EMA sweep (ema_tput_*) overrides this; 0.9 is the roadmap default.
        "throughput_ema":        0.9,
        "val_split":             0.1,
    },
    # ── Model (ResNet-18, groupnorm per roadmap, 10-class head for CIFAR-10) ───
    "model": {
        "name":          "resnet18",
        "num_classes":   10,
        "normalization": "groupnorm",
    },
    # ── Async gossip defaults (overridden per-experiment via "async:" section) ─
    "async": {
        "push_interval_steps": 10,   # proven base: push every 10 steps
        "push_fanout":         0,    # 0 = gossip to ALL neighbours
        "mixing_alpha":        0.2,  # roadmap base weight-blend coefficient
        "max_staleness":       4,    # soft staleness cap (version gap)
    },
}


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    host: str           # private subnet IP — used by peers for gRPC
    ssh_host: str       # IP the orchestrator connects to (floating for bastion,
                        # private for other nodes)
    ssh_user: str    = "decentr"
    ssh_port: int    = 22
    ssh_key_path: str= "~/.ssh/decentr_id_ed25519"
    ssh_jump: str    = ""   # bastion floating IP for ProxyJump; empty = direct
    port: int        = GRPC_PORT
    bind_host: str   = "0.0.0.0"
    platform: str    = "linux"
    project_dir: str = "/opt/decentr/DistLearn/Decentr_my_own"
    python_env: str  = "/opt/decentr/venv"
    weight: float    = 1.0
    cpu_cores: int   = 2
    relative_speed: float = 1.0


@dataclass
class Pool:
    pool_id: str
    provider: str
    nodes: list[Node]
    defaults: dict = field(default_factory=dict)


@dataclass
class Experiment:
    name: str
    enabled: bool
    description: str = ""
    # config keys are merged into the inventory under their respective sections
    # e.g. {"training": {"epochs": 1, "mode": "async"}}
    config: dict = field(default_factory=dict)
    overlay: str = "none"
    # Gossip neighbour graph: full | ring | star | expander. Only affects which
    # peers a node PUSHES model updates to — config/shard fetching still uses the
    # bootstrap directly, so adaptive scheduling is unaffected.
    topology: str = "full"


@dataclass
class Suite:
    suite_name: str
    description: str
    experiments: list[Experiment]


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_pool(path: str | Path) -> Pool:
    raw      = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {})
    nodes    = []
    for n in raw.get("nodes", []):
        m = _deep_merge(defaults, n)
        nodes.append(Node(
            id            = m["id"],
            host          = m["host"],
            ssh_host      = m.get("ssh_host",     m["host"]),
            ssh_user      = m.get("ssh_user",     "decentr"),
            ssh_port      = int(m.get("ssh_port", 22)),
            ssh_key_path  = m.get("ssh_key_path", "~/.ssh/decentr_id_ed25519"),
            ssh_jump      = m.get("ssh_jump",     ""),
            port          = int(m.get("port",     GRPC_PORT)),
            bind_host     = m.get("bind_host",    "0.0.0.0"),
            platform      = m.get("platform",     "linux"),
            project_dir   = m.get("project_dir",  "/opt/decentr/DistLearn/Decentr_my_own"),
            python_env    = m.get("python_env",   "/opt/decentr/venv"),
            weight        = float(m.get("weight",         1.0)),
            cpu_cores     = int(m.get("cpu_cores",        4)),
            relative_speed= float(m.get("relative_speed", 1.0)),
        ))
    return Pool(pool_id=raw.get("pool_id","pool"), provider=raw.get("provider","manual"),
                nodes=nodes, defaults=defaults)


def load_suite(path: str | Path) -> Suite:
    raw = yaml.safe_load(Path(path).read_text())
    exps = [
        Experiment(
            name        = e["name"],
            enabled     = bool(e.get("enabled", True)),
            description = e.get("description", ""),
            config      = e.get("config", {}),
            overlay     = e.get("overlay", "none"),
            topology    = e.get("topology", "full"),
        )
        for e in raw.get("experiments", [])
    ]
    return Suite(suite_name=raw.get("suite_name", Path(path).stem),
                 description=raw.get("description",""), experiments=exps)


# ─── Inventory builder ───────────────────────────────────────────────────────
#
# Generates the inventory YAML consumed by `decentr-my-own start-run --inventory`.
# Format matches configs/inventory.wan-tailscale.example.yaml from the project:
#
#   cluster:
#     name: <run_id>
#     overlay: subnet        # "subnet" for private-IP Selectel nodes
#     bootstrap_node: decentr-01
#
#   nodes:
#     decentr-01:            # dict keyed by node id (not a list!)
#       host: 10.0.0.11
#       port: 50051
#       bind_host: 0.0.0.0
#       platform: linux
#       weight: 1.0
#       resources:
#         cpu_cores: 4
#         accelerator: cpu
#         relative_speed: 1.0
#     ...
#
#   training:
#     mode: async
#     epochs: 50
#     ...

def build_topology(node_ids: list[str], kind: str) -> dict[str, list[str]]:
    """Return a symmetric gossip neighbour graph for the given topology.

    Builds an undirected edge set first, so the result is always symmetric
    (ClusterConfig.validate_graph requires reverse edges). Every node ends up
    with at least one neighbour for n >= 2.
    """
    n = len(node_ids)
    edges: set[tuple[int, int]] = set()

    def add(i: int, j: int) -> None:
        if i != j:
            edges.add((min(i, j), max(i, j)))

    if kind == "full":
        for i in range(n):
            for j in range(i + 1, n):
                add(i, j)
    elif kind == "ring":
        for i in range(n):
            add(i, (i + 1) % n)
    elif kind == "star":
        for i in range(1, n):
            add(0, i)
    elif kind in ("expander", "sparse", "sparse_expander"):
        # Circulant graph: ring chords at offsets {1, ~n/3}. Symmetric by edge set.
        offsets = sorted({1, max(2, n // 3)})
        for i in range(n):
            for off in offsets:
                add(i, (i + off) % n)
    else:
        raise ValueError(f"Unknown topology '{kind}' (full|ring|star|expander)")

    neighbors: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for i, j in sorted(edges):
        neighbors[node_ids[i]].append(node_ids[j])
        neighbors[node_ids[j]].append(node_ids[i])
    return neighbors


def build_inventory(pool: Pool, suite: Suite, exp: Experiment, run_id: str) -> dict:
    bootstrap = pool.nodes[0]

    node_ids = [node.id for node in pool.nodes]
    topology = getattr(exp, "topology", "full")
    neighbor_map = build_topology(node_ids, topology) if topology != "full" else None

    nodes_dict: dict[str, Any] = {}
    for node in pool.nodes:
        entry: dict[str, Any] = {
            "host":      node.host,
            "port":      node.port,
            "bind_host": node.bind_host,
            "platform":  node.platform,
            "weight":    node.weight,
            "resources": {
                "cpu_cores":    node.cpu_cores,
                "accelerator":  "cpu",
                "relative_speed": node.relative_speed,
            },
        }
        # Only non-full topologies set explicit neighbours; full mesh is left to
        # the loader's default so existing runs are byte-for-byte unchanged.
        if neighbor_map is not None:
            entry["neighbors"] = neighbor_map[node.id]
        nodes_dict[node.id] = entry

    inventory: dict[str, Any] = {
        "cluster": {
            "name":           run_id,
            "overlay":        getattr(exp, "overlay", "none"),
            "bootstrap_node": bootstrap.id,
        },
        "nodes": nodes_dict,
        "training": dict(DEFAULT_TRAINING),
    }

    # Deep-merge experiment-level config overrides
    # e.g. exp.config = {"training": {"epochs": 1, "mode": "async"}}
    if exp.config:
        inventory = _deep_merge(inventory, exp.config)

    return inventory


# ─── SSH helpers ─────────────────────────────────────────────────────────────

def _ssh_argv(node: Node) -> list[str]:
    key  = os.path.expanduser(node.ssh_key_path)
    # StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null: nodes are recreated
    # frequently (terraform destroy/apply) so their host keys change with each deploy.
    # All nodes are in our private VPC — we don't need TOFU protection here.
    argv = ["ssh", "-i", key, "-p", str(node.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if node.ssh_jump:
        # ProxyCommand with explicit -i so the jump uses the same key.
        # ProxyJump spawns a subprocess without inheriting -i, so we use
        # ProxyCommand instead to keep full control over auth.
        proxy_cmd = (
            f"ssh -i {key} -W %h:%p"
            f" -o StrictHostKeyChecking=no"
            f" -o UserKnownHostsFile=/dev/null"
            f" -o ConnectTimeout=10"
            f" {node.ssh_user}@{node.ssh_jump}"
        )
        argv += ["-o", f"ProxyCommand={proxy_cmd}"]
    argv.append(f"{node.ssh_user}@{node.ssh_host}")
    return argv


_async_subprocess = getattr(asyncio, "create_subprocess_" + "exec")


async def ssh_run(node: Node, command: str, timeout: float = 30.0) -> tuple[int, str, str]:
    args = _ssh_argv(node) + [command]
    proc = await _async_subprocess(*args,
                                   stdout=asyncio.subprocess.PIPE,
                                   stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # process already exited before we could kill it — harmless
        return -1, "", f"SSH timeout after {timeout}s"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def scp_to(node: Node, local_path: str, remote_path: str) -> bool:
    key  = os.path.expanduser(node.ssh_key_path)
    args = ["scp", "-i", key, "-P", str(node.ssh_port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10"]
    if node.ssh_jump:
        proxy_cmd = (
            f"ssh -i {key} -W %h:%p"
            f" -o StrictHostKeyChecking=no"
            f" -o UserKnownHostsFile=/dev/null"
            f" -o ConnectTimeout=10"
            f" {node.ssh_user}@{node.ssh_jump}"
        )
        args += ["-o", f"ProxyCommand={proxy_cmd}"]
    args += [local_path, f"{node.ssh_user}@{node.ssh_host}:{remote_path}"]
    proc = await _async_subprocess(*args,
                                   stdout=asyncio.subprocess.DEVNULL,
                                   stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        print(f"    scp error: {err.decode(errors='replace').strip()}")
    return proc.returncode == 0


# ─── Campaign state ──────────────────────────────────────────────────────────

def _state_path(suite: Suite, exp: Experiment) -> Path:
    CAMPAIGN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return CAMPAIGN_STATE_DIR / f"{suite.suite_name}__{exp.name}.json"

def _load_state(suite: Suite, exp: Experiment) -> dict:
    p = _state_path(suite, exp)
    return json.loads(p.read_text()) if p.exists() else {}

def _save_state(suite: Suite, exp: Experiment, state: dict) -> None:
    _state_path(suite, exp).write_text(json.dumps(state, indent=2))

def _is_done(suite: Suite, exp: Experiment) -> bool:
    return _load_state(suite, exp).get("status") == "done"


# ─── Node readiness ──────────────────────────────────────────────────────────

async def check_node(node: Node) -> tuple[bool, str]:
    rc, out, err = await ssh_run(node,
        f"test -f /opt/decentr/{BOOTSTRAP_DONE} && echo ready", timeout=15)
    if rc == 0 and "ready" in out:
        return True, "ready"
    if rc == 0:
        return False, "bootstrap not done yet"
    return False, err.strip() or f"ssh failed (rc={rc})"


async def cmd_check(pool: Pool) -> bool:
    print(f"Checking {len(pool.nodes)} node(s)…")
    # Limit to 8 concurrent SSH connections — all go through the bastion (ProxyJump),
    # which has a limited MaxSessions. Exceeding ~10 concurrent sessions causes
    # kex_exchange_identification / Connection reset errors on the jump host.
    sem = asyncio.Semaphore(8)
    async def _guarded(n: Node):
        async with sem:
            return await check_node(n)
    results = await asyncio.gather(*[_guarded(n) for n in pool.nodes])
    ready   = sum(1 for ok, _ in results if ok)
    for node, (ok, msg) in zip(pool.nodes, results):
        print(f"  [{'✓' if ok else '✗'}] {node.id:12s}  {node.ssh_host}  —  {msg}")
    print(f"\n{ready}/{len(pool.nodes)} nodes ready")
    return ready == len(pool.nodes)


# ─── Experiment runner ───────────────────────────────────────────────────────

async def _wait_config_port(node: Node, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rc, out, _ = await ssh_run(node,
            f"nc -z 127.0.0.1 {node.port + 1} 2>/dev/null && echo ok", timeout=8)
        if rc == 0 and "ok" in out:
            return True
        await asyncio.sleep(5)
    return False


async def _kick_bootstrap(node: Node, inv_remote: str, run_name: str) -> bool:
    # < /dev/null detaches stdin so SSH exits immediately after launching the
    # background process; disown removes it from the shell's job table.
    cmd = (f"cd {node.project_dir} && "
           f"nohup {node.python_env}/bin/decentr-my-own start-run "
           f"  --inventory {inv_remote} --self-node {node.id} --run-name {run_name} "
           f"  --transport-timeout-s 600 "
           f"< /dev/null > /tmp/dc_boot_{run_name}.log 2>&1 & disown $!")
    rc, _, err = await ssh_run(node, cmd, timeout=60)
    if rc != 0:
        print(f"    [WARN] bootstrap kick returned rc={rc}: {err.strip()}")
    # rc != 0 can happen if the SSH session closes before disown finishes;
    # treat it as a warning only — _wait_config_port will confirm the result.
    return True


async def _kick_follower(node: Node, bootstrap: Node, run_name: str) -> bool:
    addr = f"{bootstrap.host}:{bootstrap.port + 1}"
    cmd  = (f"cd {node.project_dir} && "
            f"setsid nohup {node.python_env}/bin/decentr-my-own join-run "
            f"  --bootstrap {addr} --self-node {node.id} --transport-timeout-s 600 "
            f"< /dev/null > /tmp/dc_fol_{run_name}_{node.id}.log 2>&1 & echo kicked")
    # Retry up to 3 times — bastion MaxSessions can reject SSH under concurrent load.
    # A rejected kick means the follower process never starts, causing other nodes
    # to time out waiting for it in _wait_for_neighbors.
    for attempt in range(1, 4):
        rc, out, err = await ssh_run(node, cmd, timeout=15)
        if "kicked" in out:
            return True
        print(f"    [WARN] {node.id} kick attempt {attempt}/3 rc={rc}: {err.strip()[:80]}")
        await asyncio.sleep(2)
    print(f"    [ERROR] {node.id} failed to kick after 3 attempts")
    return False


async def _kick_all_followers_from_bastion(
    bastion: Node, followers: list[Node], bootstrap: Node, run_name: str
) -> None:
    """
    Kick all followers via a single SSH to the bastion.
    Bastion has a direct private-subnet path to every follower, so no ProxyCommand
    is needed and all 19 kicks run in parallel on the bastion side.
    """
    # On the bastion the orchestrator key lives at /home/decentr/.ssh/<basename>.
    # We must NOT os.path.expanduser() here — that expands on the local machine.
    # Instead derive the bastion-side path from the key basename.
    key_basename = os.path.basename(os.path.expanduser(bastion.ssh_key_path))
    bastion_key = f"/home/{bastion.ssh_user}/.ssh/{key_basename}"
    addr = f"{bootstrap.host}:{bootstrap.port + 1}"

    # Build a one-liner that SSHes to each follower in parallel from bastion.
    # The bastion must have the orchestrator private key at bastion_key.
    kick_lines = []
    for f in followers:
        # nohup protects the bastion-side SSH from SIGHUP when the orchestrator's
        # SSH session to the bastion closes. Without nohup, background SSH jobs on
        # the bastion receive SIGHUP and die before the follower can start join-run.
        line = (
            f"nohup ssh -i {bastion_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "
            f"-o ConnectTimeout=10 decentr@{f.host} "
            f"'cd {f.project_dir} && "
            f"nohup {f.python_env}/bin/decentr-my-own join-run "
            f"--bootstrap {addr} --self-node {f.id} --transport-timeout-s 600 "
            f"< /dev/null > /tmp/dc_fol_{run_name}_{f.id}.log 2>&1 & disown \\$!' "
            f"< /dev/null > /dev/null 2>&1 &"
        )
        kick_lines.append(line)

    # Fire all SSH jobs in background and return immediately.
    # nohup on each SSH ensures they survive after the bastion session closes.
    # The kicks happen asynchronously; we verify progress via _node_done() polling.
    script = "\n".join(kick_lines) + "\necho 'all_kicked'"
    rc, out, err = await ssh_run(bastion, script, timeout=30)
    if "all_kicked" not in out:
        print(f"    [WARN] bastion kick script may have had issues (rc={rc}): {err.strip()[:200]}")


async def _node_done(node: Node, run_name: str) -> bool:
    # Check for any of the known sentinel filenames (async/sync/legacy).
    base = f"{node.project_dir}/artifacts/logs/{run_name}/{node.id}"
    checks = " || ".join(f"test -f {base}/{fn}" for fn in DONE_FILENAMES)
    rc, out, _ = await ssh_run(node, f"( {checks} ) && echo done", timeout=10)
    return rc == 0 and "done" in out


async def _pull_results(node: Node, run_name: str, local_base: Path) -> bool:
    dst = local_base / node.id
    dst.mkdir(parents=True, exist_ok=True)
    key = os.path.expanduser(node.ssh_key_path)
    # Build rsync SSH wrapper; add ProxyCommand for nodes behind the bastion.
    # ProxyJump cannot be used here because rsync's -e flag spawns a shell
    # that wouldn't inherit -i either.
    ssh_cmd = f"ssh -i {key} -p {node.ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    if node.ssh_jump:
        proxy_cmd = (
            f"ssh -i {key} -W %h:%p"
            f" -o StrictHostKeyChecking=no"
            f" -o UserKnownHostsFile=/dev/null"
            f" {node.ssh_user}@{node.ssh_jump}"
        )
        ssh_cmd += f" -o 'ProxyCommand={proxy_cmd}'"
    # Pull metrics + summary; exclude checkpoints (.pt / .pth).
    # A slow/unreachable node must NEVER crash the orchestrator: a rsync timeout
    # here previously propagated up through asyncio.gather and killed the whole
    # run loop mid-experiment (losing the monitor while training continued
    # headless on the nodes). Catch TimeoutExpired/OSError and treat as a failed
    # pull — the next partial-pull or the final collect will retry.
    try:
        r = subprocess.run([
            "rsync", "-avz", "--ignore-errors",
            "--include=*/",
            "--include=*.csv", "--include=*.json", "--include=*.log",
            "--exclude=*",
            "-e", ssh_cmd,
            f"{node.ssh_user}@{node.ssh_host}:{node.project_dir}/artifacts/logs/{run_name}/{node.id}/",
            str(dst) + "/",
        ], capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"    rsync {node.id}: timed out (skipped, will retry next pull)")
        return False
    except OSError as exc:
        print(f"    rsync {node.id}: {exc} (skipped)")
        return False
    ok = r.returncode in (0, 24)
    if not ok:
        print(f"    rsync {node.id}: {r.stderr.decode(errors='replace').strip()}")
    return ok


async def _cleanup_nodes(pool: Pool) -> None:
    """Kill all decentr processes and wipe artifacts on every node after a run."""
    bootstrap = pool.nodes[0]
    followers = pool.nodes[1:]
    key_local = os.path.expanduser(bootstrap.ssh_key_path)
    key_basename = os.path.basename(key_local)
    bastion_key = f"/home/{bootstrap.ssh_user}/.ssh/{key_basename}"
    proj = bootstrap.project_dir

    KILL = 'pkill -9 -f "decentr-my-own (start|join)-run" 2>/dev/null; true'
    CLEAN = (f'rm -rf {proj}/artifacts/logs/* {proj}/artifacts/checkpoints/*'
             f' /tmp/dc_boot_*.log /tmp/dc_fol_*.log /tmp/inventory_*.yaml 2>/dev/null; true')

    # Bootstrap: kill + clean directly
    print("  [cleanup] bootstrap…", end=" ", flush=True)
    await ssh_run(bootstrap, f"{KILL}; {CLEAN}", timeout=20)
    print("✓")

    # Followers: fan-out via bastion in one SSH session
    if followers:
        lines = []
        for f in followers:
            lines.append(
                f"ssh -i {bastion_key}"
                f" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
                f" -o ConnectTimeout=8 -o BatchMode=yes"
                f" decentr@{f.host}"
                f" '{KILL}; {CLEAN}'"
                f" < /dev/null > /dev/null 2>&1 &"
            )
        script = "\n".join(lines) + "\nwait\necho all_cleaned"
        print(f"  [cleanup] {len(followers)} followers…", end=" ", flush=True)
        rc, out, _ = await ssh_run(bootstrap, script, timeout=60)
        print("✓" if "all_cleaned" in out else "⚠ partial")


async def _publish_results_via_bootstrap(pool: Pool, run_id: str, exp_name: str) -> bool:
    """Bootstrap gathers every node's results and pushes them to decentr-results.

    No data touches the orchestrator (Mac). The bootstrap rsyncs each follower's
    metrics over the cloud-internal network (reliable), copies its own, then
    commits + pushes its decentr-results clone (deploy key in ~/.ssh/config).
    Returns True only if the push (or a clean no-op) succeeded.
    """
    bootstrap = pool.nodes[0]
    followers = pool.nodes[1:]
    key_basename = os.path.basename(os.path.expanduser(bootstrap.ssh_key_path))
    bastion_key = f"/home/{bootstrap.ssh_user}/.ssh/{key_basename}"
    proj = bootstrap.project_dir
    repo = "/opt/decentr/decentr-results"
    src = f"{proj}/artifacts/logs/{run_id}"
    dest = f"{repo}/{run_id}"
    sopts = (f"ssh -i {bastion_key} -o StrictHostKeyChecking=no"
             f" -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes")
    rsync_filter = ("--include='*/' --include='*.csv' --include='*.json'"
                    " --include='*.log' --exclude='*'")

    lines = [
        f"set -uo pipefail",
        f"test -d {repo}/.git || {{ echo NO_RESULTS_REPO; exit 0; }}",
        f"mkdir -p {dest}",
        # Bootstrap's own results (local copy, no SSH).
        f"cp -r {src}/{bootstrap.id} {dest}/ 2>/dev/null || true",
    ]
    # Gather each follower's metrics over the internal network (exclude .pt).
    for f in followers:
        lines.append(
            f"rsync -az {rsync_filter} -e \"{sopts}\" "
            f"decentr@{f.host}:{src}/{f.id}/ {dest}/{f.id}/ 2>/dev/null || true"
        )
    lines += [
        f"cd {repo}",
        f"git add {run_id} 2>/dev/null || true",
        f"git diff --cached --quiet && {{ echo PUBLISH_EMPTY; exit 0; }}",
        f"git commit -m 'results: {exp_name}  run={run_id}' >/dev/null 2>&1 || true",
        f"git pull --rebase --autostash origin main >/dev/null 2>&1 || true",
        f"git push origin main >/dev/null 2>&1 && echo PUBLISH_OK || echo PUBLISH_FAIL",
    ]
    script = "\n".join(lines)
    rc, out, err = await ssh_run(bootstrap, script, timeout=300)
    if "PUBLISH_OK" in out or "PUBLISH_EMPTY" in out:
        return True
    print(f"    [publish] {out.strip()[:120]} {err.strip()[:120]}")
    return False


def _git_push_results(local_res: Path, exp_name: str, run_id: str) -> None:
    """Commit and push one run's results to the dedicated results repo.

    The results dir (``local_res.parent``) is its own git repo whose LOCAL config
    carries the deploy key (core.sshCommand) and bot identity — set up once via
    ``git init`` + remote + ``core.sshCommand``. We operate on that repo directly
    (never walk up to a parent repo) and stage ONLY this run's subdirectory, so
    we never touch files the nodes pushed under a different layout.
    """
    repo = local_res.parent                      # ./results
    run_subdir = local_res.name                  # <run_id>
    if not (repo / ".git").is_dir():
        print(f"  [git] {repo}/.git missing — skipping push "
              f"(run: git init in {repo}, add the decentr-results remote)")
        return
    try:
        # Stage only this run's directory (.gitignore drops *.pt and _archive_*).
        subprocess.run(["git", "add", "--", run_subdir],
                       cwd=repo, check=True, timeout=60)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                cwd=repo, timeout=10)
        if staged.returncode == 0:
            print("  [git] nothing new to commit")
            return
        msg = f"results: {exp_name}  run={run_id}"
        subprocess.run(["git", "commit", "-m", msg], cwd=repo, check=True, timeout=30)
        # Pull-rebase first so a node-side push doesn't reject ours (best-effort).
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"],
                       cwd=repo, timeout=60)
        push = subprocess.run(["git", "push", "origin", "main"],
                              cwd=repo, capture_output=True, timeout=90)
        if push.returncode == 0:
            print(f"  [git] ✓ pushed results for {exp_name}")
        else:
            print(f"  [git] ✗ push failed (committed locally): "
                  f"{push.stderr.decode(errors='replace').strip()[:160]}")
    except subprocess.CalledProcessError as exc:
        print(f"  [git] ✗ git command failed: {exc}")
    except subprocess.TimeoutExpired:
        print("  [git] ✗ git timed out (results committed locally if commit ran)")
    except Exception as exc:
        print(f"  [git] ✗ unexpected error: {exc}")


async def run_experiment(pool: Pool, suite: Suite, exp: Experiment,
                         partial_push_minutes: float, max_runtime_hours: float,
                         dry_run: bool,
                         auto_cleanup: bool = True,
                         auto_git_push: bool = True) -> bool:
    ts     = time.strftime("%Y%m%dT%H%M%S")
    run_id = f"{suite.suite_name}__{exp.name}__{ts}"
    print(f"\n{'='*68}")
    print(f"  Experiment : {exp.name}")
    if exp.description:
        print(f"  Desc       : {exp.description}")
    print(f"  run_id     : {run_id}")
    print(f"  Nodes      : {len(pool.nodes)}")
    print(f"{'='*68}")

    if dry_run:
        inv = build_inventory(pool, suite, exp, run_id)
        print(yaml.dump(inv, default_flow_style=False, indent=2))
        return True

    bootstrap = pool.nodes[0]
    followers = pool.nodes[1:]
    local_res = Path(f"./results/{run_id}")

    import tempfile
    from dataclasses import replace as _dc_replace

    # ── Step 1: kick followers first — they wait up to 600s for bootstrap config ──
    # Copy orchestrator key to bastion so it can SSH followers directly (no ProxyJump,
    # no MaxSessions limit). Followers call fetch_run_config() and retry until bootstrap
    # provides the inventory — so we can start followers BEFORE bootstrap.
    key_local  = os.path.expanduser(bootstrap.ssh_key_path)
    key_remote = f"/home/{bootstrap.ssh_user}/.ssh/{os.path.basename(key_local)}"
    print(f"  Copying SSH key to bastion…")
    await scp_to(bootstrap, key_local, key_remote)
    await ssh_run(bootstrap, f"chmod 600 {key_remote}", timeout=5)

    print(f"  Kicking {len(followers)} followers (they will wait for bootstrap config)…")
    await _kick_all_followers_from_bastion(bootstrap, followers, bootstrap, run_id)

    # ── Step 2: wait 20s, then see who actually started ─────────────────────
    print(f"  Waiting 20s for followers to start…")
    await asyncio.sleep(20)

    _check_sem = asyncio.Semaphore(8)
    async def _proc_running(node: Node) -> bool:
        async with _check_sem:
            rc, out, _ = await ssh_run(node,
                f"pgrep -f 'join-run.*{run_id}' > /dev/null && echo yes", timeout=8)
            return rc == 0 and "yes" in out

    follower_up = await asyncio.gather(*[_proc_running(f) for f in followers])
    active_followers = [f for f, up in zip(followers, follower_up) if up]
    dead_followers   = [f for f, up in zip(followers, follower_up) if not up]
    if dead_followers:
        print(f"  [WARN] {len(dead_followers)} followers did not start: "
              f"{[f.id for f in dead_followers]} — excluded from inventory")
    print(f"  {len(active_followers)}/{len(followers)} followers running")

    if len(active_followers) == 0:
        print("  [ERROR] no followers started"); return False

    # ── Step 3: build inventory with only bootstrap + active followers ───────
    active_nodes = [bootstrap] + active_followers
    active_pool  = _dc_replace(pool, nodes=active_nodes)
    print(f"  Running with {len(active_nodes)} nodes total")

    # ── Step 4: start bootstrap with the reduced inventory ───────────────────
    inv = build_inventory(active_pool, suite, exp, run_id)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(inv, tf, default_flow_style=False)
        inv_local = tf.name

    inv_remote = f"/tmp/inventory_{run_id}.yaml"
    print(f"  SCPing inventory → {bootstrap.id}…")
    if not await scp_to(bootstrap, inv_local, inv_remote):
        os.unlink(inv_local); return False
    os.unlink(inv_local)

    print(f"  start-run on {bootstrap.id}…")
    if not await _kick_bootstrap(bootstrap, inv_remote, run_id):
        return False

    print(f"  Waiting config port {bootstrap.host}:{bootstrap.port+1}…")
    if not await _wait_config_port(bootstrap):
        print("  [ERROR] config port timeout"); return False
    print("  Config port open ✓")

    deadline      = time.monotonic() + max_runtime_hours * 3600
    push_interval = partial_push_minutes * 60
    last_push     = time.monotonic()
    all_done      = False
    # Semaphore shared across the polling loop — limits concurrent SSH through bastion.
    _poll_sem = asyncio.Semaphore(8)

    async def _guarded_done(n: Node) -> bool:
        async with _poll_sem:
            return await _node_done(n, run_id)

    # Detection loop only — no data is pulled to the Mac. _node_done is a cheap
    # `test -f` over SSH; results are published from the bootstrap (below).
    while time.monotonic() < deadline:
        await asyncio.sleep(30)
        flags = await asyncio.gather(*[_guarded_done(n) for n in pool.nodes])
        cnt   = sum(flags)
        print(f"  {cnt}/{len(pool.nodes)} done", end="\r", flush=True)
        if cnt == len(pool.nodes):
            all_done = True; break

    # ── Timeout recovery ─────────────────────────────────────────────────────
    # A transient orchestrator→cluster network drop makes _node_done fail and the
    # loop "time out" even though the nodes actually finished. Before trusting the
    # timeout, re-check completion with retries (the network may have recovered).
    if not all_done:
        print("\n  [recovery] timeout — re-checking completion (network may have dropped)…")
        for attempt in range(6):
            await asyncio.sleep(20)
            flags = await asyncio.gather(*[_guarded_done(n) for n in pool.nodes])
            cnt = sum(flags)
            print(f"  [recovery] attempt {attempt+1}/6: {cnt}/{len(pool.nodes)} done")
            if cnt == len(pool.nodes):
                all_done = True
                break

    print()
    status = "done" if all_done else "timeout"
    _save_state(suite, exp, {"status": status, "run_id": run_id,
                             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"  {'✓' if all_done else '✗'} {exp.name} → {status}")

    # ── Post-run: bootstrap publishes results to git (Mac stores nothing) ─────
    published = False
    if auto_git_push:
        print("  Publishing results from bootstrap → decentr-results…")
        published = await _publish_results_via_bootstrap(pool, run_id, exp.name)
        print("  ✓ published to git" if published
              else "  ✗ publish failed (node artifacts kept for retry)")

    # SAFETY: only wipe nodes once results are safely in git. If publish failed,
    # keep node artifacts so they can be re-published. (If git push is disabled,
    # the user opted out of results, so cleaning is fine.)
    if auto_cleanup and (published or not auto_git_push):
        print("  Cleaning nodes for next run…")
        await _cleanup_nodes(pool)
        print("  Nodes clean ✓")
    elif auto_cleanup:
        print("  [SKIP cleanup] publish failed — preserving node artifacts so the "
              "results can be re-published (re-run collect/publish before next run).")

    return all_done


# ─── Sub-commands ────────────────────────────────────────────────────────────

async def cmd_run(args: argparse.Namespace) -> None:
    pool  = load_pool(args.pool)
    suite = load_suite(args.suite)
    exps  = [e for e in suite.experiments if e.enabled]
    if args.only:
        exps = [e for e in exps if e.name == args.only]
        if not exps:
            sys.exit(f"'{args.only}' not found / not enabled")

    print(f"Suite: {suite.suite_name}  |  {len(exps)} experiment(s)")
    results: dict[str, bool] = {}
    for exp in exps:
        if _is_done(suite, exp) and not args.force:
            print(f"  [skip] {exp.name} (done — use --force to re-run)")
            continue
        ok = await run_experiment(pool, suite, exp,
                                  partial_push_minutes=args.partial_push_minutes,
                                  max_runtime_hours=args.max_runtime_hours,
                                  dry_run=args.dry_run,
                                  auto_cleanup=not args.no_cleanup,
                                  auto_git_push=not args.no_git_push)
        results[exp.name] = ok

    print("\n=== Campaign summary ===")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")


def cmd_status(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    print(f"Suite: {suite.suite_name}")
    print(f"{'Name':<35} {'Enabled':>7}  {'Status':>10}")
    print("-" * 57)
    for exp in suite.experiments:
        state  = _load_state(suite, exp)
        status = state.get("status", "pending") if exp.enabled else "disabled"
        print(f"{exp.name:<35} {'yes' if exp.enabled else 'no':>7}  {status:>10}")


def cmd_reset(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    for exp in suite.experiments:
        if args.only and exp.name != args.only:
            continue
        p = _state_path(suite, exp)
        if p.exists():
            p.unlink(); print(f"  Cleared: {exp.name}")
        else:
            print(f"  No state: {exp.name}")


def cmd_shell(args: argparse.Namespace) -> None:
    pool = load_pool(args.pool)
    na   = args.node
    node = pool.nodes[int(na)] if na.isdigit() else next(
        (n for n in pool.nodes if n.id == na), None)
    if node is None:
        sys.exit(f"Node '{na}' not found")
    print(f"→ {node.id} ({node.ssh_host})")
    subprocess.call(_ssh_argv(node))


# ─── collect command ─────────────────────────────────────────────────────────
#
# Rsync logs/metrics from all nodes for a finished (or interrupted) run.
# Useful when the orchestrator was killed before the final pull, or when you
# want to re-collect results without re-running the experiment.
#
# Results land in: {results_dir}/{run_id}/{node_id}/*.csv|json|log

async def cmd_collect(args: argparse.Namespace) -> None:
    pool       = load_pool(args.pool)
    run_id     = args.run_id
    results_dir = Path(args.results_dir)
    local_base  = results_dir / run_id
    local_base.mkdir(parents=True, exist_ok=True)

    print(f"Collecting run: {run_id}")
    print(f"Nodes         : {len(pool.nodes)}")
    print(f"Destination   : {local_base.resolve()}")
    print()

    tasks  = [_pull_results(n, run_id, local_base) for n in pool.nodes]
    flags  = await asyncio.gather(*tasks)
    ok_cnt = sum(flags)

    print()
    for node, ok in zip(pool.nodes, flags):
        dst = local_base / node.id
        files = list(dst.rglob("*")) if dst.exists() else []
        n_files = sum(1 for f in files if f.is_file())
        print(f"  [{'✓' if ok else '✗'}] {node.id:12s}  {n_files} file(s)")

    print(f"\n{ok_cnt}/{len(pool.nodes)} nodes collected successfully")


# ─── compare command ──────────────────────────────────────────────────────────
#
# Reads run_summary.json from collected results and prints a comparison table.
# Works on whatever keys happen to be present in run_summary.json.
#
# Usage:
#   python3 orchestrate.py compare --results-dir ./results
#   python3 orchestrate.py compare --results-dir ./results --run-ids smoke async_static

_COMPARE_FLOAT_KEYS = [
    # ── Accuracy / quality ────────────────────────────────────────────────────
    "best_test_accuracy",           # max test acc across epochs
    "final_test_accuracy",          # test acc at last epoch (derived from final_test_metrics.accuracy)
    "best_val_accuracy",            # max val acc across epochs
    "best_test_macro_f1",           # max macro-F1 across epochs
    # ── Timing ───────────────────────────────────────────────────────────────
    "run_duration_s",               # wall-clock time for the whole run (≡ old "wall_time_s")
    "compute_time_s",               # derived: run_duration_s − push_elapsed_s_total
    "push_elapsed_s_total",         # total time spent on gossip push RPCs (comm/transfer time)
    "avg_push_latency_s",           # derived: push_elapsed_s_total / push_count_total
    # ── Throughput ───────────────────────────────────────────────────────────
    "effective_samples_per_s",      # training samples / run_duration_s per node
    "total_samples_processed",      # total samples trained on, per node
    # ── Gossip & workload ────────────────────────────────────────────────────
    "gini_workload",                # Gini coefficient of workload share (from data_plane_stats)
    "mixed_peer_updates_total",     # number of peer weight merges accepted
    "max_observed_staleness",       # max raw version gap seen (≡ old "avg_staleness" — was misnamed)
    "failed_pushes_total",          # push RPCs that failed
    "push_success_rate",            # derived: successes / (successes + failures)
]


def _load_run_summaries(results_dir: Path, run_ids: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    """
    Returns: { run_id: { metric: aggregated_value } }

    For each run, collects the per-node summary JSON from all node subdirs,
    flattens nested keys, derives computed metrics, then averages numeric
    fields across nodes.  Per-node raw data is kept under "__nodes".

    Summary file candidates (checked in order): async_run_summary.json,
    run_summary.json, sync_run_summary.json.
    """
    summaries: dict[str, dict[str, Any]] = {}

    if run_ids:
        run_dirs = [results_dir / rid for rid in run_ids if (results_dir / rid).is_dir()]
    else:
        run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    for run_dir in run_dirs:
        run_id   = run_dir.name
        per_node: list[dict] = []

        for node_dir in sorted(run_dir.iterdir()):
            if not node_dir.is_dir():
                continue
            raw: Optional[dict] = None
            for fname in DONE_FILENAMES:
                p = node_dir / fname
                if p.exists():
                    try:
                        raw = json.loads(p.read_text())
                    except Exception:
                        pass
                    break
            if raw is None:
                continue

            # ── Flatten nested final_test_metrics ────────────────────────────
            if isinstance(raw.get("final_test_metrics"), dict):
                ftm = raw["final_test_metrics"]
                raw.setdefault("final_test_accuracy", ftm.get("accuracy"))
                raw.setdefault("final_test_loss",     ftm.get("loss"))
                raw.setdefault("final_test_macro_f1", ftm.get("macro_f1"))

            # ── Derive push_success_rate ──────────────────────────────────────
            push_total  = raw.get("push_count_total",   0) or 0
            push_failed = raw.get("failed_pushes_total", 0) or 0
            if push_total + push_failed > 0:
                raw.setdefault("push_success_rate",
                               push_total / (push_total + push_failed))

            # ── Derive avg_push_latency_s ─────────────────────────────────────
            push_elapsed = raw.get("push_elapsed_s_total", 0.0) or 0.0
            if push_total > 0:
                raw.setdefault("avg_push_latency_s", push_elapsed / push_total)

            # ── Derive compute_time_s = wall_time − comm_time ────────────────
            run_dur = raw.get("run_duration_s", 0.0) or 0.0
            if run_dur > 0:
                raw.setdefault("compute_time_s", max(0.0, run_dur - push_elapsed))

            # ── Extract gini from data_plane_stats (adaptive mode) ────────────
            dps = raw.get("data_plane_stats")
            if isinstance(dps, dict):
                gini = dps.get("gini_workload") or dps.get("gini")
                if gini is not None:
                    raw.setdefault("gini_workload", gini)

            per_node.append(raw)

        if not per_node:
            continue

        # Average numeric fields across nodes
        agg: dict[str, Any] = {"__n_nodes": len(per_node), "__nodes": per_node}
        all_keys = {k for d in per_node for k in d if not k.startswith("_")}
        for key in all_keys:
            vals = [d[key] for d in per_node
                    if key in d and isinstance(d[key], (int, float))]
            if vals:
                agg[key] = sum(vals) / len(vals)

        summaries[run_id] = agg

    return summaries


def cmd_compare(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir}")

    run_ids = args.run_ids if args.run_ids else None
    data    = _load_run_summaries(results_dir, run_ids)

    if not data:
        print("No run summary files found. Run 'collect' first.")
        return

    # Determine which metrics to show
    all_keys = {k for d in data.values() for k in d if not k.startswith("_")}
    show_keys = [k for k in _COMPARE_FLOAT_KEYS if k in all_keys]
    extra     = sorted(all_keys - set(_COMPARE_FLOAT_KEYS))
    show_keys += extra

    if not show_keys:
        print("No numeric metrics found in run_summary.json files.")
        return

    # Column widths
    run_col   = max(len(rid) for rid in data) + 2
    metric_col = max(len(k) for k in show_keys) + 2
    val_col    = 12

    # Header
    print(f"\n{'Run':<{run_col}}  {'Nodes':>5}", end="")
    for k in show_keys:
        print(f"  {k:>{val_col}}", end="")
    print()
    print("-" * (run_col + 7 + len(show_keys) * (val_col + 2)))

    for run_id, agg in sorted(data.items()):
        n = agg.get("__n_nodes", "?")
        print(f"{run_id:<{run_col}}  {n:>5}", end="")
        for k in show_keys:
            val = agg.get(k, None)
            if val is None:
                print(f"  {'—':>{val_col}}", end="")
            elif isinstance(val, float):
                # Show percentages for accuracy/f1, raw for others
                if "accuracy" in k or "f1" in k or "rate" in k or "gini" in k:
                    print(f"  {val * 100:>{val_col - 1}.2f}%", end="")
                elif ("latency" in k or "duration" in k or "wall_time" in k
                      or k in ("run_duration_s", "compute_time_s",
                               "push_elapsed_s_total", "avg_push_latency_s")):
                    print(f"  {val:>{val_col}.1f}s", end="")
                else:
                    print(f"  {val:>{val_col}.3f}", end="")
            else:
                print(f"  {str(val):>{val_col}}", end="")
        print()

    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Decentr campaign orchestrator")
    s = p.add_subparsers(dest="cmd", required=True)

    chk = s.add_parser("check");  chk.add_argument("--pool", required=True)

    run = s.add_parser("run")
    run.add_argument("--pool",   required=True)
    run.add_argument("--suite",  required=True)
    run.add_argument("--only",   default="")
    run.add_argument("--force",  action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--partial-push-minutes", type=float, default=10)
    run.add_argument("--max-runtime-hours",    type=float, default=8)
    run.add_argument("--no-cleanup",   action="store_true",
                     help="skip node cleanup (kill+rm) after each experiment")
    run.add_argument("--no-git-push",  action="store_true",
                     help="skip git push of results after each experiment")

    st = s.add_parser("status"); st.add_argument("--suite", required=True)

    rs = s.add_parser("reset")
    rs.add_argument("--suite", required=True); rs.add_argument("--only", default="")

    sh = s.add_parser("shell")
    sh.add_argument("--pool", required=True); sh.add_argument("--node", required=True)

    col = s.add_parser("collect", help="Rsync results from all nodes for a given run_id")
    col.add_argument("--pool",        required=True)
    col.add_argument("--run-id",      required=True,  dest="run_id")
    col.add_argument("--results-dir", default="./results", dest="results_dir")

    cmp = s.add_parser("compare", help="Compare run_summary.json across collected runs")
    cmp.add_argument("--results-dir", default="./results", dest="results_dir")
    cmp.add_argument("--run-ids",     nargs="*",  dest="run_ids",
                     help="Specific run IDs to compare (default: all in results-dir)")

    args = p.parse_args()
    dispatch = {
        "check":   lambda: asyncio.run(cmd_check(load_pool(args.pool))),
        "run":     lambda: asyncio.run(cmd_run(args)),
        "status":  lambda: cmd_status(args),
        "reset":   lambda: cmd_reset(args),
        "shell":   lambda: cmd_shell(args),
        "collect": lambda: asyncio.run(cmd_collect(args)),
        "compare": lambda: cmd_compare(args),
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()

# Decentr_my_own

`Decentr_my_own` — это playground для децентрализованного распределённого обучения на гетерогенных узлах.

Система разделена на три плоскости:

- `weight plane`: peer-to-peer обмен весами модели по gRPC
- `data plane`: передача данных не батчами, а immutable `micro_shards`
- `control plane`: lease plan, throughput reports, adaptive scheduling

Сейчас поддерживаются:

- `replicated` и `micro_shards` режимы хранения датасета
- режимы `local-train`, `sync` и `async`
- static и adaptive shard lease planning
- pull-based shard transfer по gRPC
- prefetch, cache accounting, retry/backoff и `scheduler_history`
- run-level reporting с `accuracy`, `macro_f1`, throughput и data-plane метриками

Дополнительные документы:

- `PROJECT_GUIDE.md`
- `MULTI_HOST_RUNBOOK.md`

## Установка

Запускай из корня репозитория, где лежит папка `Decentr_my_own/`.

```bash
cd /path/to/final_proj

python3 -m venv .venv
. .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e ./Decentr_my_own
```

Основной CLI:

```bash
python -m decentr_my_own.cli --help
```

Если пакет ещё не установлен, временный fallback:

```bash
PYTHONPATH="$PWD/Decentr_my_own" python -m decentr_my_own.cli --help
```

## Быстрая проверка

Запустить тесты:

```bash
python -m unittest discover -s Decentr_my_own/tests
```

Быстрый локальный async smoke:

```bash
python -m decentr_my_own.cli async-smoke --peer-count 3 --rounds 2
```

Локальный async smoke с `micro_shards` и adaptive planner:

```bash
PYTHONPATH="$PWD/Decentr_my_own" python - <<'PY'
import json
from decentr_my_own.algorithms.async_gossip import run_async_smoke

result = run_async_smoke(
    peer_count=3,
    rounds=2,
    storage_mode="micro_shards",
    scheduler_mode="adaptive",
)
print(json.dumps(result, indent=2))
PY
```

## Полный Runbook: Mac + 2 Linux VPS

Рекомендуемая топология:

- `node-mac`: bootstrap node
- `node-vps1`: follower
- `node-vps2`: follower

Все машины должны быть в одном Tailscale tailnet.

### 1. Подготовить хосты

На каждом Linux VPS:

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv python3-pip tmux
```

Поставить Tailscale и войти в tailnet:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4
```

Если на VPS включён firewall, открыть gRPC порт:

```bash
sudo ufw allow 55051/tcp
sudo ufw status
```

### 1.1 Как узнать IP

Для `cluster.yaml` нужно использовать:

- либо `Tailscale IPv4`
- либо `Tailscale MagicDNS hostname`

Посмотреть Tailscale IPv4:

```bash
tailscale ip -4
```

Посмотреть статус tailnet и имена хостов:

```bash
tailscale status
```

Если нужен публичный IP VPS для `ssh` или `rsync`:

```bash
curl -4 ifconfig.me && echo
```

LAN IP на Linux:

```bash
hostname -I
ip addr show
```

LAN IP на macOS:

```bash
ipconfig getifaddr en0 || ipconfig getifaddr en1
```

Что использовать где:

- `cluster.yaml -> node.host`: Tailscale IP или MagicDNS имя
- `ssh` с Mac на VPS: публичный IP или Tailscale IP
- `rsync` логов: то же самое, что для `ssh`

SSH через Tailscale:

```bash
ssh <user>@<VPS_TAILSCALE_IP>
```

Если SSH-ключей ещё нет:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-copy-id <user>@<VPS_PUBLIC_OR_TAILSCALE_IP>
```

Если на macOS нет `ssh-copy-id`:

```bash
cat ~/.ssh/id_ed25519.pub
```

Потом добавить этот ключ в `~/.ssh/authorized_keys` на VPS.

### 2. Создать WAN-конфиги

Создай `Decentr_my_own/configs/cluster.wan-3node.async.yaml`:

```yaml
cluster_name: decentr-wan-3node-async
transport: grpc
overlay_network: tailscale
tls_enabled: false
bootstrap_node_id: node-mac
nodes:
  - id: node-mac
    host: MAC_TS_IP
    bind_host: 0.0.0.0
    port: 55051
    platform: macos
    neighbors: [node-vps1, node-vps2]
    weight: 1.0
    resources:
      cpu_cores: 8
      accelerator: mps
      relative_speed: 1.2

  - id: node-vps1
    host: VPS1_TS_IP
    bind_host: 0.0.0.0
    port: 55051
    platform: linux
    neighbors: [node-mac, node-vps2]
    weight: 1.0
    resources:
      cpu_cores: 4
      accelerator: cpu
      relative_speed: 1.0

  - id: node-vps2
    host: VPS2_TS_IP
    bind_host: 0.0.0.0
    port: 55051
    platform: linux
    neighbors: [node-mac, node-vps1]
    weight: 1.0
    resources:
      cpu_cores: 4
      accelerator: cpu
      relative_speed: 1.0
```

Создай `Decentr_my_own/configs/training.wan-async.micro.yaml`:

```yaml
seed: 42
algorithm: gossip
mode: async
device_preference: [cuda, mps, cpu]

model:
  name: resnet18
  num_classes: 100
  normalization: groupnorm

dataset:
  name: CIFAR100
  root: ./Decentr_my_own/data
  storage_mode: micro_shards
  manifest_path: ./Decentr_my_own/artifacts/shards/manifest.json
  cache_dir: ./Decentr_my_own/artifacts/shards
  shard_samples: 256
  max_cache_bytes: 2147483648
  prefetch_shards: 1
  transfer_chunk_bytes: 1048576
  scheduler_mode: adaptive
  rebalance_window_batches: 1
  throughput_ema: 0.9
  warmup_windows: 1
  min_local_shards: 1
  transfer_policy: pull
  distributed_eval: false
  partitioning: heterogeneous
  shuffle_scope: round_seeded
  val_split: 0.1
  num_workers: 2
  download: true

optimization:
  epochs: 10
  local_steps: 25
  batch_size: 64
  lr: 0.03
  momentum: 0.9
  weight_decay: 0.0005
  eval_every_epochs: 1

sync:
  enabled: false
  averaging: deltas
  barrier_timeout_s: 120

async:
  enabled: true
  push_interval_steps: 10
  max_staleness: 4
  mixing_alpha: 0.5

logging:
  log_dir: ./Decentr_my_own/artifacts/logs
  checkpoint_dir: ./Decentr_my_own/artifacts/checkpoints
  save_every_round: 1
  metrics_format: csv
```

### 2.1 Для чего нужны основные параметры

Режим данных и scheduler:

- `dataset.storage_mode`
  - `replicated`: каждая нода читает полный датасет локально
  - `micro_shards`: bootstrap хранит shard set, followers тянут нужные shard'ы по gRPC
- `dataset.scheduler_mode`
  - `static`: распределение shard'ов фиксировано заранее
  - `adaptive`: leader пересчитывает будущие lease по throughput reports
- `dataset.shuffle_scope`
  - `global_seeded`: одна детерминированная shuffle-схема
  - `round_seeded`: новая детерминированная shuffle на каждый round/window
- `dataset.partitioning`
  - `homogeneous`: считать все ноды равными
  - `heterogeneous`: учитывать `weight` и `relative_speed`
- `dataset.shard_samples`
  - число sample на один shard
  - для CIFAR100 практичный старт: `256`
- `dataset.rebalance_window_batches`
  - сколько работы планировщик включает в одно adaptive окно
  - `1` — самый безопасный старт для WAN
- `dataset.prefetch_shards`
  - сколько shard'ов подтягивать заранее в фоне
- `dataset.max_cache_bytes`
  - лимит кэша shard'ов на ноде

Async-параметры:

- `async.push_interval_steps`
  - как часто пушить веса соседям
  - меньше значение — более свежие updates, но больше трафика
- `async.max_staleness`
  - насколько старый peer update ещё допустим
- `async.mixing_alpha`
  - насколько сильно смешивать локальное состояние с peer update

Optimization:

- `optimization.local_steps`
  - сколько локальных batch steps выполнять за round/window
- `optimization.batch_size`
  - batch size на одну ноду
- `optimization.epochs`
  - в текущих distributed path используется как естественный round count

Logging:

- `logging.save_every_round`
  - сохранять checkpoint каждые N раундов
  - для live-мониторинга держи `1`
- `logging.metrics_format`
  - `csv` удобнее для быстрого просмотра

Рекомендуемый первый реальный WAN run:

- `dataset.scheduler_mode: adaptive`
- `dataset.shuffle_scope: round_seeded`
- `dataset.partitioning: heterogeneous`
- `dataset.rebalance_window_batches: 1`
- `dataset.prefetch_shards: 1`
- `async.push_interval_steps: 10`
- `async.max_staleness: 4`
- `async.mixing_alpha: 0.5`
- `optimization.batch_size: 32` или `64`
- `optimization.local_steps: 10` или `25`
- `logging.save_every_round: 1`

### 3. Как выложить на Git

Текущий remote в этом workspace:

- `origin = https://github.com/takumi19/DistLearn.git`

На Mac:

```bash
cd /path/to/final_proj
git remote -v
git switch -c codex/wan-async-microshards
```

Добавь только нужные файлы проекта:

```bash
git add Decentr_my_own/decentr_my_own
git add Decentr_my_own/configs
git add Decentr_my_own/README.md
git add Decentr_my_own/MULTI_HOST_RUNBOOK.md
git add Decentr_my_own/PROJECT_GUIDE.md
git add Decentr_my_own/pyproject.toml
git status --short
```

Закоммить и отправь:

```bash
git commit -m "Prepare WAN async micro-shards run"
git push -u origin codex/wan-async-microshards
```

Не добавляй в commit `artifacts/`.

Если push по HTTPS просит логин:

```bash
gh auth login
```

Или переключи remote на SSH:

```bash
git remote set-url origin git@github.com:takumi19/DistLearn.git
git push -u origin codex/wan-async-microshards
```

Если ветка уже существует:

```bash
cd /path/to/final_proj
git switch codex/wan-async-microshards
git pull --ff-only origin codex/wan-async-microshards
```

Если потом меняешь конфиги и хочешь обновить VPS:

```bash
cd /path/to/final_proj
git add Decentr_my_own/configs Decentr_my_own/README.md
git commit -m "Tune WAN async run config"
git push origin codex/wan-async-microshards
```

### 4. Как скачать с Git на VPS

Первый clone:

```bash
git clone https://github.com/takumi19/DistLearn.git ~/DistLearn
cd ~/DistLearn
git checkout codex/wan-async-microshards
```

Если репозиторий private, клонируй по SSH:

```bash
git clone git@github.com:takumi19/DistLearn.git ~/DistLearn
cd ~/DistLearn
git checkout codex/wan-async-microshards
```

Если repo уже есть и нужно просто подтянуть:

```bash
cd ~/DistLearn
git fetch origin
git checkout codex/wan-async-microshards
git pull --ff-only origin codex/wan-async-microshards
```

После каждого `git pull` обновляй editable install:

```bash
. .venv/bin/activate
python -m pip install -e ./Decentr_my_own
```

### 5. Поставить зависимости на всех нодах

На Mac:

```bash
cd /path/to/final_proj
test -d .venv || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ./Decentr_my_own
```

На каждом VPS:

```bash
cd ~/DistLearn
test -d .venv || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ./Decentr_my_own
```

Проверка:

```bash
python -m decentr_my_own.cli --help
```

### 6. Проверить конфиги

На Mac:

```bash
python -m decentr_my_own.cli validate-config \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-mac

python -m decentr_my_own.cli wan-preflight \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-mac \
  --check-dns
```

На `node-vps1`:

```bash
python -m decentr_my_own.cli validate-config \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps1

python -m decentr_my_own.cli wan-preflight \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --check-dns
```

На `node-vps2`:

```bash
python -m decentr_my_own.cli validate-config \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps2

python -m decentr_my_own.cli wan-preflight \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --check-dns
```

### 7. Построить shard'ы только на bootstrap

Только на Mac:

```bash
python -m decentr_my_own.cli build-shards \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml
```

Чтобы пересобрать с нуля:

```bash
python -m decentr_my_own.cli build-shards \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --force
```

Followers shard'ы не строят. Они сами:

- получают manifest у bootstrap
- тянут нужные shard'ы по gRPC

### 8. Точный запуск async + adaptive + round_seeded

Эти команды запускают именно:

- `mode: async`
- `dataset.storage_mode: micro_shards`
- `dataset.scheduler_mode: adaptive`
- `dataset.shuffle_scope: round_seeded`

Очень важно:

- использовать одинаковый `--run-name` на всех нодах
- запускать из корня репозитория
- стартовать сначала bootstrap, потом followers

Выбери run id:

```bash
export RUN_ID=wan-async-3node-20260425
```

Порядок запуска:

1. `node-mac`
2. `node-vps1`
3. `node-vps2`

Bootstrap на Mac:

```bash
cd /path/to/final_proj
. .venv/bin/activate

python -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-mac \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

`node-vps1` через `tmux`:

```bash
cd ~/DistLearn
tmux new -s async-vps1
. .venv/bin/activate

python -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

Detach: `Ctrl-b`, потом `d`.

`node-vps2` через `tmux`:

```bash
cd ~/DistLearn
tmux new -s async-vps2
. .venv/bin/activate

python -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

Если хочешь запускать в фоне с логом:

```bash
python -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60 \
  > /tmp/node-vps1.async.log 2>&1 &
```

### 9. Проверить связность после старта

На любой ноде:

```bash
python -m decentr_my_own.cli probe-neighbors \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-mac \
  --include-state
```

Замени `node-mac` на `node-vps1` или `node-vps2`, если проверяешь с другой машины.

## Как смотреть промежуточные реальные логи

Есть два типа артефактов:

- live process output: всё, что процесс пишет в `stdout/stderr`
- persisted run artifacts: checkpoints, metrics, summaries

Важная деталь:

- `epoch_metrics.csv`, `sync_round_metrics.csv`, `async_round_metrics.csv` и `*_run_summary.json` пишутся в конце run
- checkpoints пишутся во время выполнения, если срабатывает `save_every_round`

Значит лучший способ смотреть промежуточный прогресс:

- редиректить `stdout/stderr` в файл
- во время run смотреть checkpoint-файлы

### Redirect process logs

Пример для VPS:

```bash
python -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-3node.async.yaml \
  --training Decentr_my_own/configs/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60 \
  > /tmp/node-vps1.async.log 2>&1
```

Смотреть live:

```bash
tail -f /tmp/node-vps1.async.log
```

### Смотреть промежуточные round checkpoints

Для `async` checkpoints лежат в:

```text
Decentr_my_own/artifacts/checkpoints/<run_id>/<node_id>/async-round-XXX.pt
```

Для `sync`:

```text
Decentr_my_own/artifacts/checkpoints/<run_id>/<node_id>/round-XXX.pt
```

Для `local-train`:

```text
Decentr_my_own/artifacts/checkpoints/<run_id>/<node_id>/epoch-XXX.pt
```

Внутри checkpoint уже лежит metrics row для этого epoch/round.

Напечатать все текущие async checkpoints:

```bash
python - <<'PY'
from pathlib import Path
import torch

run_id = "wan-async-3node-20260425"
node_id = "node-vps1"
checkpoint_dir = Path("Decentr_my_own/artifacts/checkpoints") / run_id / node_id

for path in sorted(checkpoint_dir.glob("async-round-*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    print(path.name, payload["metrics"])
PY
```

Обновлять каждые 5 секунд:

```bash
while true; do
  clear
  python - <<'PY'
from pathlib import Path
import torch

run_id = "wan-async-3node-20260425"
node_id = "node-vps1"
checkpoint_dir = Path("Decentr_my_own/artifacts/checkpoints") / run_id / node_id

for path in sorted(checkpoint_dir.glob("async-round-*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload["metrics"]
    print(
        path.name,
        "round=", metrics.get("round"),
        "samples=", metrics.get("samples_processed"),
        "train_acc=", metrics.get("local_train_accuracy"),
        "val_acc=", metrics.get("val_accuracy"),
        "test_acc=", metrics.get("test_accuracy"),
        "samples_per_s=", metrics.get("samples_per_s"),
    )
PY
  sleep 5
done
```

Если нужен live-view по эпохам для `local-train`, просто замени `async-round-*.pt` на `epoch-*.pt`.

### Какие файлы появляются после завершения run

`local-train`:

- `epoch_metrics.csv` или `epoch_metrics.json`
- `run_summary.json`

`sync`:

- `sync_round_metrics.csv` или `sync_round_metrics.json`
- `sync_run_summary.json`
- `scheduler_history.csv` для adaptive micro-shards runs

`async`:

- `async_round_metrics.csv` или `async_round_metrics.json`
- `async_run_summary.json`
- `scheduler_history.csv` для adaptive micro-shards runs

## Как собрать логи обратно на Mac

Чтобы построить общий отчёт, нужно собрать per-node logs в один `log_root`.

На Mac:

```bash
export VPS1_SSH=<user>@<vps1_public_ip_or_dns>
export VPS2_SSH=<user>@<vps2_public_ip_or_dns>
export RUN_ID=wan-async-3node-20260425

mkdir -p Decentr_my_own/artifacts/logs/"$RUN_ID"

rsync -avz "$VPS1_SSH":~/DistLearn/Decentr_my_own/artifacts/logs/"$RUN_ID"/ \
  Decentr_my_own/artifacts/logs/"$RUN_ID"/

rsync -avz "$VPS2_SSH":~/DistLearn/Decentr_my_own/artifacts/logs/"$RUN_ID"/ \
  Decentr_my_own/artifacts/logs/"$RUN_ID"/
```

## Как построить итоговый отчёт

Когда все node summaries лежат под одним `log_root`:

```bash
python -m decentr_my_own.cli report-run \
  --log-root Decentr_my_own/artifacts/logs \
  --run-id "$RUN_ID"
```

Что смотреть в результате:

- `final_test_metrics.accuracy`
- `final_test_metrics.macro_f1`
- `mixed_peer_updates_total`
- `push_count_total`
- `data_plane_stats.shards_pulled`
- `data_plane_stats.bytes_transferred`
- `data_plane_stats.cache_hit_rate`
- `scheduler_history_file`

## Частые проблемы

`ModuleNotFoundError: decentr_my_own`

- поставь пакет: `python -m pip install -e ./Decentr_my_own`
- или временно используй `PYTHONPATH="$PWD/Decentr_my_own"`

`probe-neighbors` не видит peer

- проверь `host`
- проверь Tailscale connectivity
- проверь firewall
- проверь совпадение порта

Follower не может получить shard'ы

- bootstrap не запущен
- bootstrap не собрал manifest/shards
- bootstrap порт закрыт

`report-run` не находит summaries

- ноды стартовали с разными `--run-name`
- логи не были собраны в один `log_root`

## Практическая рекомендация для первого WAN запуска

Для первого bring-up safest config:

- `dataset.rebalance_window_batches: 1`
- `dataset.prefetch_shards: 1`
- `optimization.batch_size: 32`
- `optimization.local_steps: 10`
- `optimization.epochs: 4`
- `logging.save_every_round: 1`

После того как plumbing подтверждён, можно увеличивать нагрузку и переключаться с `FakeData` на `CIFAR100`.

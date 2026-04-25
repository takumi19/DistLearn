# Подробный Runbook: Mac + 2 VPS

Этот документ собирает в одном месте полный порядок запуска `Decentr_my_own` в режиме:

- `async`
- `micro_shards`
- `adaptive`
- `round_seeded`

Сценарий рассчитан на:

- `Mac` как bootstrap leader
- `VPS1` как follower `node-vps1`
- `VPS2` как follower `node-vps2`

Документ отвечает на практические вопросы:

- какой IP где использовать
- как выложить код в git и как скачать его на VPS
- как не менять системный Python глобально, а использовать `3.12` только для проекта
- как запускать обучение правильно
- как смотреть промежуточные логи
- как собирать итоговые артефакты
- что делать при типовых сбоях

## 1. Что именно запускаем

Архитектура разбита на три плоскости:

- `weight plane`: peer-to-peer обмен весами модели по gRPC
- `data plane`: передача данных как immutable `micro_shards`
- `control plane`: lease planner, throughput reports, adaptive shard scheduling

Для этого runbook используются:

- bootstrap node: `node-mac`
- followers: `node-vps1`, `node-vps2`
- transport port: `55051`

## 2. Какие IP бывают и какой куда вставлять

Есть три типа адресов.

### 2.1 Tailscale IP

Это адрес вида `100.x.y.z`.

Он нужен для связи нод друг с другом внутри кластера.

Именно его надо писать в `cluster.yaml`.

Узнать:

```bash
tailscale ip -4
```

### 2.2 Public IP

Это внешний IP VPS.

Он нужен только для:

- `ssh`
- `scp`
- `rsync`

В `cluster.yaml` его писать не надо, если запуск идёт через Tailscale.

Узнать:

```bash
curl -4 ifconfig.me && echo
```

### 2.3 Local LAN IP

Это адрес вида `192.168.x.x` или `10.x.x.x`.

Для WAN/Tailscale запуска он не нужен.

## 3. Текущая роль машин

Для твоего стенда:

- `Mac`
  - `Tailscale IP`: `100.110.148.103`
- `VPS1`
  - `Tailscale IP`: `100.114.132.57`
  - `Public IP`: `188.225.26.247`
- `VPS2`
  - `Tailscale IP`: `100.84.195.80`
  - `Public IP`: `37.220.84.48`

Использование:

- в `cluster.yaml`
  - `node-mac.host = 100.110.148.103`
  - `node-vps1.host = 100.114.132.57`
  - `node-vps2.host = 100.84.195.80`
- для `ssh`
  - `ssh root@188.225.26.247`
  - `ssh root@37.220.84.48`

## 4. Структура файлов

Код проекта находится в репозитории:

- локально на `Mac`: `/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj`
- на VPS: `~/DistLearn`

Публичный код и документация находятся в git.

Приватные WAN-конфиги с реальными IP лучше держать вне git:

- на `Mac`: `~/decentr-run`
- на VPS: `/root/decentr-run`

Причина простая: реальные IP и tailnet-адреса не стоит коммитить в публичную ветку.

## 5. Какая git-ветка используется

Актуальная ветка:

```text
codex/decentr-my-own-runbook-20260425
```

## 6. Как выложить код в git

На `Mac`, из корня репозитория:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
git fetch origin
git checkout codex/decentr-my-own-runbook-20260425
git pull --ff-only origin codex/decentr-my-own-runbook-20260425
```

Если нужно закоммитить свои изменения:

```bash
git add Decentr_my_own
git commit -m "Your change"
git push origin codex/decentr-my-own-runbook-20260425
```

Важно:

- не коммить `~/decentr-run/*`, если там реальные IP
- не коммить `artifacts/`

## 7. Как скачать код на VPS

Если репозиторий ещё не клонирован:

```bash
git clone https://github.com/takumi19/DistLearn.git ~/DistLearn
cd ~/DistLearn
git checkout codex/decentr-my-own-runbook-20260425
```

Если репозиторий уже есть:

```bash
cd ~/DistLearn
git fetch origin
git checkout codex/decentr-my-own-runbook-20260425
git pull --ff-only origin codex/decentr-my-own-runbook-20260425
```

## 8. Как не менять системный Python глобально

Проект требует Python `>=3.12`.

На Ubuntu VPS системный `python3` часто равен `3.10`. Глобально менять его не надо.

Правильный путь:

- установить `pyenv`
- поставить `Python 3.12.9`
- сделать `pyenv local 3.12.9` только в папке `~/DistLearn`
- создать `.venv` именно этим интерпретатором

### 8.1 Установка pyenv на VPS

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential curl git make \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
```

Подключить в текущую сессию:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Сделать постоянным:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

### 8.2 Установка Python 3.12 только для `~/DistLearn`

```bash
cd ~/DistLearn
pyenv install 3.12.9
pyenv local 3.12.9
python --version
```

Ожидаемо:

```text
Python 3.12.9
```

### 8.3 Создание `.venv` строго на 3.12

```bash
cd ~/DistLearn
rm -rf .venv
~/.pyenv/versions/3.12.9/bin/python -m venv .venv
. .venv/bin/activate

python --version
which python
ls .venv/lib
```

Ожидаемо:

- `Python 3.12.9`
- `which python` указывает на `~/DistLearn/.venv/bin/python`
- в `.venv/lib` есть `python3.12`

## 9. Как ставить зависимости на VPS без переполнения диска

На follower-ноды не нужен CUDA build PyTorch. Для Linux CPU VPS лучше ставить CPU-only wheels и без `pip` cache.

```bash
cd ~/DistLearn
. .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir \
  "pydantic>=2.12,<3" \
  "PyYAML>=6.0,<7" \
  "protobuf>=5,<7" \
  "grpcio>=1.76,<2"

python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.9.1" \
  "torchvision==0.24.1"

python -m pip install --no-cache-dir -e ./Decentr_my_own --no-deps
```

Проверка:

```bash
python -m decentr_my_own.cli --help >/dev/null && echo CLI_OK
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
```

## 10. Как создать приватные WAN-конфиги

### 10.1 На Mac

```bash
mkdir -p ~/decentr-run
```

Создать `cluster.wan-3node.async.yaml`:

```bash
cat > ~/decentr-run/cluster.wan-3node.async.yaml <<'EOF'
cluster_name: decentr-wan-3node-async
transport: grpc
overlay_network: tailscale
tls_enabled: false
bootstrap_node_id: node-mac
nodes:
  - id: node-mac
    host: 100.110.148.103
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
    host: 100.114.132.57
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
    host: 100.84.195.80
    bind_host: 0.0.0.0
    port: 55051
    platform: linux
    neighbors: [node-mac, node-vps1]
    weight: 1.0
    resources:
      cpu_cores: 4
      accelerator: cpu
      relative_speed: 1.0
EOF
```

Создать `training.wan-async.micro.yaml`:

```bash
cat > ~/decentr-run/training.wan-async.micro.yaml <<'EOF'
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
  num_workers: 0
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
EOF
```

Проверить:

```bash
ls -l ~/decentr-run
sed -n '1,220p' ~/decentr-run/cluster.wan-3node.async.yaml
sed -n '1,260p' ~/decentr-run/training.wan-async.micro.yaml
```

### 10.2 Скопировать конфиги на VPS

На `Mac`:

```bash
ssh root@188.225.26.247 'mkdir -p /root/decentr-run'
ssh root@37.220.84.48 'mkdir -p /root/decentr-run'

scp ~/decentr-run/cluster.wan-3node.async.yaml \
    ~/decentr-run/training.wan-async.micro.yaml \
    root@188.225.26.247:/root/decentr-run/

scp ~/decentr-run/cluster.wan-3node.async.yaml \
    ~/decentr-run/training.wan-async.micro.yaml \
    root@37.220.84.48:/root/decentr-run/
```

## 11. Как проверить конфиги

### 11.1 Mac

На `Mac` проект можно запускать либо после `pip install -e`, либо через `PYTHONPATH`.

Практически удобнее так:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"
```

Проверка:

```bash
python -m decentr_my_own.cli --help >/dev/null && echo CLI_OK
```

Validate:

```bash
python -m decentr_my_own.cli validate-config \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac
```

### 11.2 VPS1

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate

python -m decentr_my_own.cli validate-config \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps1
```

### 11.3 VPS2

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate

python -m decentr_my_own.cli validate-config \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps2
```

## 12. Как проверить сеть перед запуском

### Mac

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"

python -m decentr_my_own.cli wan-preflight \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --check-dns
```

### VPS1

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate

python -m decentr_my_own.cli wan-preflight \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --check-dns
```

### VPS2

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate

python -m decentr_my_own.cli wan-preflight \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --check-dns
```

Если `wan-preflight` не проходит, дальше не идти.

## 13. Как строить shard-ы

`build-shards` выполняется только на `Mac`, только на bootstrap.

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"

python -m decentr_my_own.cli build-shards \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --force
```

Followers ничего не строят. Они сами:

- скачивают manifest
- тянут shard-ы по gRPC у bootstrap

## 14. Как запускать обучение

У всех трёх нод должен быть одинаковый `RUN_ID`.

Пример:

```bash
export RUN_ID=wan-async-3node-20260425
```

### 14.1 Порядок запуска

Правильный порядок:

1. `Mac`
2. `VPS1`
3. `VPS2`

Причина: bootstrap должен уже слушать gRPC и раздавать manifest/lease plans.

### 14.2 Mac

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli run-async-node \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

### 14.3 VPS1

Рекомендуется через `tmux`.

```bash
tmux new -s async-vps1
```

Внутри `tmux`:

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli run-async-node \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

Отцепиться:

- `Ctrl-b`, потом `d`

### 14.4 VPS2

```bash
tmux new -s async-vps2
```

Внутри `tmux`:

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli run-async-node \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

Отцепиться:

- `Ctrl-b`, потом `d`

## 15. Как смотреть живые логи

### 15.1 На Mac

Самый удобный путь:

Терминал 1:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli run-async-node \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60 \
  > /tmp/node-mac.async.log 2>&1
```

Терминал 2:

```bash
tail -f /tmp/node-mac.async.log
```

Если нужен только хвост:

```bash
tail -n 100 /tmp/node-mac.async.log
```

### 15.2 На VPS

Через `tmux`:

```bash
tmux ls
tmux attach -t async-vps1
tmux attach -t async-vps2
```

## 16. Как смотреть промежуточные реальные метрики

Во время выполнения полезно смотреть:

- лог stdout/stderr
- появление файлов в `artifacts`
- итоговые summary/round metrics

На `Mac`:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
export RUN_ID=wan-async-3node-20260425

while true; do
  clear
  date
  find Decentr_my_own/artifacts -type f | rg "$RUN_ID|async_round_metrics|scheduler_history|summary" | sort
  sleep 2
done
```

## 17. Как проверить, что соседи видят друг друга

На `Mac`:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"

python -m decentr_my_own.cli probe-neighbors \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --include-state
```

## 18. Как собирать логи после прогона

На `Mac`:

```bash
export VPS1_SSH=root@188.225.26.247
export VPS2_SSH=root@37.220.84.48
export RUN_ID=wan-async-3node-20260425

mkdir -p /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/artifacts/logs/"$RUN_ID"

rsync -avz "$VPS1_SSH":~/DistLearn/Decentr_my_own/artifacts/logs/"$RUN_ID"/ \
  /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/artifacts/logs/"$RUN_ID"/

rsync -avz "$VPS2_SSH":~/DistLearn/Decentr_my_own/artifacts/logs/"$RUN_ID"/ \
  /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/artifacts/logs/"$RUN_ID"/
```

## 19. Как строить итоговый отчёт

На `Mac`:

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli report-run \
  --log-root Decentr_my_own/artifacts/logs \
  --run-id "$RUN_ID"
```

## 20. Как обновлять код после нового push

На VPS:

```bash
cd ~/DistLearn
git fetch origin
git checkout codex/decentr-my-own-runbook-20260425
git pull --ff-only origin codex/decentr-my-own-runbook-20260425
```

Потом, если изменения затронули Python-код:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate

python -m pip install --no-cache-dir -e ./Decentr_my_own --no-deps
```

## 21. Типовые проблемы и что делать

### 21.1 `ModuleNotFoundError: No module named 'decentr_my_own'` на Mac

Ты запускаешь из корня `final_proj`, а пакет не установлен в текущий `.venv`.

Быстрый workaround:

```bash
export PYTHONPATH="$PWD/Decentr_my_own"
```

Более правильный путь:

```bash
python -m pip install -e ./Decentr_my_own
```

### 21.2 `Package 'decentr-my-own' requires a different Python: 3.10 ... not in '>=3.12'`

Нужно использовать `pyenv local 3.12.9` и пересоздать `.venv`.

### 21.3 `No space left on device`

Нужно:

- удалить старый `.venv`
- почистить `~/.cache/pip`
- ставить CPU-only `torch/torchvision`
- использовать `--no-cache-dir`

### 21.4 gRPC `fork()` warnings на Mac

Если видишь:

- `Other threads are currently calling into gRPC, skipping fork() handlers`
- `FD from fork parent still in poll list`

Причина: `DataLoader` workers конфликтуют с `gRPC`.

Решение: в training config держать

```yaml
dataset:
  num_workers: 0
```

### 21.5 `bincount only supports 1-d non-negative integral inputs`

Это был баг в старой версии метрик. Он уже исправлен в ветке:

```text
codex/decentr-my-own-runbook-20260425
```

Если видишь его на VPS, значит там не подтянут свежий код:

```bash
cd ~/DistLearn
git fetch origin
git checkout codex/decentr-my-own-runbook-20260425
git pull --ff-only origin codex/decentr-my-own-runbook-20260425
```

### 21.6 Followers не получают shard-ы

Проверить:

- bootstrap реально запущен
- `build-shards` выполнен на `Mac`
- `wan-preflight` проходит
- в `cluster.yaml` стоят `Tailscale IP`, а не public IP

## 22. Что нельзя делать

- не использовать `public IP` в `cluster.yaml`
- не запускать `build-shards` на VPS
- не давать разный `RUN_ID` на разных нодах
- не использовать `num_workers > 0` на `Mac` в этом gRPC-сценарии
- не понижать `requires-python` до `3.10`

## 23. Минимальный порядок действий без объяснений

### Mac

```bash
cd /Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj
. .venv/bin/activate
export PYTHONPATH="$PWD/Decentr_my_own"
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli wan-preflight \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --check-dns

python -m decentr_my_own.cli build-shards \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --force

python -m decentr_my_own.cli run-async-node \
  --cluster ~/decentr-run/cluster.wan-3node.async.yaml \
  --training ~/decentr-run/training.wan-async.micro.yaml \
  --self-node node-mac \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

### VPS1

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli wan-preflight \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --check-dns

python -m decentr_my_own.cli run-async-node \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps1 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

### VPS2

```bash
cd ~/DistLearn
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv local 3.12.9
. .venv/bin/activate
export RUN_ID=wan-async-3node-20260425

python -m decentr_my_own.cli wan-preflight \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --check-dns

python -m decentr_my_own.cli run-async-node \
  --cluster /root/decentr-run/cluster.wan-3node.async.yaml \
  --training /root/decentr-run/training.wan-async.micro.yaml \
  --self-node node-vps2 \
  --rounds 10 \
  --run-name "$RUN_ID" \
  --transport-timeout-s 60
```

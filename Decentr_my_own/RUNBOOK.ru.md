# Runbook: Mac + 2 VPS через Tailscale

Сценарий: `async` + `adaptive` + `heterogeneous`, replicated data mode.  
Ноды: Mac (bootstrap), VPS1, VPS2.

---

## Быстрый старт (3 шага)

### Шаг 0. На любой машине — получить команды

```bash
decentr-my-own run \
  --inventory configs/inventory.wan-tailscale.example.yaml \
  --run-name demo-001 \
  --epochs 50
```

Вывод — plaintext с точными командами для каждой ноды. Скопируй и запусти каждую на своей машине.

Если нужно запустить только на части нод:

```bash
decentr-my-own run \
  --inventory configs/inventory.wan-tailscale.example.yaml \
  --run-name demo-001 \
  --epochs 50 \
  --nodes mac,vps1
```

---

### Шаг 1. На Mac (bootstrap)

```bash
decentr-my-own start-run \
  --inventory configs/inventory.wan-tailscale.example.yaml \
  --self-node mac \
  --run-name demo-001 \
  --epochs 50 \
  --bind-host 0.0.0.0
```

Bootstrap запускает gRPC-сервер (порт 50051) и HTTP config-сервер (порт 50052).  
Готов принимать followers.

---

### Шаг 2. На VPS1 и VPS2 (followers)

```bash
# На VPS1:
decentr-my-own join-run \
  --bootstrap 100.64.0.1:50052 \
  --self-node vps1 \
  --bind-host 0.0.0.0

# На VPS2:
decentr-my-own join-run \
  --bootstrap 100.64.0.1:50052 \
  --self-node vps2 \
  --bind-host 0.0.0.0
```

`100.64.0.1` — Tailscale IP bootstrap-ноды (Mac).  
Порт `50052 = grpc_port + 1` — HTTP config-сервер bootstrap.  
Follower автоматически скачивает конфигурацию с bootstrap и начинает обучение.

---

### Шаг 3. Отчёт после завершения

```bash
decentr-my-own report-run \
  --log-root ./artifacts/logs \
  --run-id demo-001
```

---

## Inventory YAML (один файл на управляющей машине)

```yaml
# configs/my-cluster.yaml
cluster:
  name: decentr-lab
  overlay: tailscale   # или none
  bootstrap_node: mac

nodes:
  mac:
    host: 100.64.0.1   # Tailscale IP
    port: 50051
    bind_host: 0.0.0.0
    platform: macos
    weight: 1.0
    resources:
      cpu_cores: 8
      accelerator: mps
      relative_speed: 1.0
  vps1:
    host: 100.64.0.2
    port: 50051
    bind_host: 0.0.0.0
    platform: linux
    weight: 0.4
    resources:
      cpu_cores: 2
      accelerator: cpu
      relative_speed: 0.35
  vps2:
    host: 100.64.0.3
    port: 50051
    bind_host: 0.0.0.0
    platform: linux
    weight: 0.4
    resources:
      cpu_cores: 2
      accelerator: cpu
      relative_speed: 0.35

training:
  mode: async
  epochs: 50
  batch_size: 32
  lr: 0.03
  scheduler_mode: adaptive   # или static
  storage_mode: replicated   # или micro_shards
  partitioning: heterogeneous
```

`relative_speed` используется adaptive planner для первоначального распределения данных.  
После нескольких окон плоскость управления корректирует распределение по фактической пропускной способности через EMA.

---

## Как работает система

### Данные

- **replicated** (рекомендуется для WAN): каждая нода видит весь датасет, тренируется на своей партиции
- **micro_shards + adaptive**: датасет делится на шарды, адаптивный планировщик распределяет шарды по нодам в зависимости от их пропускной способности

### Веса

- Каждые `push_interval_steps` батчей нода пушит текущие веса соседям (gRPC payload)
- Нода принимает веса от соседей и делает weighted mix (сталость-взвешенный)
- `push_fanout=1` в WAN-режиме уменьшает трафик; `max_staleness=4` допускает небольшое отставание

### Эпохи

- `--epochs N` = ровно N глобальных проходов по train data
- В adaptive micro_shards одна эпоха может состоять из нескольких окон (windows) — это нормально
- Каждое окно = один блок шардов для одного прохода внутри эпохи

### Control plane

- Bootstrap нода держит `AdaptiveLeasePlanner` и `ControlPlaneStateStore`
- Followers запрашивают назначение шардов у bootstrap через gRPC
- Bootstrap не ждёт всех followers перед планированием следующего окна (нет hard barrier)
- Throughput reports обновляют EMA capacity каждого узла

### Config server (порт grpc_port+1)

- `start-run` запускает HTTP-сервер на `grpc_port + 1` параллельно с gRPC
- `GET /run-config` возвращает JSON с cluster, training, run_name, epochs
- `join-run --bootstrap HOST:CONFIG_PORT` скачивает конфиг и стартует обучение
- Токены не нужны — достаточно адреса bootstrap и своего node_id

### Завершение

- После завершения всех эпох нода отправляет `RunCompletionRecord` на bootstrap
- Bootstrap ждёт репортов от всех нод (timeout = `transport_timeout_s * node_count`)
- Сервер остаётся активным ещё `shutdown_grace_s` секунд, чтобы медленные followers успели отчитаться
- По умолчанию `--shutdown-grace-s 15` — достаточно для WAN

---

## WAN-специфика

| Параметр | Рекомендация |
|---|---|
| `--transport-timeout-s` | 30s для слабых VPS |
| `--shutdown-grace-s` | 15-30s |
| `push_fanout` | 1 (меньше трафика) |
| `push_interval_steps` | 25-50 (реже пушить) |
| `storage_mode` | `replicated` как default, micro_shards опционально |
| `scheduler_mode` | `adaptive` для heterogeneous нод |

### Firewall / Tailscale

Каждая нода должна принимать входящие TCP-соединения на два порта:
- `50051` — gRPC (обмен весами, control plane)
- `50052` — HTTP config server (только bootstrap, только в период старта)

---

## Диагностика

### Получить список команд для запуска

```bash
decentr-my-own run --inventory my-cluster.yaml --run-name demo-001 --epochs 50
```

### Проверка связности

```bash
decentr-my-own probe-neighbors \
  --cluster-b64 <...> --training-b64 <...> \
  --self-node vps1 --include-state
```

### Просмотр состояния удалённой ноды

```bash
decentr-my-own remote-state --target 100.64.0.1:50051
```

### Ping

```bash
decentr-my-own ping-remote --target 100.64.0.1:50051 --sender-node vps1
```

---

## Типовые ошибки

### "Could not fetch run config from ..."
- Проверь, что `start-run` на bootstrap уже запустился и вывел "Config server on port ..."
- Проверь firewall (inbound TCP на порт 50052 на bootstrap-машине)
- Увеличь `--transport-timeout-s`

### "timed out waiting for neighbors"
- Проверь, что bootstrap стартовал раньше followers
- Увеличь `--transport-timeout-s`

### "0 successful pushes"
- Проверь firewall (inbound TCP на порт 50051)
- Проверь Tailscale (`tailscale status`)

### "missing completion nodes" в report
- Нормально если некоторые followers были очень медленными
- Увеличь `--shutdown-grace-s` на bootstrap

### Follower не нашёл manifest
- В `micro_shards` режиме: сначала запусти `build_shards_command` на bootstrap
- Follower загружает manifest автоматически с bootstrap при старте

---

## Рекомендуемые режимы

| Режим | Когда использовать |
|---|---|
| `async` + `replicated` + `adaptive` | WAN с разными нодами (default, рекомендуется) |
| `async` + `micro_shards` + `adaptive` | Большой датасет, ограниченная RAM |
| `sync` | Только в надёжных сетях (LAN), иначе барьер блокирует медленные ноды |

---

## Запуск тестов

```bash
# Быстрые тесты (не требуют сети):
cd Decentr_my_own
.venv/bin/pytest tests/ --ignore=tests/test_async_gossip.py --ignore=tests/test_integration_distributed.py -q

# Smoke тест распределённого обучения (~4 мин):
.venv/bin/pytest tests/test_async_gossip.py -q --timeout=300

# Интеграционные тесты (epoch semantics, participation, completion):
.venv/bin/pytest tests/test_integration_distributed.py -q --timeout=300

# UX тесты create-run / join-run:
.venv/bin/pytest tests/test_create_run_ux.py -q
```

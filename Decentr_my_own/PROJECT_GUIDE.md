# Decentr_my_own: Полный Гайд

## 1. Что Это За Проект

`Decentr_my_own` это новая реализация децентрализованного распределенного обучения нейросети.

Цели проекта:

- запуск на нескольких машинах, в том числе вне одной локальной сети
- поддержка `macOS + Windows`
- фиксированный граф соседей
- `data parallel` обучение
- два режима синхронизации: `sync` и `async`
- минимальный разрыв с текущим проектом по базовым идеям: `rank/world_size`-логика, sampler, train loop, checkpoint, метрики

Текущее сетевое основание:

- `PyTorch` для локального обучения
- `gRPC + protobuf` для обмена между нодами
- `Tailscale` как рекомендуемый overlay для WAN-запуска

## 2. Что Сделано По Этапам

### Этап 1. Каркас проекта

Сделано:

- отдельный пакет `decentr_my_own`
- типизированные YAML-конфиги через `Pydantic`
- базовый CLI

Основные файлы:

- `decentr_my_own/config/models.py`
- `decentr_my_own/config/loader.py`
- `decentr_my_own/cli.py`

### Этап 2. Локальный training pipeline

Сделано:

- фабрика моделей
- загрузка данных
- local train/eval loop
- checkpoint и summary

Основные файлы:

- `decentr_my_own/models/factory.py`
- `decentr_my_own/data/loaders.py`
- `decentr_my_own/training/engine.py`

### Этап 3. Data partitioning и shuffling

Сделано:

- deterministic partitioning
- `homogeneous` и `heterogeneous` режимы
- reproducible shuffle по `seed + epoch/round`

Основной файл:

- `decentr_my_own/data/partitioning.py`

### Этап 4. Peer-to-peer transport

Сделано:

- `gRPC` peer server/client
- `protobuf` API
- сериализация payload с тензорами

Основные файлы:

- `decentr_my_own/comm/proto/peer.proto`
- `decentr_my_own/comm/server.py`
- `decentr_my_own/comm/client.py`
- `decentr_my_own/comm/serialization.py`
- `decentr_my_own/comm/state.py`

### Этап 5. Sync режим

Сделано:

- barrier по раунду
- обмен весами/дельтами
- weighted averaging

Основной файл:

- `decentr_my_own/algorithms/sync_barrier.py`

### Этап 6. Async режим

Сделано:

- async gossip-подобный обмен
- mixing по последним payload
- учет `staleness`

Основной файл:

- `decentr_my_own/algorithms/async_gossip.py`

### Этап 7. WAN-ready слой

Сделано:

- разделение `host` и `bind_host`
- `wan-preflight`
- `launch-plan`
- `probe-neighbors`
- example configs для `mac + windows`

Основные файлы:

- `decentr_my_own/deployment/wan.py`
- `configs/cluster.wan-2node.example.yaml`
- `configs/training.wan-sync.example.yaml`
- `configs/training.wan-async.example.yaml`

### Этап 8. Observability и reporting

Сделано:

- unified telemetry в `local/sync/async`
- run-level aggregation
- сравнение `sync vs async`

Основной файл:

- `decentr_my_own/metrics/reporting.py`

## 3. Архитектура

Каждая нода логически состоит из нескольких частей:

- `trainer`: локальный train/eval loop
- `data layer`: partitioning, shuffle, sampler
- `peer server`: принимает payload от соседей
- `peer client`: отправляет payload соседям
- `mixing engine`: sync или async логика обновления модели
- `config layer`: cluster/training config
- `metrics layer`: summary, metrics files, aggregated reports

Поток работы:

1. Нода читает `cluster.yaml` и `training.yaml`.
2. Поднимает локальный `PeerServer`.
3. Проверяет доступность соседей.
4. Строит `DataLoader` только для своей доли train-данных.
5. Делает локальное обучение.
6. В зависимости от режима:
   - `sync`: ждет соседей на barrier и усредняет состояние
   - `async`: принимает последние доступные peer-update и смешивает без глобального барьера
7. Сохраняет метрики, summary и checkpoints.

## 4. Как Работает Дата-Параллелизм

Базовая стратегия:

- датасет локально лежит на каждой машине
- по сети не передаются сами sample
- между нодами делятся только индексы и модельные обновления

Partitioning:

- `homogeneous`: всем нодам примерно равная доля
- `heterogeneous`: доля зависит от `node.weight * node.resources.relative_speed`

Shuffle:

- детерминированный
- управляется `seed`
- меняется по `epoch` или `round`

Это позволяет:

- воспроизводимость
- отсутствие пересечений в train subset между нодами
- адаптацию под быстрые и медленные машины

## 5. Как Работает Sync Режим

Файл:

- `decentr_my_own/algorithms/sync_barrier.py`

Идея:

1. Все ноды стартуют с одинакового seed.
2. Каждая нода делает локальные шаги.
3. Нода собирает payload:
   - либо веса
   - либо delta относительно исходного состояния раунда
4. Payload отправляется соседям.
5. Нода ждет barrier: должны прийти payload от всех соседей для текущего `payload_id`.
6. Состояние модели усредняется с учетом `sample_count`.

Важно:

- в `sync` режиме финальные состояния у нод должны совпадать
- это проверяется тестом

## 6. Как Работает Async Режим

Файл:

- `decentr_my_own/algorithms/async_gossip.py`

Идея:

1. Нода делает локальные шаги.
2. Периодически отправляет соседям `async_weights`.
3. Получает последние доступные peer payload.
4. Для каждого update считает `staleness`.
5. Смешивает локальное состояние с peer-состоянием через `alpha`, зависящий от:
   - `mixing_alpha`
   - `sample_count`
   - `staleness`

Важно:

- глобального барьера нет
- одинаковый финальный digest у всех нод не ожидается
- async устойчивее к медленным нодам, но может быть менее стабильным

## 7. WAN / Tailscale

Файл:

- `decentr_my_own/deployment/wan.py`

Ключевые поля конфига:

- `host`: адрес, по которому другие ноды находят эту машину
- `bind_host`: адрес, на котором локально слушает сервер

Для реального multi-host запуска рекомендуется:

- `host = *.tailnet.ts.net` или `100.x.x.x`
- `bind_host = 0.0.0.0`

Основные команды:

- `wan-preflight`: валидирует готовность конфига к WAN-запуску
- `launch-plan`: строит готовые команды запуска для каждой машины
- `probe-neighbors`: проверяет достижимость соседей после старта

## 8. Где Что Лежит

### Конфиги

- `configs/cluster.example.yaml`
- `configs/training.example.yaml`
- `configs/training.local-smoke.yaml`
- `configs/cluster.wan-2node.example.yaml`
- `configs/training.wan-sync.example.yaml`
- `configs/training.wan-async.example.yaml`

### Конфиг-слой

- `decentr_my_own/config/models.py`
  - описывает схемы конфигов
- `decentr_my_own/config/loader.py`
  - загружает YAML и строит `ResolvedConfig`

### Модели

- `decentr_my_own/models/factory.py`
  - создает `resnet18/resnet34/resnet50`
  - поддерживает `groupnorm/batchnorm/frozen_batchnorm`

### Data layer

- `decentr_my_own/data/loaders.py`
  - строит `DataLoader`
  - подключает sampler и partitioning
- `decentr_my_own/data/partitioning.py`
  - deterministic planning и sampling

### Network layer

- `decentr_my_own/comm/server.py`
  - gRPC peer server
- `decentr_my_own/comm/client.py`
  - gRPC peer client
- `decentr_my_own/comm/state.py`
  - хранение последних и history payload
- `decentr_my_own/comm/serialization.py`
  - преобразование payload <-> gRPC messages
- `decentr_my_own/comm/messages.py`
  - внутренние dataclass/структуры сообщений

### Training layer

- `decentr_my_own/training/engine.py`
  - local training pipeline
- `decentr_my_own/training/state_ops.py`
  - операции над `state_dict`
- `decentr_my_own/training/device.py`
  - выбор девайса
- `decentr_my_own/training/seed.py`
  - общий seed
- `decentr_my_own/training/io.py`
  - directories, metrics writing, summary writing

### Алгоритмы

- `decentr_my_own/algorithms/sync_barrier.py`
  - sync training
- `decentr_my_own/algorithms/async_gossip.py`
  - async training

### WAN deployment

- `decentr_my_own/deployment/wan.py`

### Reporting

- `decentr_my_own/metrics/reporting.py`

### CLI

- `decentr_my_own/cli.py`

### Тесты

- `tests/test_config_loading.py`
- `tests/test_cli.py`
- `tests/test_local_training.py`
- `tests/test_partitioning.py`
- `tests/test_comm_serialization.py`
- `tests/test_comm_transport.py`
- `tests/test_sync_barrier.py`
- `tests/test_async_gossip.py`
- `tests/test_wan_tools.py`
- `tests/test_metrics_reporting.py`

## 9. Ключевые Функции И Для Чего Они Нужны

### Конфиги

- `load_cluster_config(...)`
  - загружает cluster config
- `load_training_config(...)`
  - загружает training config
- `load_resolved_config(...)`
  - связывает cluster/training/self-node в единый runtime config

### Local training

- `run_local_training(...)`
  - запускает обучение на одной ноде без сети

### Sync training

- `run_sync_worker(...)`
  - запускает одну sync-ноду
- `run_sync_smoke(...)`
  - локальный multi-process smoke test sync-режима

### Async training

- `run_async_worker(...)`
  - запускает одну async-ноду
- `run_async_smoke(...)`
  - локальный multi-process smoke test async-режима

### WAN tools

- `build_wan_preflight(...)`
  - собирает preflight report
- `build_launch_plan(...)`
  - строит platform-aware команды запуска
- `probe_neighbors(...)`
  - пробует достучаться до соседей

### Reporting

- `build_run_report(...)`
  - собирает один run-report по папке логов
- `compare_run_reports(...)`
  - сравнивает два готовых run-report
- `build_smoke_comparison(...)`
  - быстро сравнивает sync и async на short smoke run

## 10. Как Запускать

### 10.1. Посмотреть все команды

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli --help
```

### 10.2. Проверить конфиги

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli validate-config \
  --cluster Decentr_my_own/configs/cluster.example.yaml \
  --training Decentr_my_own/configs/training.example.yaml \
  --self-node node-1
```

### 10.3. Локальное обучение одной ноды

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli local-train \
  --cluster Decentr_my_own/configs/cluster.example.yaml \
  --training Decentr_my_own/configs/training.local-smoke.yaml \
  --self-node node-1 \
  --run-name local-demo \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1
```

### 10.4. Transport smoke

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli peer-smoke --peer-count 3
```

### 10.5. Sync smoke

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli sync-smoke --peer-count 3 --rounds 1
```

### 10.6. Async smoke

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli async-smoke --peer-count 3 --rounds 2
```

### 10.7. WAN preflight

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli wan-preflight \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac
```

### 10.8. Launch plan для двух машин

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli launch-plan \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml
```

### 10.9. Запуск одной sync-ноды

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli run-sync-node \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac \
  --rounds 20
```

### 10.10. Запуск одной async-ноды

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-async.example.yaml \
  --self-node node-mac \
  --rounds 20
```

### 10.11. Проверка соседей

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli probe-neighbors \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac \
  --include-state
```

### 10.12. Отчет по run

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli report-run \
  --log-root Decentr_my_own/artifacts/logs \
  --run-id local-demo
```

### 10.13. Сравнение двух run

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli compare-runs \
  --log-root Decentr_my_own/artifacts/logs \
  --baseline-run run-a \
  --candidate-run run-b
```

### 10.14. Быстрое сравнение sync vs async

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli compare-smoke \
  --peer-count 2 \
  --sync-rounds 1 \
  --async-rounds 1
```

## 11. Куда Пишутся Результаты

Логи:

- `artifacts/logs/<run_id>/<node_id>/...`

Чекпоинты:

- `artifacts/checkpoints/<run_id>/<node_id>/...`

Типичные файлы:

- `run_summary.json`
- `sync_run_summary.json`
- `async_run_summary.json`
- `epoch_metrics.csv`
- `sync_round_metrics.csv`
- `async_round_metrics.csv`
- `epoch-001.pt`
- `round-001.pt`
- `async-round-001.pt`

## 12. Что Уже Проверено Тестами

Покрыто:

- загрузка конфигов
- CLI
- local training
- deterministic partitioning
- payload serialization
- transport ping/push/state
- sync convergence
- async smoke
- WAN tools
- metrics/reporting

## 13. Ограничения Текущей Версии

- пока нет dynamic membership
- `TLS/mTLS` не включен
- рекомендуется `Tailscale` для WAN-запуска
- optimizer state между нодами не синхронизируется
- основной smoke работает на `FakeData`, а не на полном реальном эксперименте
- async-алгоритм пока базовый, без более сложных методов компенсации staleness

## 14. Что Я Бы Делал Дальше

Практические следующие шаги:

1. Подготовить реальные конфиги под твои две машины.
2. Прогнать `wan-preflight` и `probe-neighbors`.
3. Сделать первый реальный `sync` run на двух машинах.
4. Затем сделать такой же `async` run.
5. Сравнить через `report-run` и `compare-runs`.


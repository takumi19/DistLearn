# Decentr_my_own: Запуск На Нескольких Компьютерах

## Что Это Такое

`Decentr_my_own` это децентрализованная система распределенного обучения.

Каждая нода:

- локально обучает модель на своей части данных
- поднимает свой `gRPC` сервер
- отправляет обновления соседям
- получает обновления от соседей
- в зависимости от режима работает либо через `sync`, либо через `async`

Проект не использует центральный parameter server как основную точку координации. Вместо этого у нас фиксированный граф соседей, и ноды общаются напрямую.

Сейчас лучше всего запускать систему через `Tailscale`, чтобы машины видели друг друга даже вне одной локальной сети.

## Когда Использовать `sync`, А Когда `async`

`sync`:

- проще для отладки
- все ноды синхронизируются по раундам
- финальное состояние моделей совпадает
- лучше для первого реального запуска

`async`:

- не требует глобального ожидания всех нод
- лучше переносит медленные машины
- состояния моделей могут отличаться
- требует больше экспериментов с `mixing_alpha` и `max_staleness`

Практически:

- первый реальный multi-host запуск лучше делать в `sync`
- после этого делать второй запуск в `async` и сравнивать результаты

## Что Нужно Перед Запуском

На каждой машине должно быть:

- одна и та же версия проекта
- Python и зависимости проекта
- доступ к одному и тому же датасету или локальная копия датасета
- установленный `Tailscale`
- все машины должны быть в одном `tailnet`

Минимально нужна одна папка проекта на каждой машине:

- macOS: `/path/to/final_proj`
- Windows: `C:\path\to\final_proj`

Важно:

- пути на macOS и Windows могут отличаться
- но содержимое `Decentr_my_own/configs/*.yaml` должно быть логически одинаковым

## Рекомендуемая Схема Для 2 Машин

Пример:

- `node-mac`: MacBook
- `node-win`: Windows-машина

Готовый пример конфига уже есть:

- [cluster.wan-2node.example.yaml](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/configs/cluster.wan-2node.example.yaml)
- [training.wan-sync.example.yaml](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/configs/training.wan-sync.example.yaml)
- [training.wan-async.example.yaml](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/configs/training.wan-async.example.yaml)

## Как Настроить Конфиг

Открой [cluster.wan-2node.example.yaml](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/configs/cluster.wan-2node.example.yaml) и замени:

- `host`
  - на реальный `MagicDNS` hostname машины в Tailscale
  - либо на `100.x.x.x` Tailscale IP
- `bind_host`
  - оставь `0.0.0.0`
- `port`
  - оставь одинаковый, например `50051`, если порт свободен

Пример:

```yaml
nodes:
  - id: node-mac
    host: my-mac.tailnet.ts.net
    bind_host: 0.0.0.0
    port: 50051
    platform: macos
    neighbors: [node-win]
  - id: node-win
    host: my-win.tailnet.ts.net
    bind_host: 0.0.0.0
    port: 50051
    platform: windows
    neighbors: [node-mac]
```

Что значат поля:

- `host`: адрес, по которому другие ноды стучатся к этой машине
- `bind_host`: адрес, на котором локально слушает сервер
- `neighbors`: список соседей в фиксированном графе
- `weight` и `resources.relative_speed`: влияют на `heterogeneous` partitioning данных

## Шаг 1. Проверить Конфиг Перед Запуском

На Mac:

```bash
cd /path/to/final_proj
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli wan-preflight \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac
```

На Windows PowerShell:

```powershell
cd C:\path\to\final_proj
$env:PYTHONPATH='Decentr_my_own'
python -m decentr_my_own.cli wan-preflight `
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml `
  --training Decentr_my_own/configs/training.wan-sync.example.yaml `
  --self-node node-win
```

Что должно быть в ответе:

- корректный `self_bind_target`
- корректный `self_advertise_target`
- список `neighbor_targets`
- отсутствие warning про loopback host

## Шаг 2. Получить Готовые Команды Запуска

На любой машине:

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli launch-plan \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml
```

Эта команда вернет:

- порядок старта нод
- отдельную команду для macOS
- отдельную команду для Windows
- команды `probe-neighbors`

## Шаг 3. Открыть Порт В Firewall

На каждой машине нужно разрешить входящий TCP на выбранный `port`, например `50051`.

Нужно проверить:

- macOS firewall
- Windows Defender Firewall

Без этого `probe-neighbors` и реальный training могут не пройти.

## Шаг 4. Запуск `sync` На Двух Машинах

### Mac

```bash
cd /path/to/final_proj
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli run-sync-node \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac \
  --rounds 20
```

### Windows PowerShell

```powershell
cd C:\path\to\final_proj
$env:PYTHONPATH='Decentr_my_own'
python -m decentr_my_own.cli run-sync-node `
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml `
  --training Decentr_my_own/configs/training.wan-sync.example.yaml `
  --self-node node-win `
  --rounds 20
```

Что будет происходить:

1. Каждая нода поднимет `PeerServer`.
2. Каждая нода проверит, что сосед доступен.
3. Каждая нода сделает локальные шаги обучения.
4. После каждого раунда ноды обменяются payload.
5. Ноды дождутся друг друга и усреднят состояние модели.

## Шаг 5. Проверить, Что Ноды Видят Друг Друга

После старта можно выполнить `probe-neighbors`.

### Mac

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli probe-neighbors \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-sync.example.yaml \
  --self-node node-mac \
  --include-state
```

### Windows PowerShell

```powershell
$env:PYTHONPATH='Decentr_my_own'
python -m decentr_my_own.cli probe-neighbors `
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml `
  --training Decentr_my_own/configs/training.wan-sync.example.yaml `
  --self-node node-win `
  --include-state
```

Если все хорошо:

- `reachable_count` будет равен числу соседей
- в `ping.message` будет `pong:...`

## Шаг 6. Где Смотреть Результаты

На каждой машине результаты пишутся локально:

- логи: `Decentr_my_own/artifacts/logs/<run_id>/<node_id>/`
- чекпоинты: `Decentr_my_own/artifacts/checkpoints/<run_id>/<node_id>/`

Для `sync` там будут:

- `sync_run_summary.json`
- `sync_round_metrics.csv`
- `round-001.pt`, `round-002.pt`, ...

Для `async` там будут:

- `async_run_summary.json`
- `async_round_metrics.csv`
- `async-round-001.pt`, ...

## Шаг 7. Построить Отчет По Запуску

После завершения запуска:

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli report-run \
  --log-root Decentr_my_own/artifacts/logs \
  --run-id <run_id>
```

Что покажет отчет:

- сколько нод было в запуске
- `final_test_accuracy_mean`
- `effective_samples_per_s_total`
- `run_duration_s_max`
- совпадает ли итоговое состояние у всех нод

Для `sync` режимов обычно ожидается:

- `consistent_final_state = true`

## Шаг 8. Запуск `async`

Для `async` запуска используется другой training config:

- [training.wan-async.example.yaml](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/configs/training.wan-async.example.yaml)

Команды почти те же.

### Mac

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli run-async-node \
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml \
  --training Decentr_my_own/configs/training.wan-async.example.yaml \
  --self-node node-mac \
  --rounds 20
```

### Windows PowerShell

```powershell
$env:PYTHONPATH='Decentr_my_own'
python -m decentr_my_own.cli run-async-node `
  --cluster Decentr_my_own/configs/cluster.wan-2node.example.yaml `
  --training Decentr_my_own/configs/training.wan-async.example.yaml `
  --self-node node-win `
  --rounds 20
```

Что важно для `async`:

- одинаковый финальный digest не обязателен
- лучше сравнивать не digest, а accuracy, throughput, duration

## Шаг 9. Сравнить `sync` И `async`

Если у тебя есть два реальных run-id, можно сравнить их:

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli compare-runs \
  --log-root Decentr_my_own/artifacts/logs \
  --baseline-run <sync_run_id> \
  --candidate-run <async_run_id>
```

Если нужен короткий synthetic smoke:

```bash
PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli compare-smoke \
  --peer-count 2 \
  --sync-rounds 1 \
  --async-rounds 1
```

## Практический Рекомендуемый Порядок

1. Внести реальные `host` в `cluster.wan-2node.example.yaml`.
2. Проверить `wan-preflight` на каждой машине.
3. Открыть `port` в firewall.
4. Запустить `sync` на двух машинах.
5. Проверить `probe-neighbors`.
6. Собрать `report-run`.
7. Потом повторить все то же для `async`.
8. Сравнить через `compare-runs`.

## Что Может Пойти Не Так

Если ноды не видят друг друга:

- проверь `host`
- проверь `bind_host`
- проверь firewall
- проверь, что обе машины в одном `Tailscale` tailnet
- проверь, что `port` совпадает с конфигом

Если `sync` висит:

- одна из нод не дошла до barrier
- один из peer payload не пришел
- есть проблема с сетью или firewall

Если `async` дает слабую accuracy:

- уменьшить `mixing_alpha`
- уменьшить `local_steps`
- уменьшить `max_staleness`
- сначала отладить конфиг на `sync`

## Где Смотреть Более Полное Описание

Общий технический гайд:

- [PROJECT_GUIDE.md](/Users/maxim/Documents/Study/HSE/msc/1st_year/final_proj/Decentr_my_own/PROJECT_GUIDE.md)

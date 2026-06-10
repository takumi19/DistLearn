# Команды кампании: CIFAR-10 WAN (Yandex Cloud)

Roadmap-aligned запуск. Фиксированная база:
`seed42, 50 epochs, batch32, lr0.03, push_interval10, fanout0, mixing_alpha0.2,
throughput_ema0.9, resnet18+groupnorm, micro_shards, heterogeneous, grad_clip1.0`

> **Автоматика (orchestrate.py).** После КАЖДОГО эксперимента оркестратор сам:
> 1. стягивает результаты в `./results/<run_id>/`,
> 2. **пушит их в git** (репозиторий `decentr-results`, ветка `main`),
> 3. **чистит ноды** (kill процессов + `rm` artifacts),
> 4. запускает следующий `pending` эксперимент.
>
> Отключить: `--no-git-push` и/или `--no-cleanup`.

---

## Текущий статус экспериментов

| # | Имя | Статус | Результат |
|---|---|---|---|
| 0 | smoke | pending | — (2 эпохи, инфра-чек) |
| 1 | async_adaptive_default | **pending** | — (reference — первым!) |
| 2 | async_static | pending | — |
| 3 | sync_static | pending | — |
| 4 | ema_tput_0_1 | ✅ **done** | 52.7% best (11/20 нод) |
| 5 | ema_tput_0_5 | pending | — |
| 6 | ema_tput_0_99 | pending | ⚠️ прошлый ран упал (crash ep3) |
| 7 | top_k_fastest | pending | — |
| 8 | top_k_reliable | pending | — |
| 9 | ping_aware | pending | — |
| 10 | reliability_aware_graph | pending | — |
| 11 | top_k_useful | pending | — |
| 12 | ring_topology | ✅ **done** | 54.3% best (20/20 нод) |
| 13 | sparse_expander | pending | — |
| 14 | star_topology | pending | — |
| 15 | low_comm_100 | pending | — |
| 16 | low_comm_300 | pending | — |
| 17 | low_comm_500 | pending | — |
| 18 | random_fanout_1 | pending | — |
| 19 | random_fanout_2 | pending | — |
| 20 | staleness_2 | pending | — |
| 21 | staleness_8 | pending | — |
| 22 | batch_16 | pending | — |
| 23 | batch_128 | pending | — |
| 24 | **dynamic_graph** | pending | 🆕 Phase 3 (код задеплоен) |
| 25 | **delta_exchange** | pending | 🆕 Phase 3 (код задеплоен) |
| 26 | **compress_float16** | pending | 🆕 Phase 3 (код задеплоен) |
| 27 | **compress_quant8** | pending | 🆕 Phase 3 (код задеплоен) |
| 28 | **compress_topk** | pending | 🆕 Phase 3 (код задеплоен) |

`single_node_baseline` — disabled (нужен local-train драйвер).

---

## Перед каждым запуском

```bash
cd /Users/maxim/dev/Decentr_my_own/cloud

# 1. Обновить токен (истекает каждые 12ч)
export YC_TOKEN=$(~/yandex-cloud/bin/yc iam create-token)

# 2. Проверить что ноды живые и готовые (cleanup уже делается оркестратором)
python3 orchestrate.py check --pool pool.yaml
```

> ⚠️ **Phase 3 код (dynamic_graph/delta/compress) надо задеплоить на ноды ОДИН раз**
> перед их запуском (на бегущий ран не влияет — модули грузятся в память при старте):
> ```bash
> ./deploy_patches.sh            # выложить патчи на все 20 нод
> ./deploy_patches.sh --verify   # проверить md5 совпадение
> ```

> ⚠️ Если IP бастиона изменился (после stop/start VM):
> ```bash
> NEW_IP=$(~/yandex-cloud/bin/yc compute instance list 2>/dev/null | awk '/decentr-01/{print $6}')
> sed -i '' "s/[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+/$NEW_IP/g" pool.yaml kill_cluster.sh nodes_table.sh
> echo "New bastion IP: $NEW_IP"
> ```

---

## Запуск всей оставшейся кампании (рекомендуется)

Оркестратор пропускает `done`, запускает по очереди все `pending`:

```bash
PYTHONUNBUFFERED=1 nohup python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --max-runtime-hours 9 > /tmp/campaign.log 2>&1 &

tail -f /tmp/campaign.log
```

---

## Запуск по одному (если нужно)

```bash
# Самый важный — запускать первым (reference для всего)
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_adaptive_default --max-runtime-hours 9

# Baselines
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_static --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only sync_static --max-runtime-hours 9

# EMA sweep (0.9 == async_adaptive_default, 0.1 уже done)
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ema_tput_0_5 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ema_tput_0_99 --max-runtime-hours 9
# ^^^ ema_tput_0_99 может снова упасть: bootstrap получает слишком много
# шардов из-за медленного EMA. Если упадёт — смотри раздел "Если ema_0_99 падает"

# Peer selection policies (Phase 2 код задеплоен)
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only top_k_fastest --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only top_k_reliable --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ping_aware --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only reliability_aware_graph --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only top_k_useful --max-runtime-hours 9

# Topology
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ring_topology --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only sparse_expander --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only star_topology --max-runtime-hours 9

# Communication frequency
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only low_comm_100 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only low_comm_300 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only low_comm_500 --max-runtime-hours 9

# Fanout
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only random_fanout_1 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only random_fanout_2 --max-runtime-hours 9

# Bonus
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only staleness_2 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only staleness_8 --max-runtime-hours 9

# ПОСЛЕДНИМИ — batch sweep
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only batch_16 --max-runtime-hours 9

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only batch_128 --max-runtime-hours 9
```

---

## Если ema_tput_0_99 снова падает

Причина: slow EMA → bootstrap получает 9000+ сэмплов/эпоху вместо 2048 → crash.

Вариант 1 — запустить как есть и принять частичные результаты (у нас уже есть 6 эпох).

Вариант 2 — ограничить максимум шардов на ноду перед запуском (нужно в коде).

---

## Утилиты

```bash
# Статус кампании
python3 orchestrate.py status --suite suites/cifar10_wan_campaign.yaml

# Таблица нод (что делает каждая)
./nodes_table.sh

# Лог текущего запуска
tail -f /tmp/campaign.log

# Сравнение всех завершённых
python3 orchestrate.py compare --results-dir ./results

# Сброс одного эксперимента для перезапуска
python3 orchestrate.py reset --suite suites/cifar10_wan_campaign.yaml --only <name>

# Собрать результаты вручную
python3 orchestrate.py collect --pool pool.yaml --run-id <run_id>
```

---

## Остановить ноды (удалить кластер)

```bash
export YC_TOKEN=$(~/yandex-cloud/bin/yc iam create-token)
cd provision/terraform-yc
terraform destroy -auto-approve
cd ../..
~/yandex-cloud/bin/yc compute instance list  # убедиться что всё удалено
```

---

## Время и деньги

| | Время | ~Стоимость |
|---|---|---|
| 1 эксперимент (50 эпох) | ~6-7 ч | ~400-500 ₽ |
| Оставшиеся 23 эксперимента | ~140-160 ч (~6 дней) | ~9200-11500 ₽ |
| Токен истекает | 12 часов | — |

**Обновлять токен каждые 12 часов:**
```bash
export YC_TOKEN=$(~/yandex-cloud/bin/yc iam create-token)
```

---

## Phase 3 эксперименты — РЕАЛИЗОВАНЫ ✅ (нужен `./deploy_patches.sh` перед запуском)

Код в `patches/async_gossip.py` + `config_models.py`, 8/8 unit-тестов
(`patches/test_phase3_compression.py`). Всё флаг-gated: дефолты = базовое поведение.

> **Перед первым запуском любого из них:** `./deploy_patches.sh` (выложить на ноды).

### dynamic_graph (Phase 12 roadmap)
**Что:** активный push-набор перестраивается каждые 5 окон по ping_aware score
(top-4 соседа). Получаем по-прежнему от всех. **min-degree floor** не даёт изолировать ноды.
**Что смотрим:** как меняется граф, network cost, не изолируются ли слабые ноды.
```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only dynamic_graph --max-runtime-hours 9
```

### delta_exchange (Phase 13 roadmap)
**Что:** вместо полных весов шлём дельту `current − last_sent`. Получатель хранит базу
на отправителя и реконструирует. Если все push упали → следующий раз полные веса (само-лечение).
**Что смотрим:** payload bytes, push latency, accuracy, численная стабильность.
```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only delta_exchange --max-runtime-hours 9
```

### compress_float16 / compress_quant8 / compress_topk (Phase 14 roadmap)
**Что:** сжатие payload перед отправкой.
- `float16` — fp16 cast (~2× меньше, минимальный риск)
- `quant8` — 8-bit min-max квантизация (~4× меньше, ошибка ≤ range/255)
- `topk_sparse` — top-10% по модулю + fp16 (~10×+ меньше, высокий риск accuracy)

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only compress_float16 --max-runtime-hours 9
# затем compress_quant8, compress_topk
```

---

## Ещё не реализовано (disabled)

### single_node_baseline
**Что:** одна нода, без gossip — абсолютный reference для качества и времени.
**Что нужно в коде:** драйвер `local-train` в `orchestrate.py` (~50 строк). Скажи — реализую.

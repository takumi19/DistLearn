# Runbook: Yandex Cloud — Decentr WAN Campaign

Инструкция по запуску кампании экспериментов на 20 нодах Yandex Cloud.

---

## Содержание

1. [Что потребуется](#1-что-потребуется)
2. [Первоначальная настройка (один раз)](#2-первоначальная-настройка-один-раз)
3. [Поднять ноды](#3-поднять-ноды)
4. [Дождаться bootstrap](#4-дождаться-bootstrap)
5. [Запустить тесты](#5-запустить-тесты)
6. [Собрать и сравнить результаты](#6-собрать-и-сравнить-результаты)
7. [СТОП — удалить все ноды срочно](#7-стоп--удалить-все-ноды-срочно)
8. [Стоимость и биллинг](#8-стоимость-и-биллинг)
9. [Что делать если что-то пошло не так](#9-что-делать-если-что-то-пошло-не-так)

---

## 1. Что потребуется

| Компонент | Статус |
|---|---|
| Аккаунт Yandex Cloud с платёжным аккаунтом | ✅ `distlearn` / folder `distlearnforall` |
| Yandex Cloud CLI (`yc`) | ✅ установлен |
| Terraform ≥ 1.3 | ✅ v1.12.0 |
| SSH ключ для нод | ✅ `~/.ssh/decentr_id_ed25519` |
| Deploy key для results repo | ✅ `~/.ssh/decentr_results_deploy_key` |
| secrets.env заполнен | ✅ |
| cloud-init отрендерен | ✅ `provision/terraform-yc/cloud-init.rendered.yaml` |

---

## 2. Первоначальная настройка (один раз)

Эти шаги уже выполнены. Повторять только если переустанавливаешь с нуля.

```bash
# Инициализировать yc CLI
yc init
# folder: distlearnforall (id = b1g90g9camntoutebg36)
# zone:   ru-central1-a

# Отрендерить cloud-init
cd /Users/maxim/dev/Decentr_my_own/cloud
python3 render_cloud_init.py \
  --template provision/terraform-yc/cloud-init.yc.yaml.template \
  --out provision/terraform-yc/cloud-init.rendered.yaml

# Проверить что deploy key для results repo добавлен в GitHub
ssh -i ~/.ssh/decentr_results_deploy_key -T git@github.com 2>&1
# Должно ответить: Hi maxtro91627/decentr-results! You've successfully authenticated...
```

---

## 3. Поднять ноды

```bash
# Получить IAM-токен (действует 12 часов)
export YC_TOKEN=$(yc iam create-token)

# Поднять 20 VM (~3-5 минут)
cd /Users/maxim/dev/Decentr_my_own/cloud/provision/terraform-yc
terraform apply
# Напечатать: yes
```

После успешного apply terraform автоматически создаст файл `cloud/pool.yaml` с IP-адресами всех нод.

Что создаётся (26 ресурсов):
- 1 VPC сеть + 1 подсеть (192.168.199.0/24)
- 1 NAT gateway (даёт интернет всем нодам для bootstrap)
- 1 security group (SSH снаружи только на bastion, gRPC внутри кластера)
- **20 VM** (4 vCPU / 8 GB / 40 GB SSD, Ubuntu 24.04)
  - `decentr-01` — bastion, единственный с публичным IP
  - `decentr-02..20` — приватные, доступны через ProxyJump
- 1 pool.yaml (локальный файл)

---

## 4. Дождаться bootstrap

После создания VM каждая нода запускает `bootstrap.sh` в фоне.  
Bootstrap: apt install → git clone проекта → pip install → скачать CIFAR-10.  
**Занимает ~8-15 минут.**

```bash
cd /Users/maxim/dev/Decentr_my_own/cloud

# Проверить статус всех нод
python3 orchestrate.py check --pool pool.yaml
```

Вывод когда все готовы:
```
  [✓] decentr-01    192.168.199.x  —  ready
  [✓] decentr-02    192.168.199.x  —  ready
  ...
  [✓] decentr-20    192.168.199.x  —  ready

20/20 nodes ready
```

Повторяй `check` каждые 2-3 минуты пока все не станут ready.

---

## 5. Запустить тесты

Все команды запускать из `cloud/`:
```bash
cd /Users/maxim/dev/Decentr_my_own/cloud
export YC_TOKEN=$(yc iam create-token)   # если прошло > 12ч
```

---

### 5.0 Smoke-тест (сначала всегда!)

**Зачем:** 1 эпоха, проверить что все 20 нод стартуют и пишут результаты.  
**Время:** ~10-20 минут.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only smoke
```

Если smoke не прошёл — **не запускай 50-эпоховые эксперименты**.  
Смотри раздел [9. Что делать если что-то пошло не так](#9-что-делать-если-что-то-пошло-не-так).

---

### 5.1 Synchronous baseline

**Зачем:** классический синхронный AllReduce — baseline для сравнения.  
**Время:** ~2-3 часа.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only sync_static
```

---

### 5.2 Async static

**Зачем:** async gossip без adaptive scheduler.  
**Время:** ~2-3 часа.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_static
```

---

### 5.3 Async adaptive (главный метод)

**Зачем:** async + adaptive workload scheduler — основной метод статьи.  
**Время:** ~2-3 часа.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_adaptive_default
```

---

### 5.4–5.6 EMA sweep

**Зачем:** насколько быстро scheduler должен реагировать на скорость нод.

```bash
# EMA 0.1 — агрессивная адаптация
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ema_alpha_0_1

# EMA 0.5 — баланс
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ema_alpha_0_5

# EMA 0.99 — медленная адаптация
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only ema_alpha_0_99
```

---

### 5.7–5.8 Low communication

**Зачем:** можно ли реже слать веса без потери точности.

```bash
# Push каждые 25 шагов (sparse)
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only low_comm_sparse

# Push каждые 50 шагов (very sparse)
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only low_comm_very_sparse
```

---

### 5.9–5.10 Staleness sweep

**Зачем:** как свежесть обновлений влияет на качество.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only staleness_2

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only staleness_8
```

---

### 5.11–5.12 Batch size sweep

**Зачем:** влияние размера батча на скорость и качество.

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only batch_16

python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only batch_128
```

---

### Запустить всю кампанию сразу (все 13 экспериментов)

Оркестратор запускает их последовательно и пропускает уже завершённые.  
**Время: ~30-40 часов суммарно.**

```bash
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml
```

---

### Посмотреть статус кампании

```bash
python3 orchestrate.py status --suite suites/cifar10_wan_campaign.yaml
```

Пример вывода:
```
Suite: cifar10_wan_campaign
Name                                Enabled      Status
---------------------------------------------------------
smoke                                   yes        done
sync_static                             yes     pending
async_static                            yes     pending
async_adaptive_default                  yes     pending
...
top_k_fastest                            no    disabled
```

---

### Перезапустить конкретный эксперимент

```bash
# Сбросить статус одного эксперимента
python3 orchestrate.py reset \
  --suite suites/cifar10_wan_campaign.yaml \
  --only smoke

# Затем запустить снова
python3 orchestrate.py run --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only smoke --force
```

---

### Открыть SSH в любую ноду

```bash
# По номеру (0 = bastion, 1 = decentr-02, ...)
python3 orchestrate.py shell --pool pool.yaml --node 0
python3 orchestrate.py shell --pool pool.yaml --node decentr-05
```

---

## 6. Собрать и сравнить результаты

### Результаты сохраняются двумя способами

**1. Локально через rsync** (оркестратор тянет сам в конце каждого эксперимента):
```
cloud/results/{run_id}/{node_id}/
  ├── run_summary.json      # итоговые метрики
  ├── async_epoch_metrics.csv
  └── *.log
```

**2. В git-репозиторий** (ноды пушат сами, если deploy key настроен):
```
git@github.com:maxtro91627/decentr-results.git
```

---

### Собрать результаты вручную (если оркестратор был убит)

```bash
python3 orchestrate.py collect \
  --pool pool.yaml \
  --run-id cifar10_wan_campaign__smoke__20260601T120000
```

Результаты окажутся в `cloud/results/{run_id}/`.

---

### Сравнить все эксперименты таблицей

```bash
python3 orchestrate.py compare --results-dir ./results
```

Или выбрать конкретные:
```bash
python3 orchestrate.py compare --results-dir ./results \
  --run-ids \
    cifar10_wan_campaign__sync_static__... \
    cifar10_wan_campaign__async_static__... \
    cifar10_wan_campaign__async_adaptive_default__...
```

---

## 7. СТОП — удалить все ноды срочно

**Основная статья расходов — VM. Удаляй сразу как закончил.**

```bash
export YC_TOKEN=$(yc iam create-token)
cd /Users/maxim/dev/Decentr_my_own/cloud/provision/terraform-yc
terraform destroy -auto-approve
```

Удаляет всё за ~1-3 минуты. VM удаляются первыми — биллинг останавливается почти мгновенно.

**Проверить что всё удалено:**
```bash
yc compute instance list
# Должно вернуть пустой список
```

---

## 8. Стоимость и биллинг

| Конфигурация | Стоимость |
|---|---|
| 1 нода: 4 vCPU / 8 GB / `standard-v3` | ~5-7 ₽/час |
| 20 нод | ~100-140 ₽/час |
| Smoke-тест (~20 мин) | ~35-50 ₽ |
| 1 эксперимент 50 эпох (~2-3ч) | ~200-420 ₽ |
| Вся кампания 13 экспериментов | ~2600-5500 ₽ |

> **Важно:** биллинг идёт пока VM существуют — даже если они простаивают.  
> Удаляй ноды сразу после завершения кампании.

**Токен истекает через 12 часов.** Если запускаешь длинную кампанию — обнови токен заранее:
```bash
export YC_TOKEN=$(yc iam create-token)
```

---

## 9. Что делать если что-то пошло не так

### Нода не прошла check (bootstrap не завершён)

```bash
# Зайти на ноду и посмотреть лог
python3 orchestrate.py shell --pool pool.yaml --node decentr-03
tail -f /opt/decentr/bootstrap.log
```

### Эксперимент завис / оркестратор убит

```bash
# Собрать что есть
python3 orchestrate.py collect --pool pool.yaml --run-id <run_id>

# Перезапустить с нуля
python3 orchestrate.py reset --suite suites/cifar10_wan_campaign.yaml --only <exp_name>
python3 orchestrate.py run   --pool pool.yaml --suite suites/cifar10_wan_campaign.yaml --only <exp_name>
```

### Посмотреть логи обучения на конкретной ноде

```bash
python3 orchestrate.py shell --pool pool.yaml --node decentr-01
# На ноде:
tail -f /tmp/dc_boot_<run_id>.log          # bootstrap-нода
tail -f /tmp/dc_fol_<run_id>_decentr-05.log  # follower-нода
```

### Токен истёк (ошибка 401)

```bash
export YC_TOKEN=$(yc iam create-token)
```

### Пересоздать ноды (если что-то сломалось на уровне VM)

```bash
# Удалить
terraform destroy -auto-approve

# Поднять заново
terraform apply
# Затем дождаться bootstrap (~10 мин) и запустить нужные эксперименты
```

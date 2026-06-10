# RUNBOOK — запуск кампании на 20 нодах Selectel

## Быстрый старт (Terraform путь — рекомендуется)

```
cloud/
├── provision/terraform/   ← поднять ноды (1 раз)
├── orchestrate.py         ← запустить кампанию
└── suites/cifar10_wan_campaign.yaml
```

---

## Шаг 0. Один раз — ключи и конфиг

```bash
cd ~/dev/final_proj/Decentr_my_own/cloud

# 1. SSH-ключ для нод
ssh-keygen -t ed25519 -f ~/.ssh/decentr_id_ed25519 -N ""

# 2. Deploy key для results repo (read/write)
ssh-keygen -t ed25519 -f ~/.ssh/decentr_results_deploy_key -N ""
# Добавь ~/.ssh/decentr_results_deploy_key.pub в GitHub → decentr-results → Deploy Keys

# 3. Заполни secrets.env
cp secrets.env.example secrets.env
# Отредактируй: ORCHESTRATOR_SSH_PUBKEY_PATH, RESULTS_*, PROJECT_*
```

---

## Шаг 1. Поднять ноды через Terraform

```bash
cd provision/terraform

# Разово — скопировать и заполнить tfvars
cp terraform.tfvars.example terraform.tfvars
# Заполни: domain_name, username, password, project_id
# flavor_id узнай:
export SEL_DOMAIN_NAME="..." SEL_USERNAME="..." SEL_PASSWORD="..." SEL_PROJECT_ID="..."
python3 ../selectel_new.py list-flavors --min-vcpu 4 --min-ram-gb 8

# Сгенерировать cloud-init (один раз после изменений в secrets.env)
python3 ../../render_cloud_init.py \
  --template cloud-init.subnet.yaml.template \
  --out cloud-init.rendered.yaml

# Поднять ноды
terraform init
terraform plan
terraform apply   # подтвердить 'yes'
# → генерирует ../../pool.yaml автоматически
```

---

## Шаг 2. Дождаться bootstrap (~10 мин)

```bash
cd ../../    # обратно в cloud/

# Проверять каждые 2 минуты:
python3 orchestrate.py check --pool pool.yaml
# Ждать: "20/20 nodes ready"
```

---

## Шаг 3. Smoke-тест

```bash
python3 orchestrate.py run \
  --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only smoke \
  --dry-run   # сначала проверить inventory без подключения к нодам

python3 orchestrate.py run \
  --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only smoke
```

---

## Шаг 4. Полная кампания

```bash
python3 orchestrate.py run \
  --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --partial-push-minutes 10 \
  --max-runtime-hours 8
```

Эксперименты запускаются **последовательно** (все 20 нод на каждый).
Промежуточные результаты сохраняются в `./results/<run_id>/` каждые 10 минут.

---

## Шаг 5. Статус кампании

```bash
python3 orchestrate.py status --suite suites/cifar10_wan_campaign.yaml
```

---

## Шаг 6. Повторить упавший эксперимент

```bash
# Сбросить статус одного эксперимента
python3 orchestrate.py reset \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_adaptive_default

# Запустить только его
python3 orchestrate.py run \
  --pool pool.yaml \
  --suite suites/cifar10_wan_campaign.yaml \
  --only async_adaptive_default
```

---

## Шаг 7. Сохранить результаты и снести ноды

```bash
# Убедись что results/ зафиксированы в decentr-results git repo
# (orchestrate.py делает это автоматически через rsync, но проверь)

# ВАЖНО: снести ноды сразу после кампании (деньги!)
cd provision/terraform
terraform destroy   # подтвердить 'yes'
```

---

## SSH в отдельную ноду

```bash
python3 orchestrate.py shell --pool pool.yaml --node 0        # bootstrap
python3 orchestrate.py shell --pool pool.yaml --node decentr-05
```

---

## Структура результатов

```
results/
└── cifar10_wan_campaign__smoke__20240601T120000/
    ├── decentr-01/
    │   ├── async_run_summary.json
    │   ├── metrics.csv
    │   └── *.log
    ├── decentr-02/
    ...
```

---

## Подводные камни

| Проблема | Решение |
|----------|---------|
| `20/20 nodes ready` не достигается | `python3 orchestrate.py shell --node 0` → `cat /opt/decentr/bootstrap.log` |
| Config port не открывается | Проверь security group в Selectel — порты 50051-50052 открыты? |
| rsync зависает | Проверь SSH ключ в `pool.yaml` / `secrets.env` |
| `terraform apply` ругается на `external_network_name` | Проверь в панели Selectel → Сети → Публичные, подставь правильное имя в `variables.tf` |
| Хочу меньше нод для теста | `server_count = 3` в `terraform.tfvars`, `terraform apply` |

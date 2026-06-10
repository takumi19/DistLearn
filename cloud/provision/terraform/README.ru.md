# Provisioning через Terraform (Selectel) — основной путь

Один `terraform apply` поднимает N серверов (по умолчанию 20) в одном проекте
Selectel на одной приватной подсети, вешает на каждый floating IP, и
**автоматически генерирует `cloud/pool.yaml`**. `terraform destroy` сносит всё.

Почему так, а не Tailscale: все 20 нод в одном проекте на одной подсети видят
друг друга по приватным IP напрямую — Tailscale для связи между нодами не нужен.
Это убирает самую хрупкую часть (auth key, hostname propagation). Если позже
захочешь настоящий WAN (ноды в разных регионах/у разных провайдеров) — там
вернётся Tailscale (см. `../../cloud-init.yaml.template`).

```
            terraform apply
                  │
   ┌──────────────┼───────────────────────────────┐
   │ network + subnet + router + security group     │
   │ 20 × {port, boot volume, instance, floating IP} │
   └──────────────┬───────────────────────────────┘
                  │ outputs
          ┌───────▼────────┐
          │ cloud/pool.yaml │  (сгенерирован автоматически)
          └────────────────┘
```

## 0. Один раз — в панели Selectel

1. Создай **проект** (Облачная платформа → Проекты). Запомни его UUID → `project_id`.
2. Создай **сервисного пользователя** с ролью `member` на этом проекте
   (Управление доступом → Пользователи → вкладка «Сервисные пользователи»).
   Запомни имя и пароль → `username`, `password`.
3. Узнай **account ID** (справа вверху в панели) → `domain_name`.
4. Выбери регион и зону (например `ru-9` / `ru-9a`).

## 1. Один раз — поставь Terraform

```bash
# macOS:
brew install terraform
# проверь:
terraform version
```

## 2. Один раз — секреты и cloud-init

Из `cloud/`:

```bash
cp secrets.env.example secrets.env
# заполни: ORCHESTRATOR_SSH_PUBKEY_PATH, RESULTS_REPO_URL, RESULTS_DEPLOY_KEY_PATH,
#          RESULTS_GIT_NAME/EMAIL, PROJECT_REPO_URL, PROJECT_REPO_BRANCH.
# В subnet-режиме TAILSCALE_AUTH_KEY не используется — можно оставить любую заглушку.
```

Сгенерируй cloud-init для subnet-режима:

```bash
cd provision/terraform
python3 ../../render_cloud_init.py \
  --template cloud-init.subnet.yaml.template \
  --out cloud-init.rendered.yaml
```

## 3. Один раз — настрой terraform.tfvars

```bash
cp terraform.tfvars.example terraform.tfvars
```

Заполни `domain_name`, `username`, `password`, `project_id`, `region`,
`availability_zone`, `volume_type`.

`flavor_id` узнай так (после `terraform init`, либо через Python-хелпер):

```bash
# вариант A — Python-хелпер (корректная аутентификация):
python3 ../selectel.py list-flavors
# вариант B — позже через openstack CLI, если установишь его.
```

Подбери flavor ~4 vCPU / 8 GB, впиши его id в `terraform.tfvars`.

## 4. Поднимаем 20 серверов

```bash
terraform init
terraform plan      # посмотри что будет создано (≈ 5 ресурсов + 5×count)
terraform apply     # подтверди 'yes'
```

Через несколько минут:
- 20 серверов созданы и стартуют cloud-init (~5-10 мин на установку torch);
- `cloud/pool.yaml` сгенерирован автоматически (приватные + публичные IP);
- `terraform output` покажет floating IPs и SSH-подсказку.

## 5. Ждём bootstrap и проверяем

Cloud-init на каждой ноде ставит зависимости и клонирует репозитории. Проверка:

```bash
cd ../..            # обратно в cloud/
python3 orchestrate.py check --pool pool.yaml
```

Когда `20/20 nodes ready` — переходи к `../../RUNBOOK.ru.md`, раздел 3 (smoke) и 4 (кампания).

## 6. После кампании — снести серверы (важно для денег!)

```bash
cd provision/terraform
terraform destroy   # подтверди 'yes' — удалит все 20 серверов, IP, диски, сеть
```

Состояние кампании уже в git-репо `decentr-results`, так что серверы можно
безопасно убивать — результаты не потеряются.

## Заметки / подводные камни

- **`external_network_name`**: на большинстве пулов Selectel floating-IP пул
  называется `external-network`. Если `apply` ругается на пул — проверь имя в
  панели (Сети → Публичные) и поправь переменную.
- **`volume_type` / `availability_zone`** зонозависимы: для `ru-9` это
  `fast.ru-9a`. Для другого региона поменяй суффикс (`ru-7` → `fast.ru-7a` и т.п.).
- **hostname**: OpenStack ставит OS-hostname из имени инстанса (`decentr-NN`).
  В subnet-режиме нам это не критично (общаемся по IP), но имена видны в `nova`.
- **Стоимость**: floating IP и boot volume тарифицируются отдельно от сервера.
  20 нод × (сервер + 40 GB диск + 1 IP) — прикинь в калькуляторе Selectel перед
  `apply`. `terraform destroy` обнуляет биллинг.
- **Меньше нод для теста**: поставь `server_count = 3` в tfvars, прогони smoke,
  потом `server_count = 20` и снова `apply` — Terraform до-создаст недостающие.

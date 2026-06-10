# Кампания экспериментов: подробное описание и сравнения

**Проект:** децентрализованное gossip-обучение ResNet-18 на CIFAR-10, 20 нод в Yandex Cloud (WAN, гетерогенные машины).
**Дата:** 2026-06-08.
**Связанные файлы:** [`CAMPAIGN_COMMANDS.ru.md`](CAMPAIGN_COMMANDS.ru.md) · [`suites/cifar10_wan_campaign.yaml`](suites/cifar10_wan_campaign.yaml) · [`run_selected.sh`](run_selected.sh) · [`01_EXPERIMENT_ROADMAP.ru.md`](../../../Downloads/01_EXPERIMENT_ROADMAP.ru.md)

## Содержание
1. [Цель и метод](#1-цель-и-метод)
2. [Фиксированная база](#2-фиксированная-база-reference)
3. [Глоссарий](#3-глоссарий)
4. [Детальные карточки по 28 экспериментам](#4-детальные-карточки-по-28-экспериментам)
5. [Сравнения между экспериментами](#5-сравнения-между-экспериментами)
6. [Текущая очередь](#6-текущая-очередь)
7. [Операционная логика прогона](#7-операционная-логика-прогона)
8. [Метрики](#8-метрики)
9. [Сводка гипотез](#9-сводка-гипотез)
10. [Дисциплина безопасности](#10-дисциплина-безопасности)
11. [Псевдокод всех запусков](#11-псевдокод-всех-запусков)

---

## 1. Цель и метод

Изучаем **децентрализованное обучение без центрального сервера**: 20 нод учат свою копию ResNet-18 на своём куске CIFAR-10 и периодически обмениваются весами с соседями («gossip»). Главный вопрос:

> Какие решения по **коммуникации, расписанию и топологии** дают лучший компромисс между **качеством** (test accuracy) и **стоимостью** (нагрузка на сеть, время) в условиях WAN и неодинаковых по мощности нод?

**Метод — «один фактор за раз».** Есть фиксированная база (`async_adaptive_default`). Каждый эксперимент = база **плюс ровно одно изменение**. Тогда разницу в результате можно честно приписать именно этому изменению. Все 28 тестов — это 28 контролируемых вариаций одного и того же базового прогона.

---

## 2. Фиксированная база (reference)

`orchestrate.py → DEFAULT_TRAINING`. Менять нельзя.

| Параметр | База | Дефолт схемы | Почему так |
|---|---|---|---|
| модель / норм | ResNet-18 / **GroupNorm** | — | GN корректен при малых/неоднородных батчах между нодами |
| seed | 42 | — | воспроизводимость |
| epochs | 50 | — | ради **сопоставимости**, не рекорда |
| batch_size | 32 | — | — |
| lr | **0.03** | — | проверено; lr=0.005 давал лишь 44–47.7% |
| momentum / wd | 0.9 / 5e-4 | — | стандарт ResNet/CIFAR |
| grad_clip | 1.0 | — | страховка от взрыва при async-подмешивании |
| партиции | micro_shards, heterogeneous | — | мелкая **неравномерная** нарезка — имитация перекоса |
| push_interval | **10** | 25 | как часто слать веса |
| push_fanout | **0 = всем** | 2 | скольким соседям слать |
| mixing_alpha | **0.2** | 0.5 | вес подмешивания чужих весов |
| max_staleness | **4** | 4 | мягкий decay вклада устаревших апдейтов |
| throughput_ema | **0.9** | 0.9 | инерция оценки скорости нод планировщиком |
| communication_policy | **full** | full | кому слать (full = всем) |
| mode / scheduler | **async / adaptive** | static / static | без барьеров + балансировка нагрузки |

> Где «База» ≠ «Дефолт схемы» — значение задано явно в `DEFAULT_TRAINING`. Это важно: новые поля Phase 2/3 по умолчанию воспроизводят базу, поэтому каждый эксперимент меняет ровно один фактор.

---

## 3. Глоссарий

- **gossip / push** — раз в `push_interval` шагов нода рассылает веса части соседей; получатель **подмешивает** их с `mixing_alpha`. Знание расползается без сервера.
- **async vs sync** — async: каждая нода в своём темпе + асинхронный обмен; sync: барьер + AllReduce, все в ногу.
- **adaptive scheduler** — раздаёт `micro_shards` пропорционально `throughput_ema` ноды, чтобы быстрые не простаивали.
- **client drift** — между обменами локальные модели расходятся (каждая тянет к своим данным); реже обмен → сильнее drift.
- **PeerScoreTracker** — на каждый пир ведёт EMA: `success` (доля успешных пушей), `latency` (round-trip), `capacity` (мощность = `relative_speed`), `usefulness` (снижение loss от его апдейтов).
- **staleness** — насколько устаревший апдейт принимаем; в коде — *мягкий decay* вклада по нормированному version-gap (не жёсткий обрез).
- **fanout** — скольким соседям слать за шаг; **policy** — *кому именно*; **topology** — *кто кому сосед*.

---

## 4. Детальные карточки по 28 экспериментам

Формат карточки: **Меняем** (точный knob) · **Механизм/зачем** · **Гипотеза** · **Смотреть** (метрики) · **Читать исход** (как интерпретировать «лучше/хуже базы»).

### Группа 1. Baselines — опорные точки

**1. `async_adaptive_default` — РЕФЕРЕНС (= база)**
- **Меняем:** ничего (`mode=async`, `scheduler=adaptive`).
- **Механизм:** локальный SGD; каждые 10 шагов push **всем 19** соседям; получатель подмешивает с α=0.2; adaptive раздаёт micro_shards по `throughput_ema`.
- **Роль:** нулевая точка отсчёта. С её числами сравнивается всё остальное (Δacc, Δсеть, Δwall-clock).
- **Смотреть:** финальная test acc, кривая сходимости, `mixed_peer_updates`, баланс размеров shards по нодам — это эталонные значения.

**2. `async_static` — нужен ли adaptive-планировщик?**
- **Меняем:** `scheduler_mode: adaptive → static`.
- **Механизм:** раскладка данных фиксирована, не подстраивается под скорость нод.
- **Гипотеза:** на гетерогенном кластере static хуже — быстрые ноды недозагружены, общий throughput ниже. Ждём `adaptive ≥ static`.
- **Смотреть:** Δacc vs база; утилизация по нодам; wall-clock/эпоху.
- **Читать исход:** заметно лучше у adaptive → планировщик оправдан (вклад работы). Равно → гетерогенность на этом кластере не критична.

**3. `sync_static` — оправдан ли async вообще?**
- **Меняем:** `mode: async → sync`.
- **Механизм:** классический синхронный SGD: на каждом шаге барьер + AllReduce; нет client drift, нет staleness, но самая медленная нода тормозит всех.
- **Гипотеза:** sync даёт «чище» усреднение → возможно выше acc-per-epoch, НО на WAN барьер+страгглеры убивают wall-clock.
- **Смотреть:** финальная acc И **wall-clock/эпоху** (ключевое), throughput.
- **Читать исход:** если async-adaptive близок по acc за кратно меньшее время → async оправдан (H1). Если sync сильно выше по acc → за децентрализацию платим качеством.

### Группа 2. EMA sweep — реактивность планировщика

Общее: `throughput_ema` = вес истории в оценке скорости ноды. **Высокий = инертно** (медленно реагирует), **низкий = дёргано** (быстро). База 0.9. Ось сравнения: 0.1 → 0.5 → 0.9 → 0.99.

**4. `ema_tput_0_1` — быстрая реакция · ✅ СДЕЛАН = 52.7%**
- **Меняем:** `throughput_ema: 0.1`.
- **Гипотеза:** мгновенно подстраивает раскладку под смену скоростей, но может «звенеть» на шуме. Факт: **52.7%** — рабочий режим.

**5. `ema_tput_0_5` — компромисс**
- **Меняем:** `throughput_ema: 0.5`.
- **Гипотеза:** среднее между реактивностью 0.1 и инерцией 0.9.

**6. `ema_tput_0_99` — очень медленная адаптация**
- **Меняем:** `throughput_ema: 0.99`.
- **Гипотеза:** почти не реагирует — если нода замедлилась/ускорилась, планировщик узнаёт поздно → дисбаланс. ⚠️ Ранее **падал на эпохе 3** (медленная реакция перегружала bootstrap).
- **Читать исход (вся группа):** ждём **немонотонную** кривую с оптимумом в середине; края (0.99) рискуют дисбалансом/крашем.
- **Смотреть:** финальная acc, дисперсия размеров shards во времени (стабильность раскладки), наличие краша.

### Группа 3. Peer-selection policies (Phase 2) — главная новизна

Общее: вместо «всем 19» нода отбирает **4** пира (`push_fanout: 4`) через `PeerScoreTracker`. Якорь сравнения — `random_fanout` (тот же приём «не всем», но случайно). Главный вопрос группы (H2): **бьёт ли умный отбор случайный, и какой фактор важнее?**

**7. `top_k_fastest` — по скорости** ⏳ *идёт сейчас*
- **Меняем:** `communication_policy: top_k_fastest`, `push_fanout: 4`.
- **Механизм:** score = `capacity / (1 + latency)`. `capacity` = `relative_speed` из инвентаря, `latency` = EMA round-trip. Шлём 4 быстрейшим/ближайшим.
- **Гипотеза:** быстрые ноды = хорошие хабы → быстрее распространение. **Риск:** централизация, медленные голодают → drift.
- **Смотреть:** acc vs `random_fanout`; **разброс acc по нодам** (голодание); диверсити получателей (на кого приходятся пуши).

**8. `top_k_reliable` — по надёжности**
- **Меняем:** `communication_policy: top_k_reliable`, `push_fanout: 4`.
- **Механизм:** score = `success_rate / (1 + latency)`. Избегаем нод, куда пуши часто фейлятся.
- **Гипотеза:** меньше бюджета впустую на сбойные линки → стабильнее, выше эффективная частота обмена.
- **Смотреть:** `failed_pushes` (ждём ниже), acc, стабильность.

**9. `ping_aware` — скорость × надёжность**
- **Меняем:** `communication_policy: ping_aware`, `push_fanout: 4`.
- **Механизм:** score = `success_rate · capacity / (1 + ping_ms)`. Комбинирует оба фактора.
- **Гипотеза:** лучший компромисс — пир должен быть И быстрым, И достижимым.
- **Читать исход:** ключевая проверка — **бьёт ли `ping_aware` отдельно взятые `top_k_fastest` и `top_k_reliable`?** Если да → комбинация факторов оправдана.

**10. `reliability_aware_graph` — прунинг рёбер**
- **Меняем:** `communication_policy: reliability_aware` (fanout не задаётся).
- **Механизм:** не top-k, а *прунинг* — на статическом графе глушим рёбра с `success` ниже порога, но не опускаемся ниже `min-degree` (чтоб не изолировать ноду).
- **Гипотеза:** убрать стабильно битые линки, сохранив связность → меньше потерь без централизации.
- **Смотреть:** сколько рёбер отрезано, связность графа, `failed_pushes`, acc.
- **Отличие:** режем «плохое», а не выбираем «лучшее» (в отличие от top_k).

**11. `top_k_useful` — по полезности**
- **Меняем:** `communication_policy: top_k_useful`, `push_fanout: 4`.
- **Механизм:** score = usefulness EMA (насколько апдейты пира снижали loss). ⚠️ **только train/val, никогда test** — иначе утечка.
- **Гипотеза:** слать тем, чьи веса реально информативны (не просто быстрым/надёжным).
- **Смотреть:** acc; не центрируется ли на нодах с «удобными» данными; согласованность usefulness с реальным вкладом.

### Группа 4. Topology — сколько связности нужно

Общее: меняем граф соседства; «степень» = число соседей. База = full (19). Ось: ring(2) → expander(4) → star(центр.) → full(19).

**12. `ring_topology` — кольцо (степень 2)**
- **Меняем:** `topology: ring`.
- **Механизм:** каждый связан с 2 соседями по кольцу; диаметр ~N/2.
- **Гипотеза:** информация ползёт медленно по кольцу → ниже acc при фикс-50-эпохах. Минимум сети.

**13. `sparse_expander` — экспандер (степень 4)**
- **Меняем:** `topology: expander`.
- **Механизм:** циркулянт/экспандер — малая степень, но хорошая спектральная связность (малый диаметр, быстрое смешивание).
- **Гипотеза:** **≈ full при кратно меньшей сети** — «sweet spot» топологии.

**14. `star_topology` — звезда (хаб = decentr-01)**
- **Меняем:** `topology: star`.
- **Механизм:** все связаны только с хабом; диаметр 2.
- **Гипотеза:** быстрое усреднение через хаб, НО узкое горло на хабе + единая точка отказа.
- **Смотреть (группа):** acc vs степень графа; суммарная сеть; **нагрузка на хаб** (star); время.

### Группа 5. Communication frequency — цена частоты обмена

Общее: `push_interval_steps`, база = 10. Реже = дешевле, но больше drift. Ось: 10 → 100 → 300 → 500.

**15. `low_comm_100` — ×10 реже**
- **Меняем:** `push_interval_steps: 100`.

**16. `low_comm_300` — ×30 реже**
- **Меняем:** `push_interval_steps: 300`.

**17. `low_comm_500` — ×50 реже**
- **Меняем:** `push_interval_steps: 500`.
- **Гипотеза (группа):** монотонно — чем реже, тем дешевле сеть и тем ниже acc (растёт client drift). Ищем «колено» приемлемой деградации.
- **Смотреть:** кривая «acc vs частота», суммарный трафик, расхождение моделей нод между обменами (drift-индикатор).

### Группа 6. Random fanout — разреженный случайный gossip

Общее: `push_fanout`, база = 0 (все 19). Якорь для Группы 3.

**18. `random_fanout_1` — 1 сосед/шаг · ✅ СДЕЛАН, опубликован**
- **Меняем:** `push_fanout: 1`.
- **Механизм:** push 1 ротируемому соседу (offset = step + node_id). ~**19× экономия** сети.
- **Гипотеза:** большая экономия при умеренной потере качества.

**19. `random_fanout_2` — 2 соседа/шаг**
- **Меняем:** `push_fanout: 2`.
- **Механизм:** ~**9.5× экономия**.
- **Читать исход (группа):** 2 должен быть ближе к базе, чем 1. Эти точки — фон, на котором оцениваем «умные» политики (Группа 3).

### Группа 7. Bonus — staleness и batch

**20. `staleness_2` — только свежее (жёсткий decay)**
- **Меняем:** `max_staleness: 2` (база 4).
- **Механизм:** вклад апдейта быстро затухает с ростом version-gap → фактически учитываем только свежие.
- **Гипотеза:** меньше шума от устаревших, но выкидываем полезный сигнал → возможно медленнее.

**21. `staleness_8` — терпим старое (мягкий decay)**
- **Меняем:** `max_staleness: 8`.
- **Гипотеза:** больше данных в смешивании, но шум от старых весов → возможно нестабильнее.
- **Читать исход (пара):** ось 2 → 4(база) → 8; ждём оптимум у компромисса. Смотреть: acc, стабильность, средний реальный version-gap.

**22. `batch_16` — мельче батч**
- **Меняем:** `batch_size: 16` (база 32).
- **Механизм:** вдвое больше шагов/эпоха → вдвое больше пушей → больше обмена и дольше; градиент шумнее.
- **Гипотеза:** два эффекта сразу — оптимизация (шум) и коммуникация (число пушей ∝ шагам).

**23. `batch_128` — крупнее батч**
- **Меняем:** `batch_size: 128`.
- **Механизм:** вчетверо меньше шагов → меньше пушей → меньше обмена; градиент глаже, но реже обновления.
- **Читать исход (пара):** ось 16 → 32(база) → 128; разделяем эффект на качество vs на интенсивность коммуникации.

### Группа 8. Phase 3 — эффективность коммуникации (flag-gated)

**24. `dynamic_graph` — граф, адаптирующийся в рантайме**
- **Меняем:** `communication_policy: dynamic`, `push_fanout: 4`, `dynamic_graph_rebuild_windows: 5`, `dynamic_graph_min_degree: 4`.
- **Механизм:** активный набор пиров **пересобирается каждые 5 окон** по `ping_aware`-скорам; `min_degree=4` держит связность.
- **Гипотеза:** рантайм-адаптация графа к реальным скоростям/надёжности бьёт статический выбор.
- **Читать исход:** сравнить с `ping_aware` (та же формула, но статичный выбор) и с фиксированными топологиями.

**25. `delta_exchange` — шлём дельты, а не полные веса**
- **Меняем:** `payload_mode: delta`.
- **Механизм:** отправляем `state − last_sent`; получатель реконструирует. **Лосслес** (дельта точная).
- **Гипотеза:** дельты компактнее (особенно поздно, когда веса почти не меняются) → меньше трафика при той же мат.точности → acc ≈ база.
- **Риск:** version skew / накопление ошибки реконструкции.
- **Смотреть:** acc ≈ база? объём payload во времени (должен падать к концу); отсутствие drift реконструкции.

**26. `compress_float16` — fp16 (~2×)**
- **Меняем:** `compression: float16`.
- **Механизм:** payload в половинной точности.
- **Гипотеза:** почти без потерь — усреднение терпит fp16.

**27. `compress_quant8` — 8-бит (~4×)**
- **Меняем:** `compression: quant8`.
- **Механизм:** uint8 + per-tensor scale.
- **Гипотеза:** заметнее сжатие, но качество держится; проверяем порог.

**28. `compress_topk` — top-k 10% (~10×+)**
- **Меняем:** `compression: topk_sparse` (`topk_fraction: 0.1`).
- **Механизм:** шлём только 10% наибольших по модулю координат (+ fp16); 90% обнуляются.
- **Гипотеза:** агрессивно — проверяем устойчивость mixing к **сильно разреженным** апдейтам (не «рассыпается» ли сходимость).
- **Читать исход (группа):** ось none(база) → fp16(2×) → q8(4×) → topk(10×); ждём, что acc держится до некоторого порога, потом ломается. **Где ломается — и есть ответ.**

---

## 5. Сравнения между экспериментами

Все 28 — вариации одного прогона, поэтому сравнивать их осмысленно тремя способами: **(А) лестницы** — один knob по возрастанию; **(Б) дуэли** — прямой head-to-head с общим вопросом; **(В) кросс-группа** — разные механизмы, одна цель.

### 5.А «Лестницы» (один фактор по оси)

| Лестница | Точки (база **жирным**) | Ожидаемый тренд | Что выявляет |
|---|---|---|---|
| Частота обмена | **10** → 100 → 300 → 500 | acc ↓, сеть ↓ (монотонно) | где «колено» цена/качество |
| Fanout | **19(все)** → 2 → 1 | acc ↓, сеть ↓ | цена разреженности |
| EMA реактивность | 0.1 → 0.5 → **0.9** → 0.99 | **немонотонно**, оптимум в середине | лучшую инерцию планировщика |
| Топология (степень) | ring 2 → expander 4 → **full 19** (+ star) | acc ↑ со связностью; expander≈full? | сколько графа реально нужно |
| Staleness | 2 → **4** → 8 | немонотонно, оптимум у компромисса | терпимость к устаревшему |
| Batch | 16 → **32** → 128 | два эффекта (шум + коммуникация) | разделить оптимизацию и обмен |
| Сжатие | **none** → fp16 2× → q8 4× → topk 10× | сеть ↓ монотонно; acc держится до порога | предел сжатия без потерь |

### 5.Б Дуэли (head-to-head)

**Треугольник baseline** (`sync_static` ⟷ `async_static` ⟷ `async_adaptive_default`):
- `async_adaptive` **vs** `async_static` → **вклад планировщика** (помогает ли adaptive).
- `async` **vs** `sync` → **цена/выгода децентрализации** (acc против wall-clock). Ответ на H1.

**Шут-аут политик** (Группа 3, бюджет `fanout=4`):
- `top_k_fastest` vs `top_k_reliable` vs `top_k_useful` → **какой одиночный фактор важнее** (скорость / надёжность / польза).
- `ping_aware` vs (`top_k_fastest`, `top_k_reliable`) → **бьёт ли комбинация** отдельные факторы.
- `reliability_aware_graph` vs top_k-политики → **прунинг «плохого» vs выбор «лучшего»**.
- любая политика vs `random_fanout` → **умный срез vs случайный** (ядро H2).

> ⚠️ **Важная оговорка по бюджету.** Политики работают при `fanout=4`, а `random_fanout` — при 1 и 2. Это **не идеально равный бюджет**: умные политики шлют большему числу пиров. Поэтому «умный-4 vs случайный-1/2» надо читать с поправкой — корректнее сравнивать с *интерполяцией* random между full(19)→2→1 в точке 4. Для чистой дуэли «умный vs случайный при равном бюджете» не хватает `random_fanout_4`. Если хочешь железобетонный вывод по H2 — стоит добавить этот эксперимент (одна строка в suite).

**Phase-3 дуэли:**
- `dynamic_graph` vs `ping_aware` → **динамический пересбор vs статический скоринг** (та же формула) — окупается ли рантайм-адаптация.
- `dynamic_graph` vs топологии (ring/expander/star) → адаптивный граф vs фиксированный.
- `delta_exchange` vs `compress_*` → **лосслес сжатие payload (дельта) vs лоссовое (квантизация)** — два пути к меньшему объёму.

### 5.В Кросс-группа: 5 способов урезать сеть

Пять групп — это **пять ортогональных рычагов** уменьшения трафика. Общая ось сравнения: **сколько качества сохраняется на единицу сэкономленной сети** (это и есть H4).

| # | Рычаг | Эксперименты | Как режет | Лоссовость | Экономия |
|---|---|---|---|---|---|
| 1 | Реже слать | `low_comm_*` | частота пушей | косвенно (drift) | 10–50× |
| 2 | Меньшему числу — случайно | `random_fanout_*` | получателей наугад | косвенно | 9.5–19× |
| 3 | Меньшему числу — умно | `top_k_*`, топологии, `dynamic_graph` | получателей по скору/структуре | косвенно | ~5× (4/19) |
| 4 | Меньше байт — лосслес | `delta_exchange` | объём payload точно | нет | зависит |
| 5 | Меньше байт — лоссово | `compress_*` | объём payload с потерей | да | 2–10× |

Рычаги **в принципе комбинируются** (напр. expander + delta + fp16), но в кампании меняем по одному, чтобы измерить чистый вклад каждого. Победитель по H4 — рычаг с самым пологим падением acc на росте экономии.

### 5.Г Матрица прямой сопоставимости

С чем что **честно** сравнивать (общий контроль = всё, кроме одного фактора):

| Пара/набор | Общий контроль | Выявляет | Оговорка |
|---|---|---|---|
| adaptive vs static | всё кроме scheduler | вклад планировщика | — |
| async vs sync | всё кроме mode | цена async | sync ещё и static |
| top_k_* между собой | fanout=4 | какой скор-фактор важнее | — |
| top_k_* vs random_fanout | приём «не всем» | умный vs случайный срез | ⚠️ разный fanout (4 vs 1/2) |
| random 1 vs 2 vs full | push_fanout | кривая разреженности | — |
| low_comm_* + база | push_interval | кривая частоты | — |
| ema_* + база | throughput_ema | кривая реактивности | — |
| топологии между собой | граф | связность vs стоимость | — |
| compress_* + база | уровень сжатия | порог потерь | — |
| dynamic_graph vs ping_aware | формула скоринга | статика vs динамика | — |

---

## 6. Текущая очередь

`run_selected.sh` (выбор пользователя): reference (есть) + якорь разреженного gossip + вся группа политик + динамический граф. ~42 ч (~6 ч/прогон). Каждый прогон сам публикует в git, чистит ноды и запускает следующий.

| # | Эксперимент | Статус |
|---|---|---|
| — | `async_adaptive_default` | **пропущен** — результаты есть |
| 1 | `random_fanout_1` | ✅ готов, опубликован, ноды очищены |
| 2 | `top_k_fastest` | ⏳ **идёт сейчас** (`…top_k_fastest__20260608T195127`) |
| 3 | `top_k_reliable` | ⌛ в очереди |
| 4 | `ping_aware` | ⌛ в очереди |
| 5 | `reliability_aware_graph` | ⌛ в очереди |
| 6 | `dynamic_graph` | ⌛ в очереди |

> **Полнота списка.** В suite **31** запись = **28 тестов выше** + 3 не-теста: `smoke` (`enabled`, 2-эпоховая инфра-проверка, см. раздел 10); `smoke_dynamic` (**disabled**, валидатор Phase 3-кода, пройден 2026-06-08); `single_node_baseline` (**disabled**: одна нода без gossip — абсолютный нижний baseline, ждёт драйвера `orchestrate local-train`).

---

## 7. Операционная логика прогона

`orchestrate.py run --only X` для каждого эксперимента:
1. Обновляет YC-токен.
2. Поднимает/кикает 19 followers, ждёт готовности.
3. SCP inventory → decentr-01 (bootstrap), `start-run`.
4. Followers ждут конфиг от bootstrap, учатся 50 эпох.
5. **Detection-only polling** — Mac лишь проверяет `test -f *_run_summary.json`, **ничего не качает**.
6. По завершении **bootstrap** собирает метрики по внутренней сети, `rsync → decentr-results`, `commit + push` (deploy-key).
7. Cleanup нод (kill + rm артефактов/логов/tmp).
8. Следующий эксперимент.

> Mac не хранит результаты — всё уходит с нод в git и удаляется. Hardening: `caffeinate` (Mac не спит), timeout-recovery (обучение идёт на нодах даже при потере сети Mac'ом), partial-summary при краше, cleanup только после успешной публикации.

---

## 8. Метрики

- **Качество:** финальная test acc; кривая сходимости (эпох до целевой acc).
- **Сеть/стоимость:** доля времени на push, объём payload, частота пушей, экономия vs база.
- **Здоровье gossip:** `mixed_peer_updates`, диверсити отправителей/получателей, `failed_pushes`, распределение staleness.
- **Стабильность:** отсутствие NaN; **разброс acc по нодам** — главный индикатор централизации/голодания (особенно `top_k_*`, `star`).

---

## 9. Сводка гипотез

- **H1.** async + adaptive ≥ sync по качеству и **существенно** лучше по wall-clock на WAN.
- **H2.** Умный выбор пиров (policy) бьёт случайный fanout при равном бюджете *(см. оговорку 5.Б — нужен `random_fanout_4` для чистоты)*.
- **H3.** Полная связность избыточна — expander степени 4 ≈ full.
- **H4.** Коммуникацию можно сильно удешевить (реже / дельты / сжатие) до порога без потери качества.
- **H5.** Динамическая адаптация графа в рантайме > статических решений.

**Уже известно:** база lr=0.03 (0.005 слишком мал — 44–47.7%); `ema_tput_0_1` = **52.7%**; `random_fanout_1` завершён и опубликован.

---

## 10. Дисциплина безопасности

- Каждое новое поле конфига по умолчанию **воспроизводит базу** → отсутствие ключа не ломает прогон.
- Каждая новая возможность сперва проходит **2-эпоховый smoke** (нужны `async_run_summary.json` + mixed-апдейты + нет NaN), потом 50-эпоховый прогон. `smoke_dynamic` валидирован (20/20, loss↓, mixed=53, без NaN).
- Patch-файлы деплоятся на все 20 нод и сверяются по md5 (60/60).
- Упавший прогон **не останавливает** остальные: помечается timeout, идём дальше.

---

## 11. Псевдокод всех запусков

> Упрощённо, но повторяет реальный control-flow. Источники: `patches/async_gossip.py`,
> `patches/communication_policy.py`, `patches/sync_barrier.py`, `decentr_my_own/data/*`.
> Все 28 — это **один каркас (11.0–11.4) с одним изменением** (таблица 11.6).

### 11.0. Базовый каркас — общий для всех async-прогонов

```python
# ── выполняется на КАЖДОЙ ноде (self_id) со своим списком соседей neighbors ──
model     = ResNet18(norm="group")
opt       = SGD(lr=0.03, momentum=0.9, weight_decay=5e-4)         # база
tracker   = PeerScoreTracker()           # EMA success/latency/capacity/usefulness на пир
last_sent = None                         # буфер для delta_exchange
active    = neighbors                    # для dynamic_graph пересобирается на лету

for epoch in range(50):
    sampler.set_epoch(epoch)                      # детерминированный решафл (11.4)
    for step, batch in enumerate(local_loader):   # local_loader = свои шарды (heterogeneous)
        loss = cross_entropy(model(batch.x), batch.y)
        loss.backward()
        clip_grad_norm_(model, 1.0)
        opt.step(); opt.zero_grad()

        # ── PUSH: раз в push_interval шагов рассылаем веса ──
        if step % push_interval == 0:                          # база: push_interval=10
            targets = select_push_neighbors(active, policy,    # 11.1; база: policy=full,
                          fanout, step, self_id, tracker)      #          fanout=0 (всем 19)
            tensors, kind = build_payload(model.state_dict(),  # 11.2; база: full + none
                          payload_mode, compression, last_sent)
            for t in targets:
                ok, dt = send_async(t, tensors, kind)
                tracker.record_push(t, ok=ok, latency_s=dt)
            if payload_mode == "delta":
                last_sent = clone(model.state_dict())

        # ── MIX: подмешиваем пришедшие чужие веса ──
        for payload in drain_incoming():                       # 11.3
            mix_peer_payload(model, payload, mixing_alpha=0.2,  # база: alpha=0.2,
                             max_staleness=4, tracker=tracker)  #       max_staleness=4

        # ── ADAPTIVE: перебалансируем нагрузку по скорости нод ──
        if scheduler == "adaptive" and step % rebalance_window == 0:   # каждые 50 батчей
            rebalance_shards(throughput_ema=0.9)               # тяжёлым нодам больше шардов

        # ── DYNAMIC GRAPH: пересобрать активный набор (только policy=dynamic) ──
        if policy == "dynamic" and window % rebuild_windows == 0:
            ranked = sort(neighbors, key=lambda n: -tracker.score(n, "ping_aware"))
            active = ranked[: max(min_degree, fanout)]         # min_degree держит связность

    if epoch % eval_every == 0:
        evaluate(model, val)                                   # usefulness берём отсюда, не из test
evaluate(model, test)                                          # финальная метрика
publish_metrics()                                              # → bootstrap → git, ноды чистятся
```

### 11.1. select_push_neighbors — Phase 2, «кому слать»

```python
def select_push_neighbors(neighbors, policy, fanout, step, self_id, tracker):
    ordered = sort(neighbors, key=id)                          # детерминизм
    if policy in {"full", "random_fanout"}:
        return rotate_fanout(ordered, fanout, step, self_id)   # 0 → все; N → N по кругу (offset=step+self_id)
    if policy == "reliability_aware":                          # прунинг рёбер, не top-k
        good = [n for n in ordered if tracker.success(n) >= MIN_SUCCESS]
        return good if len(good) >= MIN_DEGREE else best_by_success(ordered, MIN_DEGREE)
    ranked = sort(ordered, key=lambda n: (-tracker.score(n, policy), n.id))   # top-k
    return ranked[:fanout] if 0 < fanout < len(ranked) else ranked

def score(peer, policy):                                       # из PeerScoreTracker
    if policy == "top_k_fastest":  return capacity / (1 + latency)
    if policy == "top_k_reliable": return success  / (1 + latency)
    if policy == "ping_aware":     return success * capacity / (1 + latency_ms)
    if policy == "top_k_useful":   return usefulness           # ↓loss после подмешивания (train/val)
```

### 11.2. build_payload — Phase 3, «что слать»

```python
def build_payload(state, payload_mode, compression, last_sent):
    if payload_mode == "delta":                                # delta_exchange
        state = {k: state[k] - last_sent[k] for k in state}    # лосслес дельта
    if compression == "none":        return state,             "async_weights"
    if compression == "float16":     return half(state),       "async_weights_f16"   # ~2x
    if compression == "quant8":      return uint8(state)+scale, "async_weights_q8"    # ~4x
    if compression == "topk_sparse":                                                  # ~10x+
        mask = top_k_abs(state, frac=0.10)                     # оставить 10% крупнейших |w|
        return half(state * mask),                             "async_weights_topk"
```

### 11.3. mix_peer_payload — приём и подмешивание

```python
def mix_peer_payload(model, payload, mixing_alpha, max_staleness, tracker):
    peer = decompress(payload.tensors, payload.kind, reference=model.state_dict())
    if peer is None: return                                    # битый payload → пропустить
    if payload.kind == "async_delta":
        peer = model.state_dict() + peer                       # реконструкция дельты
    w = mixing_alpha * staleness_decay(payload.version_gap, max_staleness)  # мягкий decay
    model.load_state_dict((1 - w) * model.state_dict() + w * peer)
    tracker.record_usefulness(payload.src, val_loss_drop)      # для top_k_useful
```

### 11.4. Данные — одинаково во всех (см. раздел 3)

```python
def shuffle_indices(total, seed, scope, token):                # token = epoch
    offset = 0 if scope == "global_seeded" else 10_000_019     # база: global_seeded
    rng = Random(seed + offset + token * 1_000_003)            # seed=42
    return rng.shuffle(range(total))                           # детерминированный решафл/эпоху
# партиция: heterogeneous — шарды по нодам ∝ мощности; split train/val сидирован seed=42
```

### 11.5. Sync-вариант — только `sync_static`

```python
for round in range(50):
    sampler.set_epoch(round)
    for batch in local_loader:
        loss = cross_entropy(model(batch.x), batch.y); loss.backward()
        clip_grad_norm_(model, 1.0); opt.step(); opt.zero_grad()
    all_reduce_average(model.parameters())     # БАРЬЕР: ждём всех, усредняем — нет gossip/staleness/drift
    evaluate(model, val)
evaluate(model, test)
```

### 11.6. Чем каждый из 28 запусков отличается от каркаса

Каждый = база + один override. «Что меняется» — строки каркаса (11.0–11.5).

| # | Запуск | Override vs база | Что меняется в псевдокоде |
|---|---|---|---|
| 1 | `async_adaptive_default` | — | каркас как есть (policy=full, fanout=0, adaptive) |
| 2 | `async_static` | scheduler=static | блок ADAPTIVE выключен (нет `rebalance_shards`) |
| 3 | `sync_static` | mode=sync | каркас → **11.5** (all-reduce-барьер вместо PUSH/MIX) |
| 4–6 | `ema_tput_0_1/0_5/0_99` | throughput_ema=X | `rebalance_shards(throughput_ema=X)` — инерция оценки скоростей |
| 7 | `top_k_fastest` | policy, fanout=4 | select → `capacity/(1+lat)`, top-4 |
| 8 | `top_k_reliable` | policy, fanout=4 | select → `success/(1+lat)`, top-4 |
| 9 | `ping_aware` | policy, fanout=4 | select → `success*capacity/(1+ping)`, top-4 |
| 10 | `reliability_aware_graph` | policy | select → прунинг рёбер по success (min-degree floor) |
| 11 | `top_k_useful` | policy, fanout=4 | select → `usefulness` EMA, top-4 |
| 12 | `ring_topology` | topology=ring | `neighbors` = 2 соседа по кольцу |
| 13 | `sparse_expander` | topology=expander | `neighbors` = 4 (циркулянт/экспандер) |
| 14 | `star_topology` | topology=star | `neighbors` = {hub}; у хаба — все |
| 15–17 | `low_comm_100/300/500` | push_interval=X | условие PUSH срабатывает раз в X шагов |
| 18–19 | `random_fanout_1/2` | fanout=1/2 | `rotate_fanout` шлёт 1/2 соседям (offset=step+self_id) |
| 20–21 | `staleness_2/8` | max_staleness=X | `staleness_decay` в MIX жёстче/мягче |
| 22–23 | `batch_16/128` | batch_size=X | размер batch в `local_loader` → число шагов/пушей |
| 24 | `dynamic_graph` | policy=dynamic, fanout=4, rebuild=5, min_degree=4 | блок DYNAMIC GRAPH активен (`active` пересобирается) |
| 25 | `delta_exchange` | payload_mode=delta | `build_payload` шлёт дельты; `last_sent` обновляется |
| 26–28 | `compress_float16/quant8/topk` | compression=X | `build_payload` сжимает payload (2× / 4× / 10×) |

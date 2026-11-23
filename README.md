# llm — собственная языковая модель с нуля

Decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU, KV cache), Byte-Level BPE токенизатор, data pipeline,
Training Engine (pretrain, SFT, DPO), Inference Engine, Reasoning Engine, Evaluation Engine и локальный
OpenAI-совместимый API. Всё написано на PyTorch без Transformers и готовых весов.

[Технический аудит Core](docs/technical-audit.md) содержит измеренные результаты, ограничения модели и команды для повторения проверок. Экспорт создаёт экспериментальный пакет для inference. Публикация production-модели требует отдельного решения по результатам оценки.

## Установка
```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

## Быстрый старт
```bash
.venv/bin/llm hardware                                  # оборудование и доступные профили
.venv/bin/llm prepare-data --vocab-size 8192             # загрузка, очистка, дедупликация, токенизатор, бинарные файлы
.venv/bin/python scripts/prepare_preference.py data/processed data/tokenizer 4000
.venv/bin/llm train --config configs/debug.yaml          # проверочное обучение (1,84 млн параметров)
.venv/bin/llm train --config configs/tiny_session.yaml   # ограниченная сессия (15,7 млн параметров)
.venv/bin/llm resume --config configs/tiny.yaml          # продолжить с последнего корректного checkpoint
.venv/bin/llm finetune --config configs/tiny_sft.yaml    # SFT, затем configs/tiny_dpo.yaml для DPO
.venv/bin/llm evaluate --checkpoint checkpoints/tiny_session
.venv/bin/llm generate --checkpoint checkpoints/tiny_session "The history of" --max-new-tokens 60
.venv/bin/llm chat --checkpoint checkpoints/tiny_sft
.venv/bin/llm serve --checkpoint checkpoints/tiny_sft    # http://127.0.0.1:8000/v1/chat/completions
tensorboard --logdir experiments
```

## Abylkhaiyr AI Studio

Локальная панель управления использует тот же FastAPI backend, реестр экспериментов, checkpoint и Training Engine.
Она доступна только на `127.0.0.1` и не передаёт данные во внешние AI API.

```bash
.venv/bin/llm studio
```

Studio автоматически использует самый свежий checkpoint. Для явного выбора:

```bash
.venv/bin/llm studio --checkpoint checkpoints/tiny_session
```

Из корня проекта команда проверяет свободный порт и наличие Node.js и npm, устанавливает
зависимости из lock-файла и пересобирает интерфейс при изменении исходников. Production-интерфейс
и API доступны по адресу [http://127.0.0.1:8000](http://127.0.0.1:8000). Если checkpoint ещё
не создан, Studio открывается без загруженной модели. Её можно выбрать позднее на странице «Модели».

Начальные локальные учётные данные: `admin` / `admin` (роль ADMIN). Первый вход обязывает сменить пароль;
пароль существующего администратора при перезапусках не сбрасывается.
Сессии, пользователи, чаты и уведомления хранятся в `studio/studio.db` (в Git не попадает).

### Аккаунты, роли и уведомления

Любой пользователь может зарегистрироваться на странице `/register` (имя, фамилия, email, логин, пароль; роль
`USER` назначается автоматически, подтверждение email не требуется). Вход возможен по логину или email.
ADMIN видит весь Studio и раздел «Пользователи»; USER видит только AI Chat, свои диалоги, уведомления, профиль и
настройки и всегда общается с моделью, выбранной администратором. Права проверяются на сервере (REST и WebSocket).
Подробности API и протокола событий: [docs/accounts.md](docs/accounts.md).

### Каталоги, режимы и оценка качества

- **Каталог запусков** объединяет состояние pipeline и метрики обучения (включая вложенные `experiments/<группа>/<запуск>`).
  На Dashboard по умолчанию выбирается активный запуск, иначе последний завершённый; исторический запуск выбирается явно.
  Раздел «История» показывает все запуски с фильтрами, поиском, сортировкой, серверной пагинацией и сравнением.
- **Каталог моделей и реестр checkpoint** разделены. Пользователь видит только модели, которые разрешил администратор;
  статусы checkpoint: production, validated, candidate, experimental, rejected. Экспериментальные модели помечаются явно.
- **Профили inference** Lite, Medium, High, Very High задают реальные лимиты запроса (токены, контекст, время, бюджет
  инструментов, попытки проверки, Reasoning Engine) и не меняют знания модели. Профиль недоступен, если контекст
  загруженной модели слишком мал. Администратор включает профили для USER.
- **Центр оценки** группирует отчёты по эксперименту, датасету и методике, показывает только измеренные метрики
  («Не проверено» вместо нуля), проверяет критерии качества и сравнивает checkpoint только внутри совместимой группы.

Подробности и таблица критериев приёмки: [docs/studio_p2.md](docs/studio_p2.md).

Для разработки из корня проекта:

```bash
.venv/bin/llm studio --dev
```

Команда запускает FastAPI, ждёт готовности сервера и затем запускает Vite на
[http://127.0.0.1:5173](http://127.0.0.1:5173). Запросы HTTP и WebSocket идут через тот же
адрес, который открыт в браузере. Если порты заняты, укажите свободные через
`--port` и `--frontend-port`. Для ручного запуска Vite с нестандартным портом backend
передайте `STUDIO_BACKEND_PORT` в окружении frontend.

При разрыве соединения открытые страницы сохраняют последние полученные данные и помечают их как
неактуальные. Studio повторно подключает поток событий с задержкой и ограниченным числом попыток.
Если попытки исчерпаны, нажмите «Переподключить». После перезапуска сервера частично
сохранённый ответ доступен в истории чата. При несовпадении версии API запустите Studio из
того же checkout, из которого собран интерфейс, и обновите страницу.

## Конфигурации

### Обучение на NVIDIA GPU в Windows

Для установленного окружения с CUDA:

```powershell
.\scripts\start_gpu_training.ps1
.\scripts\stop_gpu_training.ps1
```

Запуск работает в фоне и продолжает `checkpoints/tiny_gpu` с последней корректной
контрольной точки. Остановка сохраняет прогресс после текущего шага. Повторный запуск
снимает запрос остановки. Конфигурация `configs/tiny_gpu.yaml` использует CUDA, BF16,
быстрое causal attention и пакетные операции AdamW; лимит — 6000 шагов, проверка и
сохранение каждые 100 шагов. При шести проверках без улучшения обучение завершается.
Выключение компьютера останавливает процесс; для продолжения выполните запуск снова.

Для диалогового дообучения после базового обучения:

```powershell
.venv\Scripts\python.exe scripts/prepare_gpu_chat.py
.\scripts\start_gpu_training.ps1 -Stage sft
```

SFT использует лучшие базовые веса и отдельный корпус диалогов на русском, английском
и казахском; лимит — 1600 шагов. `-Stage pretrain` явно продолжает базовое обучение.
Без аргумента скрипт выбирает SFT, если его лучшие веса уже существуют. Повторный
запуск во время обучения сообщает PID и ничего не перезапускает. Обе стадии ограничены
заданным числом шагов: повторение команды после завершения не добавляет шаги автоматически.

Studio при запуске предпочитает диалоговые веса базовым и загружает лучший checkpoint.
Для ручного выбора используйте `checkpoints/tiny_gpu_sft/best`. Проверка реальных ответов:

```powershell
.venv\Scripts\python.exe scripts/check_gpu_chat.py
```

Примеры сохраняются в `experiments/tiny_gpu_sft/chat_check.json`. Это небольшая проверка
поведения, а не подтверждение общей точности: модель на 15,7 млн параметров по-прежнему
может выдумывать факты и ошибаться на незнакомых запросах.

Журнал: `experiments/tiny_gpu/train.log`, метрики: `experiments/tiny_gpu/metrics.jsonl`.
Лучшие веса сохраняются в `checkpoints/tiny_gpu/best`. Это базовое обучение на корпусе,
а качество ответов в чате требует последующего instruction tuning и отдельной оценки.

Если данных и токенизатора ещё нет, сначала выполните:

```powershell
.venv\Scripts\llm.exe prepare-data --data-config configs/data_gpu.yaml --vocab-size 8192
```

Подготовка загружает открытые источники; прямые файлы и лицензии записаны в
`configs/data_gpu.yaml` и сохраняются в манифесте корпуса. Не запускайте подготовку
повторно одновременно с обучением: она пересоздаёт файлы данных.

| Файл | Параметры | Назначение |
|---|---:|---|
| `configs/debug.yaml` | 1,84 млн | проверка пайплайна (300 шагов, ~40 с) |
| `configs/tiny_session.yaml` | 15,7 млн | ограниченная сессия, 600 шагов |
| `configs/tiny.yaml` | 15,7 млн | продолжительное обучение, 6000 шагов (~4 часа на M2 Pro) |
| `configs/data_small.yaml` | — | источники данных и лицензии |

Данные: Wikipedia (kk, ru, simple en), TinyStories, GSM8K, CodeAlpaca, Dolly, HH-RLHF (лицензии в `configs/data_small.yaml`
и `data/processed/manifest.json`).

## Этап 11: SFT и сохранение языковых способностей

[Результаты MPS, подбор SFT и выбор checkpoint](docs/stage11_sft.md) ·
[Model Card](docs/model_card_stage11.md) ·
[Диагностика: почему модель повторяет вопрос](docs/diagnosis_stage12.md) ·
[Корпус v4 и continued pretraining](docs/stage13_v4.md) ·
[Этап 14. Данные v4.1, оценка и эксперименты](docs/stage14_data_eval.md) ·
[Этап 15. Корпус v5 и расширенная валидация](docs/stage15_conversational_v5.md) ·
[Этап 16. Базовые знания и dialogue-formatted pretraining](docs/stage16_core_training.md).

## Проверка качества кода

Фоновое обучение на Windows управляется командами
`.\scripts\control_continuous.ps1 start`, `status` и `stop`.
Контроллер читает локальный список разрешённых источников, сохраняет состояние
цикла и сравнивает новые веса с текущими перед публикацией.
Локальный агент запускается через `scripts/run_developer_agent.py` и использует
модель проекта для решения Python CLI задач с проверкой в Docker.

[Архитектура цикла, измерения и ограничения](docs/local-learning.md).

[Dialogue v4. Аудит, CUDA SFT и проверка диалога](docs/dialogue-v4.md).

`scripts/check.sh` запускает ruff, black, mypy и pytest.

Документация: `docs/mathematics.md`, `docs/architecture.md`, `docs/data.md`, `docs/instruction_tuning.md`.

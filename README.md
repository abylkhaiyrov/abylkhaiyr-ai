# llm — собственная языковая модель с нуля

Decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU, KV cache), Byte-Level BPE токенизатор, data pipeline,
Training Engine (pretrain, SFT, DPO), Inference Engine, Reasoning Engine, Evaluation Engine и локальный
OpenAI-совместимый API. Всё написано на PyTorch без Transformers и готовых весов.

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

Начальные локальные учётные данные: `admin` / `admin`. Смените пароль после первого входа.
Сессия и чаты хранятся в `studio/studio.db`.

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
[Model Card](docs/model_card_stage11.md).

## Проверка качества кода
`scripts/check.sh` запускает ruff, black, mypy и pytest.

Документация: `docs/mathematics.md`, `docs/architecture.md`, `docs/data.md`, `docs/instruction_tuning.md`.

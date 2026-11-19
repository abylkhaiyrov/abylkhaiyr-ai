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

## Конфигурации
| Файл | Параметры | Назначение |
|---|---:|---|
| `configs/debug.yaml` | 1,84 млн | проверка пайплайна (300 шагов, ~40 с) |
| `configs/tiny_session.yaml` | 15,7 млн | ограниченная сессия, 600 шагов |
| `configs/tiny.yaml` | 15,7 млн | продолжительное обучение, 6000 шагов (~4 часа на M2 Pro) |
| `configs/data_small.yaml` | — | источники данных и лицензии |

Данные: Wikipedia (kk, ru, simple en), TinyStories, GSM8K, CodeAlpaca, Dolly, HH-RLHF (лицензии в `configs/data_small.yaml`
и `data/processed/manifest.json`).

## Проверка качества кода
`scripts/check.sh` запускает ruff, black, mypy и pytest.

Документация: `docs/mathematics.md`, `docs/architecture.md`.

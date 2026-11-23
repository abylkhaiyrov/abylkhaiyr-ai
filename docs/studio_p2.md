# Studio: каталоги, профили, оценка качества и проверка

## Каталог запусков

`RunCatalog` (`src/llm/studio/run_catalog.py`) объединяет pipeline-state и `metrics.jsonl` (глубина до двух уровней).
Идентификаторы устойчивы: `pipeline:<имя>` и `run:<путь>`. Статусы: `running`, `completed`, `stopped`, `failed`.
Выбор по умолчанию: активный запуск, иначе последний завершённый или остановленный, иначе пустое состояние.
Для запусков без конфигурации число шагов неизвестно, поэтому статус «завершён» для них не выводится.

API (ADMIN): `GET /api/catalog/runs`, `GET /api/catalog/runs/{id}`, поле `runtime.runs`.
`GET /api/runs` не изменялся.

## Процессы

Список объединяет управляемые процессы из SQLite и завершённые pipeline. У завершённых записей PID сбрасывается, а
журнал событий доступен через `GET /api/processes/{id}/events`. Очередь задач в проекте отсутствует: раздел «Очередь»
показывает только процессы со статусом `pending`.

## Модели

- Каталог: `GET /api/models/catalog` (ADMIN), `PUT /api/models/allowed`.
- Реестр checkpoint с серверной пагинацией: `GET /api/models/checkpoints`.
- Пользователь: `GET /api/chat/models`. Выбор модели проверяется на сервере при каждой генерации.
- По умолчанию (политика не задана) пользователю доступны только production-модели.

## Профили inference

`src/llm/studio/inference_profiles.py`. Параметры профиля: максимум токенов ответа, бюджет контекста, лимит времени
генерации, бюджет инструментов, число кандидатов и попыток проверки Reasoning Engine, лимиты ресурсов.
`GET /api/inference/profiles`, `PUT /api/inference/profiles/allowed`. Профиль сохраняется в настройках чата и в
`generation_params` сообщения. USER не может использовать отключённый или недоступный профиль.

## Центр оценки

- Реестр метрик с направлением улучшения: `GET /api/evaluation/metrics`.
- Группы: `GET /api/evaluation/groups`, `GET /api/evaluation/groups/{id}`.
- Сравнение: `GET /api/evaluation/compare?group=...&checkpoint=...` (от 2 до 8). Несовместимый выбор отклоняется (409).
- Критерии качества берутся из `llm.training.promotion` (порог доли пройденных диалоговых проверок и повторов).
- Победитель назначается только по доле пройденных диалоговых проверок при достаточной разнице и пройденных критериях;
  одного Perplexity недостаточно. Доверительные интервалы не рассчитываются, пока в отчётах нет числа примеров.

## Проверка

Реальные E2E-сценарии запускаются против работающего Studio:

```bash
E2E_BASE_URL=http://127.0.0.1:8000 E2E_ADMIN_PASSWORD=... E2E_BACKEND=1 npx playwright test
node frontend/e2e/bootstrap.mjs   # смена пароля admin и тестовый пользователь
```

`E2E_RESTART_SCRIPT` включает сценарий перезапуска backend (скрипт принимает `stop` и `start`).
В CI (`.github/workflows/frontend-e2e.yml`) запускаются проверки login-страницы, реальный Studio на пустом проекте и
backend-проверки. Ubuntu, Windows и macOS отмечаются проверенными только после фактического прогона CI.

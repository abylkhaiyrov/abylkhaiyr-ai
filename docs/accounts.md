# Аккаунты, роли, чаты и уведомления Studio

## Модель данных (SQLite, `studio/studio.db`)
| Таблица | Назначение |
| --- | --- |
| `users` | `uuid`, `first_name`, `last_name`, `email` (уникальный), `username` (уникальный), `password_hash` (Argon2id), `role` (`ADMIN`/`USER`), `is_active`, `last_login_at`, `last_active_at`, `requires_password_change` |
| `sessions` | хеш токена, CSRF-токен, срок действия, `last_seen_at`, `user_agent` |
| `chats`, `messages` | диалоги и сообщения; каждый чат принадлежит `user_id`; поля `pinned`, `archived`, варианты ответа |
| `generation_jobs` | задачи генерации: `request_id`, `user_id`, `chat_id`, сообщение, статус, `created_at`, `started_at`, `completed_at`, ошибка |
| `notifications` | `uuid`, `user_id`, `type`, `title`, `message`, `chat_id`, `generation_id`, `is_read`, `read_at`, `created_at` |

Роли хранятся как строка с проверкой на сервере (две фиксированные роли), поэтому отдельной таблицы `roles` нет.
`generation_jobs` — это таблица задач генерации; статусы API: `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`,
`CANCELLED` (поле `task_status`; внутренний `status` сохранён для совместимости).
Миграция (`alembic/versions/20261121_05_*`, а также автоматическое дополнение схемы при старте) сохраняет
существующих пользователей и чаты; существующий `admin` получает роль ADMIN, его чаты остаются за ним.
Уникальность уведомления о генерации обеспечивает частичный уникальный индекс `(user_id, generation_id)`.

## Авторизация
HttpOnly-cookie сессии + CSRF-заголовок `X-CSRF-Token` (двойная отправка), срок сессии 12 часов. Новый токен
выдаётся при каждом входе и регистрации, предъявленная ранее сессия отзывается (защита от session fixation).
Ограничения: вход — 10 попыток с адреса и 5 на аккаунт за 5 минут; регистрация — 5 с адреса и 60 всего за час.
Ответы `401` — нет сессии, `403` — нет прав или заблокирован аккаунт, `429` — лимит.

## REST API
Публичные: `POST /api/auth/register`, `POST /api/auth/login` (`identifier` или `username`).
Авторизованные: `GET /api/auth/me`, `POST /api/auth/logout`, `PUT /api/auth/password`, `PATCH /api/auth/profile`,
`GET /api/auth/sessions`, `DELETE /api/auth/sessions/{id}`, `POST /api/auth/sessions/revoke-others`.
Чаты (только свои): `GET /api/chats` (`query`, `archived`, `limit`, `offset`), `POST /api/chats`,
`GET /api/chats/{id}?limit=` (последние сообщения + `total_messages`), `PATCH /api/chats/{id}`
(`title`, `pinned`, `archived`, настройки), `DELETE /api/chats/{id}`, экспорт, выбор варианта ответа.
`GET /api/chat/model` — активная модель (только публичные поля).
Уведомления (только свои): `GET /api/notifications` (`unread`, `type`, `limit`, `offset`, `sort`),
`GET /api/notifications/unread-count`, `GET /api/notifications/{id}`, `POST /api/notifications/{id}/read`,
`POST /api/notifications/read-all`, `DELETE /api/notifications/{id}`. Создаёт уведомления только backend.
Администратор: `GET /api/users` (`query`, `role`, `active`, `limit`, `offset`), `GET/PATCH /api/users/{id}`,
`POST /api/users/{id}/reset-password` (временный пароль показывается один раз, потребуется смена),
`POST /api/users/{id}/revoke-sessions`, а также все прежние административные маршруты (обучение, ресурсы, логи,
датасеты, реестр моделей, процессы). Действия администратора пишутся в `audit_logs`. Единственного активного
администратора нельзя заблокировать или лишить роли.

## WebSocket `/api/ws`
Одно соединение на вкладку, авторизация по cookie, проверка `Origin`. События получает только владелец:
`generation.queued|started|delta|completed|failed|cancelled|snapshot` (с `task_status`),
`notification.created|read|deleted|count.updated`, `session.revoked` (затем соединение закрывается).
Канал `studio` (runtime, система, модели, метрики) доступен только ADMIN. Heartbeat, `seq` соединения,
ограниченная очередь отправки и восстановление активных генераций после переподключения сохранены.
Генерация продолжается при смене раздела и закрытии вкладки; по завершении создаётся ровно одно уведомление.

## Запуск проверок
```bash
.venv/bin/ruff check src tests && .venv/bin/mypy src && .venv/bin/pytest -q
cd frontend && npx tsc -b && npx eslint . --max-warnings=0 && npx vitest run && npm run build
```

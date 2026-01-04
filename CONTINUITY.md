# CONTINUITY — rss_digest_bot

## Что это
Telegram-бот для работы с RSS-лентами:
- добавление RSS
- ручная синхронизация
- просмотр новостей за сегодня / неделю
- авто-дайджест по расписанию (на пользователя)
- fallback без ИИ при отсутствии баланса

Бот развёрнут и работает на сервере (`/opt/rss_digest_bot`), управление через systemd.

---

## Текущий статус (актуально на сейчас)

### ✅ Работает
- Основной бот (`rss_digest_bot.service`)
- Периодическая синхронизация RSS (`rss_digest_bot_sync.timer`, каждые 30 минут)
- Меню бота с кнопками
- Команда `/sync`, `/news`, `/digest`, `/digest_week`
- Авто-дайджест **по настройке пользователя**
- Кнопка **⚙️ Настройки** в меню
- Выбор интервала авто-дайджеста: 1 / 2 / 4 часа
- Корректная работа при нулевом балансе DeepSeek (fallback без ИИ)
- Корректная нарезка длинных сообщений (без обрыва)
- Часовой пояс сервера: **Europe/Moscow (MSK)**

---

## Архитектура авто-дайджеста

### Таблица user_settings
```sql
user_settings (
  user_id INTEGER PRIMARY KEY,
  digest_enabled INTEGER,
  digest_interval_min INTEGER,
  next_digest_at TEXT,
  last_digest_at TEXT
)

Логика

digest_tick запускается часто (через systemd timer)

для каждого пользователя проверяется next_digest_at

если время пришло:

берутся новости за последние 24 часа

формируется дайджест

отправляется в Telegram

next_digest_at сдвигается на интервал

systemd
Основной бот

rss_digest_bot.service

Синхронизация RSS

rss_digest_bot_sync.service

rss_digest_bot_sync.timer (30 минут)

Авто-дайджест

rss_digest_bot_digest.service

rss_digest_bot_digest.timer (каждые 10 минут)

Ключевые файлы

app/main.py — вся бизнес-логика

app/keyboards/main_menu.py — главное меню

app/keyboards/settings_menu.py — меню настроек

data/rss.db — SQLite база

logs/ — логи (НЕ коммитить)

Важные детали реализации
1. Время

сервер переведён в Europe/Moscow

в БД время хранится в ISO (UTC)

сравнение работает корректно

2. Ограничение Telegram

сообщения режутся по абзацам

лимит ~3800 символов

отправка частями через send_long_message

3. ИИ (DeepSeek)

используется для дайджеста

при ошибке / 402 — fallback без ИИ

бот никогда не падает из-за ИИ

Как проверить вручную
# проверить статус
systemctl status rss_digest_bot
systemctl status rss_digest_bot_digest.timer

# принудительно запустить авто-дайджест
/opt/rss_digest_bot/.venv/bin/python -m app.main digest_tick

# проверить настройки пользователя
sqlite3 data/rss.db "SELECT * FROM user_settings;"

Следующие шаги (НЕ СДЕЛАНО)

Унифицировать работу с датами:

использовать published_at вместо created_at

одинаковая логика для today / week / digest

Добавить:

настройку типа дайджеста (с ИИ / без ИИ)

выбор количества новостей

категории / фильтры

Очистка старых записей (entries) по TTL

Правила работы с проектом

Любые изменения:

коммит в Git

обновление этого файла (CONTINUITY.md)

Логи и zip-файлы в Git не добавлять

---

### Сохранить файл
В nano:
- `Ctrl + O` → Enter  
- `Ctrl + X`

---

## 5️⃣ Закоммитить якорь

```bash
git add CONTINUITY.md
git commit -m "Update CONTINUITY: auto-digest, settings, scheduler"
git push


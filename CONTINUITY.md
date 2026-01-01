# CONTINUITY — rss_digest_bot

## Что это
Telegram-бот для RSS:
- /add /list /del — управление RSS-каналами
- /sync — скачивает новые записи в SQLite
- /news — новости за сегодня
- /news week — новости за 7 дней

## Как запускать (важно)
Запускать только через:
./run.sh

Если бот “молчит” — первое проверить, что он запущен через ./run.sh.

## Хранилище
- SQLite: data/rss.db (не коммитится)
- .env с BOT_TOKEN (не коммитится)

## Установка
- Termux: python + venv
- requirements.txt фиксирует зависимости

## Известные нюансы
- Возможны telegram.error.TimedOut при длинных ответах/плохой сети — уменьшали объём выдачи для week.
- HTML в summary чистится через clean_html().

## Проверки
- /sync → должно писать “Новых записей: …”
- /news → должен выдавать список за сегодня
- /news week → должен выдавать короткий список за неделю

## Следующие шаги (план)
1) Фильтр по теме/категории: /news week football или /news football
2) /news month
3) Авто-sync по расписанию (cron/termux-job-scheduler)
4) Улучшить очистку summary (HTML entities, лишние пробелы)

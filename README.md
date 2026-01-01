# RSS Digest Bot (Termux)

Telegram bot: RSS subscriptions + sync + news for today/week.

## Commands
- /add <rss_url> — add feed
- /list — list feeds
- /del <id> — delete feed
- /sync — fetch new items into SQLite
- /news — today
- /news week — last 7 days

## Run (Termux)

./run.sh

## Notes
- DB stored in data/rss.db
- Do NOT commit .env (BOT_TOKEN)


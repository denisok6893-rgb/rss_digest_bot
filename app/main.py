import sys
import time
from pathlib import Path
import asyncio
import os
import re
import hashlib
import json
from datetime import datetime, timezone
from typing import Tuple
from telegram.request import HTTPXRequest
from openai import AsyncOpenAI

import aiosqlite
import feedparser
import httpx
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from app.keyboards.main_menu import main_menu_keyboard

load_dotenv()

DB_PATH = "data/rss.db"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


async def init_db() -> None:
    os.makedirs("data", exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, url)
            );
            """
        )

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subscription_id INTEGER NOT NULL,
                guid TEXT,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                summary TEXT,
                published_at TEXT,
                categories TEXT,
                hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, hash)
            );
            """
        )

        await db.execute("CREATE INDEX IF NOT EXISTS idx_entries_user_date ON entries(user_id, published_at);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_entries_user_sub_date ON entries(user_id, subscription_id, published_at);")
        await db.commit()


async def fetch_and_parse_feed(url: str) -> Tuple[str, int]:
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "rss-digest-bot/0.1"})
        r.raise_for_status()

    parsed = feedparser.parse(r.content)
    if not getattr(parsed, "feed", None) or parsed.entries is None:
        raise ValueError("Похоже, это не RSS/Atom (нет feed/entries).")

    title = (parsed.feed.get("title") or "").strip() or "Без названия"
    return title, len(parsed.entries)


def _to_text(v) -> str:
    return (v or "").strip()


def _entry_hash(url: str, guid: str, link: str, title: str) -> str:
    base = guid or link or (title + "|" + url)
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()


def _published_iso(entry) -> str:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not t:
        return ""
    dt = datetime(*t[:6], tzinfo=timezone.utc)
    return dt.isoformat()


def _categories_json(entry) -> str:
    tags = getattr(entry, "tags", None)
    if not tags:
        return "[]"
    cats = []
    for t in tags:
        if isinstance(t, dict):
            term = (t.get("term") or "").strip()
        else:
            term = (getattr(t, "term", "") or "").strip()
        if term:
            cats.append(term)
    cats = list(dict.fromkeys(cats))
    return json.dumps(cats, ensure_ascii=False)

import re

HTML_RE = re.compile(r"<[^>]+>")

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = HTML_RE.sub("", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    return " ".join(text.split())

def _get_deepseek_client() -> AsyncOpenAI:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return AsyncOpenAI(api_key=key, base_url=base_url)

async def make_digest(items: list[tuple[str, str, str]], period_label: str) -> str:
    # items: (title, summary, link)
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    lines = []
    for i, (title, summary, link) in enumerate(items, 1):
        s = clean_html(summary or "")
        s = (s[:300] + "…") if len(s) > 300 else s
        lines.append(f"{i}. {title}\n{s}\n{link}")
    source_block = "\n\n".join(lines)

    system = (
        "Ты помощник, который делает краткую сводку новостей строго по предоставленным пунктам. "
        "Нельзя добавлять факты, которых нет в тексте. Если данных мало — так и скажи."
    )
    user = (
        f"Сделай сводку за период: {period_label}.\n"
        "Формат:\n"
        "1) 4–7 буллетов: ключевые события/темы (без выдумок)\n"
        "2) 'Ключевые ссылки' — 3–5 ссылок из списка (без новых)\n\n"
        f"Данные:\n{source_block}"
    )

    client = _get_deepseek_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()

async def save_entries(user_id: int, subscription_id: int, feed_url: str, parsed) -> int:
    added = 0
    async with aiosqlite.connect(DB_PATH) as db:
        for e in parsed.entries[:200]:
            title = _to_text(getattr(e, "title", ""))
            link = _to_text(getattr(e, "link", ""))
            if not title or not link:
                continue

            guid = _to_text(getattr(e, "id", "")) or _to_text(getattr(e, "guid", ""))
            summary = _to_text(getattr(e, "summary", "")) or _to_text(getattr(e, "description", ""))
            published_at = _published_iso(e)
            categories = _categories_json(e)
            h = _entry_hash(feed_url, guid, link, title)

            try:
                await db.execute(
                    """
                    INSERT INTO entries(user_id, subscription_id, guid, title, link, summary, published_at, categories, hash)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (user_id, subscription_id, guid, title, link, summary, published_at, categories, h),
                )
                added += 1
            except aiosqlite.IntegrityError:
                pass

        await db.commit()
    return added


async def add_subscription(user_id: int, url: str, title: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO subscriptions(user_id, url, title) VALUES(?,?,?)",
                (user_id, url, title),
            )
            await db.commit()
            return "ok"
        except aiosqlite.IntegrityError:
            return "exists"


async def list_subscriptions(user_id: int) -> list[tuple[int, str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, title, url FROM subscriptions WHERE user_id=? ORDER BY id DESC",
            (user_id,),
        )
        rows = await cur.fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


async def get_subscriptions(user_id: int) -> list[tuple[int, str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, url, title FROM subscriptions WHERE user_id=? ORDER BY id DESC",
            (user_id,),
        )
        rows = await cur.fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]

async def get_today_news(user_id: int, limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND date(published_at, 'localtime') = date('now', 'localtime')
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return await cur.fetchall()

async def get_week_news(user_id: int, limit: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND published_at != ''
              AND date(published_at) >= date('now', '-6 days')
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return await cur.fetchall()

async def search_news(user_id: int, days: int, query: str, limit: int = 10):
    q = (query or "").strip().lower()
    if not q:
        return []

    like = f"%{q}%"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND published_at != ''
              AND date(published_at) >= date('now', ?)
              AND (
                lower(title) LIKE ?
                OR lower(summary) LIKE ?
                OR lower(categories) LIKE ?
              )
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (user_id, f"-{days} days", like, like, like, limit),
        )
        return await cur.fetchall()

async def search_news_advanced(user_id: int, days: int, query: str, category: str, limit: int = 10):
    q = (query or "").strip().lower()
    cat = (category or "").strip().lower()

    # days=6 => последние 7 дней (включая сегодня)
    modifier = f"-{max(int(days), 0)} days"

    where = [
        "user_id = ?",
        "published_at IS NOT NULL",
        "published_at != ''",
        "date(published_at) >= date('now', ?)",
    ]
    params = [user_id, modifier]

    if q:
        where.append("(lower(title) LIKE ? OR lower(summary) LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    if cat:
        # categories хранится JSON-строкой, поэтому простой LIKE
        where.append("lower(categories) LIKE ?")
        params.append(f"%{cat}%")

    where_sql = " AND ".join(where)

    sql = f"""
        SELECT title, link, summary, published_at
        FROM entries
        WHERE {where_sql}
        ORDER BY published_at DESC
        LIMIT ?
    """
    params.append(limit)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(sql, tuple(params))
        return await cur.fetchall()

async def delete_subscription(user_id: int, sub_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM subscriptions WHERE user_id=? AND id=?",
            (user_id, sub_id),
        )
        await db.commit()
        return cur.rowcount > 0


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я собираю новости из RSS и помогаю быстро их просматривать.\n"
        "Добавь источники и смотри сводки 👇",
        reply_markup=main_menu_keyboard(),
    )



async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /add <rss_url>")
        return

    url = context.args[0].strip()
    try:
        title, count = await fetch_and_parse_feed(url)
    except Exception as e:
        await update.message.reply_text(f"Не удалось добавить RSS.\nПричина: {type(e).__name__}: {e}")
        return

    status = await add_subscription(user_id, url, title)
    if status == "exists":
        await update.message.reply_text(f"Уже есть в списке: {title}\n{url}")
    else:
        await update.message.reply_text(f"Добавлено: {title}\nЗаписей в ленте: {count}\n{url}")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    rows = await list_subscriptions(user_id)
    if not rows:
        await update.message.reply_text("Список пуст. Добавь RSS через /add <url>.")
        return

    lines = ["Ваши RSS-каналы:"]
    for sub_id, title, url in rows:
        lines.append(f"{sub_id}) {title}\n{url}")
    await update.message.reply_text("\n\n".join(lines))


async def cmd_del(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /del <id>\nID смотри в /list.")
        return

    sub_id = int(context.args[0])
    ok = await delete_subscription(user_id, sub_id)
    await update.message.reply_text("Удалено." if ok else "Не найдено. Проверь ID в /list.")

async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    subs = await get_subscriptions(user_id)
    if not subs:
        await update.message.reply_text("Нет RSS-каналов. Добавь через /add <url>.")
        return

    total_added = 0
    failed = 0

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for sub_id, url, _title in subs:
            try:
                r = await client.get(url, headers={"User-Agent": "rss-digest-bot/0.1"})
                r.raise_for_status()
                parsed = feedparser.parse(r.content)
                if not getattr(parsed, "feed", None) or parsed.entries is None:
                    failed += 1
                    continue
                total_added += await save_entries(user_id, sub_id, url, parsed)
            except Exception:
                failed += 1

    await update.message.reply_text(f"Синхронизация готова.\nНовых записей: {total_added}\nОшибок: {failed}")

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    rows = await get_today_news(user_id, limit=10)  # (title, link, summary, published_at)
    if not rows:
        await update.message.reply_text("Нет новостей за сегодня. Сначала /sync или подожди авто-sync.")
        return

    items = [(r[0], r[2], r[1]) for r in rows]  # title, summary, link
    text = await make_digest(items, period_label="сегодня")
    await update.message.reply_text(text[:4000])

async def cmd_digest_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    rows = await get_week_news(user_id, limit=15)
    if not rows:
        await update.message.reply_text("Нет новостей за неделю. Сначала /sync или подожди авто-sync.")
        return

    items = [(r[0], r[2], r[1]) for r in rows]
    text = await make_digest(items, period_label="неделя")
    await update.message.reply_text(text[:4000])

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    args = [a.strip().lower() for a in (context.args or []) if a.strip()]
    category = ""
    # вытаскиваем cat:...
    args2 = []
    for a in args:
        if a.startswith("cat:") and len(a) > 4:
            category = a[4:].strip()
        else:
            args2.append(a)
    args = args2
    period = "today"
    query = ""

    if args:
        if args[0] in ("week", "7", "7d"):
            period = "week"
            query = " ".join(args[1:]).strip()
        else:
            query = " ".join(args).strip()

    if period == "week":
        rows = await get_week_news(user_id, limit=15)
        header = "Новости за неделю:"
        if category:
            header += f" категория: {category}"
        if query:
            header += f" запрос: {query}"
    else:
        if query:
            rows = await search_news(user_id, days=0, query=query, limit=10)
            header = f"Новости за сегодня по запросу: {query}"
        else:
            rows = await get_today_news(user_id, limit=10)
            header = "Новости за сегодня:"
            if category:
                header += f" категория: {category}"
            if query:
                header += f" запрос: {query}"

    messages = []
    messages.append(header)
    for title, link, summary, published_at in rows:
        clean = clean_html(summary or "")
        short = (clean[:120] + "…") if clean else ""
        messages.append(f"📰 {title}\n{short}\n🔗 {link}")

        text = "\n\n".join(messages)

        # Telegram limit ~4096. Отправляем частями.
        MAX_LEN = 3900
        if len(text) <= MAX_LEN:
            await update.message.reply_text(text)
        else:
            chunk = ""
            for part in text.split("\n\n"):
                # +2 за разделитель
                if len(chunk) + len(part) + 2 > MAX_LEN:
                    if chunk:
                        await update.message.reply_text(chunk)
                    chunk = part
                else:
                    chunk = part if not chunk else (chunk + "\n\n" + part)
            if chunk:
                await update.message.reply_text(chunk)

async def post_init(app: Application) -> None:
    await init_db()

async def handle_menu_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text

    if text in ("📰 Сегодня", "Сегодня"):
        context.args = []
        await cmd_news(update, context)
        return

    if text in ("📅 Неделя", "Неделя"):
        context.args = ["week"]
        await cmd_news(update, context)
        return

    if text in ("📃 Мои источники", "Мои источники"):
        context.args = []
        await cmd_list(update, context)
        return

    if text in ("➕ Добавить RSS", "Добавить RSS"):
        await update.message.reply_text(
            "Пришли ссылку на RSS в формате:\n/add https://example.com/rss"
        )
        return

    if text in ("⚙️ Настройки", "Настройки"):
        await update.message.reply_text("Настройки появятся позже.")
        return

async def list_all_subscriptions() -> list[tuple[int, int, str, str, int, int]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT user_id, id, url, title, disabled, fail_count "
            "FROM subscriptions ORDER BY user_id, id"
        )
        rows = await cur.fetchall()
        return [
            (int(r[0]), int(r[1]), str(r[2]), str(r[3]), int(r[4]), int(r[5]))
            for r in rows
        ]

async def run_sync_cli() -> None:
    start_ts = time.time()
    load_dotenv("/opt/rss_digest_bot/.env")
    log_dir = Path("/opt/rss_digest_bot/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sync.log"

    def log_line(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    async def send_telegram_alert(text: str) -> None:
        token = (os.getenv("SYNC_ALERT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
        chat_id = (os.getenv("SYNC_ALERT_CHAT_ID") or "").strip()
        if not token or not chat_id:
            return  # не настроено — пропускаем

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                await c.post(url, json=payload)
        except Exception:
            return  # алерт не должен ломать sync

    # гарантируем, что БД/таблицы есть
    await init_db()
    
    async def mark_success(sub_id: int) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE subscriptions SET fail_count=0, last_error=NULL WHERE id=?",
                (sub_id,),
            )
            await db.commit()

    async def mark_failure(sub_id: int, err: str) -> int:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "UPDATE subscriptions "
                "SET fail_count=fail_count+1, last_error=? "
                "WHERE id=? "
                "RETURNING fail_count",
                (err[:500], sub_id),
            )
            row = await cur.fetchone()
            await db.commit()
            return int(row[0]) if row else 0

    async def disable_subscription(sub_id: int) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE subscriptions SET disabled=1 WHERE id=?", (sub_id,))
            await db.commit()
   
    total_added = 0
    failed = 0
    subs = await list_all_subscriptions()
    users_count = len({row[0] for row in subs})
    log_line(f"start users={users_count} feeds={len(subs)}")

    if not subs:
        print("No subscriptions to sync.")
        # нет подписок — это не ошибка, завершаемся успешно
        print("No subscriptions to sync.")
        return

    total_added = 0
    failed = 0

    max_fail = int(os.getenv("SYNC_MAX_FAILS", "5"))

    max_fail = int(os.getenv("SYNC_MAX_FAILS", "5"))

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for user_id, sub_id, url, _title, disabled, fail_count in subs:
            if disabled == 1:
                continue

            try:
                r = await client.get(url, headers={"User-Agent": "rss-digest-bot/0.1"})
                r.raise_for_status()
                parsed = feedparser.parse(r.content)

                if not getattr(parsed, "feed", None) or parsed.entries is None:
                    failed += 1
                    new_fc = await mark_failure(sub_id, "invalid feed content")
                    if new_fc >= max_fail:
                        await disable_subscription(sub_id)
                        await send_telegram_alert(
                            f"rss_digest_bot: disabled subscription id={sub_id} after {new_fc} fails: {url}"
                        )
                    continue

                added = await save_entries(user_id, sub_id, url, parsed)
                total_added += added
                await mark_success(sub_id)

            except Exception as e:
                failed += 1
                new_fc = await mark_failure(sub_id, repr(e))
                if new_fc >= max_fail:
                    await disable_subscription(sub_id)
                    await send_telegram_alert(
                        f"rss_digest_bot: disabled subscription id={sub_id} after {new_fc} fails: {url}"
                    )

    duration = time.time() - start_ts
    log_line(f"done new_entries={total_added} errors={failed} duration_sec={duration:.2f}")
    print(f"Sync done. New entries: {total_added}. Errors: {failed}.")
    if failed > 0:
        await send_telegram_alert(
            f"rss_digest_bot: sync errors={failed}, new_entries={total_added} (host={os.uname().nodename})"
        )

    raise SystemExit(1 if failed > 0 else 0)

def main() -> None:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN в .env")

    request = HTTPXRequest(connect_timeout=20.0, read_timeout=60.0, write_timeout=20.0, pool_timeout=20.0)
    app = Application.builder().token(token).request(request).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("del", cmd_del))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("digest_week", cmd_digest_week))

    print("Polling started...")
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_buttons))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        asyncio.run(run_sync_cli())
    else:
        main()

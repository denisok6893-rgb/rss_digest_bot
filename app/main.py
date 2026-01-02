import os
import re
import hashlib
import json
from datetime import datetime, timezone
from typing import Tuple
from telegram.request import HTTPXRequest

import aiosqlite
import feedparser
import httpx
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

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
              AND date(published_at) = date('now')
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
        "RSS Digest Bot\n\n"
        "Команды:\n"
        "/add <rss_url> — добавить канал\n"
        "/list — список каналов\n"
        "/del <id> — удалить канал\n"
        "/sync — скачать новые новости из всех каналов\n"
        "/news — новости за сегодня"
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

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    args = [a.strip().lower() for a in (context.args or []) if a.strip()]
    period = "today"
    query = ""

    if args:
        if args[0] in ("week", "7", "7d"):
            period = "week"
            query = " ".join(args[1:]).strip()
        else:
            query = " ".join(args).strip()

    if period == "week":
        if query:
            rows = await search_news(user_id, days=6, query=query, limit=5)
            header = f"Новости за неделю по запросу: {query}"
        else:
            rows = await get_week_news(user_id, limit=5)
            header = "Новости за неделю:"
    else:
        if query:
            rows = await search_news(user_id, days=0, query=query, limit=10)
            header = f"Новости за сегодня по запросу: {query}"
        else:
            rows = await get_today_news(user_id, limit=10)
            header = "Новости за сегодня:"

    messages = []
    messages.append(header)
    for title, link, summary, published_at in rows:
        clean = clean_html(summary or "")
        short = (clean[:120] + "…") if clean else ""
        messages.append(f"📰 {title}\n{short}\n🔗 {link}")

    text = "\n\n".join(messages)
    await update.message.reply_text(text)

async def post_init(app: Application) -> None:
    await init_db()


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

    print("Polling started...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

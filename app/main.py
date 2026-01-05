import sys
import time
from pathlib import Path
import asyncio
import os
import re
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Tuple
from telegram.request import HTTPXRequest
from openai import AsyncOpenAI
import calendar
from datetime import timezone

import aiosqlite
import feedparser
import httpx
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import ReplyKeyboardMarkup
from app.keyboards.main_menu import main_menu_keyboard
from app.keyboards.settings_menu import settings_menu_keyboard
from telegram import Update, Bot

load_dotenv()

DB_PATH = "data/rss.db"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
CATEGORY_PRESETS = {
    "AI": [
        ("RB.RU (AI)", "https://rb.ru/rss/"),
        ("Google DeepMind Blog", "https://deepmind.googleblog.com/feeds/posts/default"),
        ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ],
    "Разработка": [
        ("Habr / Разработка", "https://habr.com/ru/rss/flows/develop/articles/?fl=ru"),
        ("Dev.to", "https://dev.to/feed/"),
    ],
    "Security": [
        ("The Hacker News", "https://feeds.feedburner.com/TheHackersNews"),
    ],
    "Бизнес": [
        ("TechCrunch", "https://techcrunch.com/feed/"),
    ],
}


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
        # user settings for auto-digest
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                digest_enabled INTEGER NOT NULL DEFAULT 0,
                digest_interval_min INTEGER NOT NULL DEFAULT 120,
                next_digest_at TEXT,
                last_digest_at TEXT
            );
            """
        )

        # ensure columns in subscriptions (for new installs)
        try:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN fail_count INTEGER NOT NULL DEFAULT 0;")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN disabled INTEGER NOT NULL DEFAULT 0;")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN last_error TEXT;")
        except Exception:
            pass

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

async def make_digest_safe(items: list[tuple[str, str, str]], period_label: str) -> str:
    """
    Пытаемся сделать ИИ-сводку. Если API недоступно/баланс=0/ошибка — шлём простой дайджест без ИИ.
    items: (title, summary, link)
    """
    try:
        return await make_digest(items, period_label=period_label)
    except Exception as e:
        # fallback: простой список
        header = f"Новости за {period_label} (без ИИ-сводки: {type(e).__name__})"
        lines = [header, ""]
        for i, (title, summary, link) in enumerate(items, 1):
            s = clean_html(summary or "")
            s = (s[:200] + "…") if len(s) > 200 else s
            lines.append(f"{i}. {title}")
            if s:
                lines.append(s)
            lines.append(link)
            lines.append("")
        return "\n".join(lines).strip()

def normalize_published_at(entry) -> str:
    """
    Приводим дату публикации к ISO-8601, чтобы SQLite date()/datetime() корректно работали.
    Возвращаем строку вида: 2026-01-03T15:16:11+00:00
    """
    # 1) feedparser часто даёт struct_time в published_parsed/updated_parsed
    st = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if st:
        dt = datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
        return dt.isoformat(timespec="seconds")

    # 2) иногда есть строка published/updated (RFC822 или ISO)
    s = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if s:
        # 2a) RFC822 (например arXiv)
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat(timespec="seconds")
        except Exception:
            pass

        # 2b) ISO-8601 (например Habr) — оставим как есть, если парсится
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat(timespec="seconds")
        except Exception:
            pass

    # 3) если даты нет/не распарсилась — пишем текущую (лучше чем пусто)
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

async def save_entries(user_id: int, sub_id: int, url: str, parsed) -> int:
    added = 0

    entries = getattr(parsed, "entries", None) or []
    if not entries:
        return 0

    async with aiosqlite.connect(DB_PATH) as db:
        for entry in entries:
            try:
                title = str(getattr(entry, "title", "") or "").strip()
                link = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()

                try:
                    published_at = normalize_published_at(entry) or ""
                except Exception:
                    published_at = ""

                categories = []
                for t in (getattr(entry, "tags", None) or []):
                    if isinstance(t, dict):
                        term = (t.get("term") or "").strip()
                        if term:
                            categories.append(term)

                guid = str(
                    getattr(entry, "id", "")
                    or getattr(entry, "guid", "")
                    or ""
                ).strip()

                base = (guid or link or title or "") + "|" + published_at
                h = hashlib.sha256(base.encode("utf-8")).hexdigest()

                cur = await db.execute(
                    """
                    INSERT OR IGNORE INTO entries
                    (user_id, subscription_id, guid, title, link, summary, published_at, categories, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(user_id),
                        int(sub_id),
                        guid,
                        title or "(no title)",
                        link or url,
                        summary,
                        published_at,
                        json.dumps(categories, ensure_ascii=False),
                        h,
                    ),
                )

                if cur.rowcount > 0:
                    added += 1

            except Exception:
                # битая запись — пропускаем, sync не ломаем
                continue

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

async def get_user_settings(user_id: int) -> tuple[int, int, str | None, str | None]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT digest_enabled, digest_interval_min, next_digest_at, last_digest_at FROM user_settings WHERE user_id=?",
            (user_id,),
        )
        row = await cur.fetchone()
        if not row:
            await db.execute(
                "INSERT INTO user_settings(user_id, digest_enabled, digest_interval_min) VALUES (?, 0, 120)",
                (user_id,),
            )
            await db.commit()
            return 0, 120, None, None
        return int(row[0]), int(row[1]), row[2], row[3]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _add_minutes_iso(dt_iso: str, minutes: int) -> str:
    dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    return (dt + timedelta(minutes=minutes)).isoformat(timespec="seconds")


async def set_digest_settings(user_id: int, enabled: int, interval_min: int) -> None:
    now = _now_utc_iso()
    next_at = _add_minutes_iso(now, interval_min) if enabled else None
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_settings(user_id, digest_enabled, digest_interval_min, next_digest_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              digest_enabled=excluded.digest_enabled,
              digest_interval_min=excluded.digest_interval_min,
              next_digest_at=excluded.next_digest_at
            """,
            (user_id, int(enabled), int(interval_min), next_at),
        )
        await db.commit()

async def get_today_news(user_id: int, limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND date(created_at, 'localtime') = date('now', 'localtime')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return await cur.fetchall()

async def get_week_news(user_id: int, limit: int = 15):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND datetime(created_at) >= datetime('now', '-7 days')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return await cur.fetchall()

async def get_last_hours_news(user_id: int, hours: int = 24, limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT title, link, summary, published_at
            FROM entries
            WHERE user_id = ?
              AND datetime(created_at) >= datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, f"-{int(hours)} hours", limit),
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
    text = await make_digest_safe(items, period_label="сегодня")
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
            
            if not rows:
                await update.message.reply_text(
                    "📰 Сегодня новых новостей нет.\n"
                    "Попробуйте позже или нажмите «📅 Неделя»."
                )
                
                return
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

        # Telegram limit ~4096, берём запас
        MAX_LEN = 3800

        if len(text) <= MAX_LEN:
            await update.message.reply_text(text)
        else:
            chunk = ""
            for part in text.split("\n\n"):
                part2 = part + "\n\n"
                if len(chunk) + len(part2) > MAX_LEN:
                    await update.message.reply_text(chunk.strip())
                    chunk = part2
                else:
                    chunk += part2
            if chunk.strip():
                await update.message.reply_text(chunk.strip())

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
        user_id = update.effective_user.id
        enabled, interval_min, next_at, last_at = await get_user_settings(user_id)
        status = "включен" if enabled else "выключен"
        await update.message.reply_text(
            f"⚙️ Настройки\n\nАвто-дайджест: {status}\nИнтервал: {interval_min} мин\nСледующая отправка: {next_at or '—'}",
            reply_markup=settings_menu_keyboard(),
        )
        return

    if text == "⏱ Авто-дайджест: 1 час":
        await set_digest_settings(update.effective_user.id, 1, 60)
        await update.message.reply_text("Ок: авто-дайджест каждые 1 час.", reply_markup=main_menu_keyboard())
        return

    if text == "⏱ Авто-дайджест: 2 часа":
        await set_digest_settings(update.effective_user.id, 1, 120)
        await update.message.reply_text("Ок: авто-дайджест каждые 2 часа.", reply_markup=main_menu_keyboard())
        return

    if text == "⏱ Авто-дайджест: 4 часа":
        await set_digest_settings(update.effective_user.id, 1, 240)
        await update.message.reply_text("Ок: авто-дайджест каждые 4 часа.", reply_markup=main_menu_keyboard())
        return

    if text == "⛔️ Авто-дайджест: выключить":
        await set_digest_settings(update.effective_user.id, 0, 120)
        await update.message.reply_text("Ок: авто-дайджест выключен.", reply_markup=main_menu_keyboard())
        return

    if text == "⬅️ Назад":
        await update.message.reply_text("Главное меню.", reply_markup=main_menu_keyboard())
        return

    if text == "🗂 Категории":
        kb = [[f"📌 {name}"] for name in CATEGORY_PRESETS.keys()]
        kb.append(["⬅️ Назад"])
        await update.message.reply_text(
            "Выберите категорию. Я подключу набор RSS, лишнее можно удалить в «📃 Мои источники».",
            reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True),
        )
        return

    if text.startswith("📌 "):
        category_name = text.replace("📌 ", "", 1).strip()
        items = CATEGORY_PRESETS.get(category_name)
        if not items:
            await update.message.reply_text(
                "Неизвестная категория.",
                reply_markup=main_menu_keyboard(),
            )
            return

        added = 0
        skipped = 0
        for title, url in items:
            try:
                status = await add_subscription(user_id, url, title)
                if "✅" in status:
                    added += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        await update.message.reply_text(
            f"Категория «{category_name}» подключена.\n"
            f"Добавлено: {added}\n"
            f"Пропущено (уже было/ошибка): {skipped}",
            reply_markup=main_menu_keyboard(),
        )
        return

    if text == "⬅️ Назад":
        await update.message.reply_text(
            "Главное меню.",
            reply_markup=main_menu_keyboard(),
        )
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

    async def get_top_failures(limit: int = 5) -> list[tuple[int, str, int, str | None]]:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "SELECT id, url, fail_count, last_error "
                "FROM subscriptions "
                "WHERE disabled=0 AND fail_count>0 "
                "ORDER BY fail_count DESC, id DESC "
                "LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
            out: list[tuple[int, str, int, str | None]] = []
            for r in rows:
                out.append((int(r[0]), str(r[1]), int(r[2]), (str(r[3]) if r[3] is not None else None)))
            return out

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
    # Дедуп алертов: шлём только при смене состояния ошибок
    state_file = log_dir / "sync_alert_state.json"
    now_ts = int(time.time())
    prev = {"had_errors": None, "last_sent_ts": 0}

    try:
        if state_file.exists():
            prev = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        prev = {"had_errors": None, "last_sent_ts": 0}

    had_errors = failed > 0
    prev_had = prev.get("had_errors", None)
    last_sent = int(prev.get("last_sent_ts", 0) or 0)

    # Шлём если изменилось состояние (0->1 или 1->0) или если давно не слали (раз в 6 часов)
    force_interval_sec = 6 * 3600
    should_send = (prev_had is None) or (had_errors != prev_had) or (now_ts - last_sent >= force_interval_sec)

    if should_send:
        if had_errors:
            top = await get_top_failures(5)
            lines = [
                f"rss_digest_bot: sync errors={failed}, new_entries={total_added} (host={os.uname().nodename})"
            ]
            if top:
                lines.append("Top failed feeds:")
                for sid, url, fc, err in top:
                    e = (err or "—")
                    if len(e) > 160:
                        e = e[:160] + "…"
                    lines.append(f"- id={sid} fails={fc}\n  {url}\n  err={e}")
            await send_telegram_alert("\n".join(lines))

        else:
            await send_telegram_alert(
                f"rss_digest_bot: sync OK, new_entries={total_added} (host={os.uname().nodename})"
            )

        try:
            state_file.write_text(
                json.dumps({"had_errors": had_errors, "last_sent_ts": now_ts}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    raise SystemExit(1 if failed > 0 else 0)

async def send_long_message(bot: Bot, chat_id: int, text: str, max_len: int = 3800) -> None:
    """
    Отправляет длинный текст частями, стараясь резать по абзацам.
    Если один абзац слишком длинный — режет по символам.
    """
    if not text:
        return

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunk = ""

    async def _flush(c: str) -> None:
        c = c.strip()
        if not c:
            return
        await bot.send_message(chat_id=chat_id, text=c, disable_web_page_preview=True)
        await asyncio.sleep(0.2)  # небольшой анти-флуд

    for p in parts:
        candidate = (chunk + "\n\n" + p).strip() if chunk else p

        if len(candidate) <= max_len:
            chunk = candidate
            continue

        # если текущий chunk уже есть — отправляем его
        if chunk:
            await _flush(chunk)
            chunk = ""

        # если абзац сам по себе длиннее лимита — режем его по кускам
        if len(p) > max_len:
            start = 0
            while start < len(p):
                await _flush(p[start:start + max_len])
                start += max_len
        else:
            chunk = p

    if chunk:
        await _flush(chunk)

async def run_digest_tick_cli() -> None:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN в .env")

    await init_db()

    now_iso = _now_utc_iso()

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT user_id, digest_interval_min, next_digest_at, last_digest_at
            FROM user_settings
            WHERE digest_enabled=1
              AND (next_digest_at IS NULL OR datetime(next_digest_at) <= datetime('now'))
            """
        )
        users = await cur.fetchall()

    if not users:
        return

    bot = Bot(token=token)

    for (user_id, interval_min, next_at, last_at) in users:
        user_id = int(user_id)
        interval_min = int(interval_min)

        rows = await get_last_hours_news(user_id, hours=24, limit=10)
        if not rows:
            # всё равно двигаем next_digest_at, чтобы не спамить пустыми
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE user_settings SET next_digest_at=?, last_digest_at=COALESCE(last_digest_at, ?) WHERE user_id=?",
                    (_add_minutes_iso(now_iso, interval_min), now_iso, user_id),
                )
                await db.commit()
            continue

        items = [(r[0], r[2], r[1]) for r in rows]
        text = await make_digest_safe(items, period_label="сегодня")

        # Telegram лимит
        await send_long_message(bot, user_id, text, max_len=3800)

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE user_settings SET last_digest_at=?, next_digest_at=? WHERE user_id=?",
                (now_iso, _add_minutes_iso(now_iso, interval_min), user_id),
            )
            await db.commit()

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
    elif len(sys.argv) > 1 and sys.argv[1] == "digest_tick":
        asyncio.run(run_digest_tick_cli())
    else:
        main()

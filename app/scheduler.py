import os
import asyncio
from dotenv import load_dotenv

import os
os.environ["DB_PATH"] = "data/rss.db"

from app.main import (
    init_db,
    get_subscriptions,
    fetch_and_parse_feed,
    save_entries,
    get_today_news,
    make_digest,
)

from telegram import Bot


async def run_scheduler():
    load_dotenv()

    BOT_TOKEN = os.getenv("BOT_TOKEN")
    ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

    if not BOT_TOKEN or not ADMIN_CHAT_ID:
        print("BOT_TOKEN or ADMIN_CHAT_ID not set")
        return

    bot = Bot(token=BOT_TOKEN)

    print("scheduler started")

    # 1. init db
    await init_db()

    # 2. sync all feeds (user_id = 1, как в боте)
    user_id = int(ADMIN_CHAT_ID)
    subs = await get_subscriptions(user_id)

    new_entries = 0
    for sub_id, url, _ in subs:
        feed_title, added = await fetch_and_parse_feed(url)
        new_entries += added
        # save_entries вызывается внутри fetch_and_parse_feed → у тебя так и есть

    print(f"sync done, new entries: {new_entries}")

    # 3. get today news
    from app.main import get_week_news
    items = await get_today_news(user_id, limit=20)
    print(f"entries found: {len(items)}")

    if not items:
        return

    # привести к формату (title, summary, link)
    digest_items = [
        (title, summary, link)
        for (_, title, summary, link, *_) in items
    ]

    digest_text = await make_digest(digest_items, period_label="today")

    # 5. send message
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text=digest_text)

    print("digest sent")


if __name__ == "__main__":
    asyncio.run(run_scheduler())

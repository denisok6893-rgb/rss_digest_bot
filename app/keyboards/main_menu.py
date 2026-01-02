from telegram import ReplyKeyboardMarkup, KeyboardButton

def main_menu_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("📰 Сегодня"), KeyboardButton("📅 Неделя")],
            [KeyboardButton("➕ Добавить RSS"), KeyboardButton("📃 Мои источники")],
            [KeyboardButton("⚙️ Настройки")],
        ],
        resize_keyboard=True,
    )

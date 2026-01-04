from telegram import ReplyKeyboardMarkup, KeyboardButton

def settings_menu_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("⏱ Авто-дайджест: 1 час"), KeyboardButton("⏱ Авто-дайджест: 2 часа")],
            [KeyboardButton("⏱ Авто-дайджест: 4 часа")],
            [KeyboardButton("⛔️ Авто-дайджест: выключить")],
            [KeyboardButton("⬅️ Назад")],
        ],
        resize_keyboard=True,
    )

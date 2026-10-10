from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Справка"), KeyboardButton(text="Об игре")],
        [KeyboardButton(text="О боте"), KeyboardButton(text="Контакты")],
        [KeyboardButton(text="Настройки")],
    ],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Задай вопрос по игре «Берсерк»",
)

ANSWER_FEEDBACK = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="👍", callback_data="feedback:like"),
            InlineKeyboardButton(text="👎", callback_data="feedback:dislike"),
        ]
    ],
)

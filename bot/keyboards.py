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


def answer_feedback_keyboard(answer_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍", callback_data=f"feedback:{answer_id}:like"),
                InlineKeyboardButton(text="👎", callback_data=f"feedback:{answer_id}:dislike"),
            ]
        ]
    )

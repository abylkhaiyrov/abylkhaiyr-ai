from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.evaluation.comparison import FIXED_PARAMS, load_engine
from llm.tokenizer.bpe import Message

PROMPTS: tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], ...] = (
    ("req_howare_ru", "ru", "Как дела?", ()),
    ("req_mood_ru", "ru", "Привет, как настроение?", ()),
    ("req_abilities_ru", "ru", "Что ты умеешь делать?", ()),
    ("req_help_ru", "ru", "Чем ты можешь мне помочь?", ()),
    ("req_about_ru", "ru", "Расскажи немного о себе.", ()),
    ("req_ai_ru", "ru", "Что такое искусственный интеллект?", ()),
    ("req_java_python_ru", "ru", "Объясни разницу между Java и Python.", ()),
    ("req_rain_ru", "ru", "Почему идёт дождь?", ()),
    ("req_mult_ru", "ru", "Сколько будет 25 умножить на 48?", ()),
    ("req_sort_ru", "ru", "Напиши функцию сортировки.", ()),
    (
        "req_simpler_ru",
        "ru",
        "Объясни предыдущий ответ проще.",
        (
            (
                "Что такое фотосинтез?",
                "Фотосинтез — процесс, при котором растения превращают свет в энергию.",
            ),
        ),
    ),
    ("req_translate_kk", "ru", "Переведи это на казахский язык: Я иду в школу.", ()),
    ("req_howare_en", "en", "How are you doing today?", ()),
    ("req_abilities_en", "en", "What are you able to do?", ()),
    ("req_ai_en", "en", "What is artificial intelligence?", ()),
    ("req_rain_en", "en", "Why does it rain?", ()),
    ("req_mult_en", "en", "What is 25 times 48?", ()),
    ("req_howare_kk", "kk", "Қалың қалай?", ()),
    ("req_abilities_kk", "kk", "Сен не істей аласың?", ()),
    ("req_ai_kk", "kk", "Жасанды интеллект дегеніміз не?", ()),
    ("req_rain_kk", "kk", "Жаңбыр неге жауады?", ()),
    ("req_para_ru", "ru", "Ну как ты там вообще, всё хорошо?", ()),
    ("req_para2_ru", "ru", "Расскажи, в чём тебе можно довериться?", ()),
    ("natural_en", "en", "Hey, I'm a bit bored today. Can you suggest something to do?", ()),
    ("natural_ru", "ru", "Привет! Мне сегодня скучно, предложи, чем заняться.", ()),
    ("natural_kk", "kk", "Сәлем! Бүгін зерігіп отырмын, не істесем екен?", ()),
    ("simple_q_en", "en", "How many legs does an insect have?", ()),
    ("simple_q_ru", "ru", "Сколько месяцев в году?", ()),
    ("simple_q_kk", "kk", "Күн жүйесінде қанша планета бар?", ()),
    ("negation_en", "en", "Name a fruit that is not red.", ()),
    ("negation_ru", "ru", "Назови цвет, который не является тёплым.", ()),
    ("typo_en", "en", "waht is teh captial of Kazakhstna?", ()),
    ("typo_ru", "ru", "скоко будет двапдцать плюс тры?", ()),
    ("clarify_en", "en", "Make it better.", ()),
    ("clarify_ru", "ru", "Сделай как в прошлый раз.", ()),
    ("explain_en", "en", "Explain what an electric circuit is to a ten year old.", ()),
    ("explain_ru", "ru", "Объясни, почему небо голубое.", ()),
    ("explain_kk", "kk", "Желтоқсан айында неге қар жауады?", ()),
    ("translate_en_ru", "en", "Translate to Russian: The weather is nice today.", ()),
    ("translate_ru_en", "ru", "Переведи на английский: Я люблю читать книги.", ()),
    ("translate_kk_ru", "kk", "Мынаны орысшаға аудар: Мен мектепке барамын.", ()),
    ("math_en", "en", "I have 3 boxes with 12 pencils each. How many pencils do I have?", ()),
    ("math_ru", "ru", "В классе 28 учеников, 9 из них отсутствуют. Сколько пришло?", ()),
    ("code_en", "en", "Write a Python function that returns the largest of three numbers.", ()),
    ("code_fix_en", "en", "Fix this code: def add(a, b) return a + b", ()),
    (
        "analysis_en",
        "en",
        "Text: The shop opens at 8 and closes at 20, except on Sunday when it closes at 14. When does it close on Sunday?",
        (),
    ),
    ("unknown_en", "en", "What is the name of my best friend's dog?", ()),
    ("unknown_ru", "ru", "Какая завтра будет температура на улице?", ()),
    (
        "memory_en",
        "en",
        "What city did I say I live in?",
        (("I live in Taraz and I work as a teacher.", "Nice to meet you!"),),
    ),
    (
        "memory_ru",
        "ru",
        "Как зовут мою сестру?",
        (("Мою сестру зовут Алия, ей двадцать лет.", "Приятно познакомиться с вами обеих."),),
    ),
    (
        "followup_en",
        "en",
        "And what is that multiplied by 3?",
        (("What is 12 + 8?", "12 + 8 = 20."),),
    ),
    (
        "long_en",
        "en",
        "Here is a long note: "
        + " ".join(f"Item {i} costs {i * 3} dollars." for i in range(1, 25))
        + " How much does item 17 cost?",
        (),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="manual_review")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    rows = []
    for case_id, language, prompt, history in PROMPTS:
        messages = [
            m
            for user, assistant in history
            for m in (Message("user", user), Message("assistant", assistant))
        ]
        messages.append(Message("user", prompt))
        generation = engine.chat(messages, FIXED_PARAMS)
        rows.append(
            {
                "id": case_id,
                "language": language,
                "prompt": prompt,
                "response": generation.text.strip(),
                "finished": generation.finish_reason == "stop",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(rows)} responses written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

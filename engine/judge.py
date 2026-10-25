"""Судья «та же идея?» (prompts/judge.txt) на util-модели.

Эмбеддинги путают слова с механикой: «агент в Docker выполняет README» и «агенты-персонажи
проходят сайт» люди в pairs.csv считают одной идеей, а косинус ≈ 0. Поэтому эмбеддинги
только отбирают кандидатов, решает судья."""
import asyncio

from engine import llm

VERDICTS = ("да", "частично", "нет")


async def same(a: str, b: str) -> str:
    # без рассуждения Haiku отвечает наугад (почти дословные пары называл «нет»), с low — стабильно
    r = await llm.complete(llm.UTIL_MODEL, llm.prompt("judge", a=a, b=b), temperature=0, max_tokens=1500,
                           reasoning="low")
    word = r.strip(" .!«»\"'\n").lower().split()[0] if r.strip() else ""
    return word if word in VERDICTS else "нет"


async def many(a: str, bs: list[str]) -> list[str]:
    """Единичный сбой — «нет» для этой пары; если упала четверть и больше — исключение,
    вызывающий перейдёт на порог."""
    out = await asyncio.gather(*(same(a, b) for b in bs), return_exceptions=True)
    errors = [x for x in out if isinstance(x, Exception)]
    if errors and len(errors) * 4 >= len(out):
        raise errors[0]
    return [x if isinstance(x, str) else "нет" for x in out]

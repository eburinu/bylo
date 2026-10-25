"""Судья «та же идея?» (prompts/judge.txt) на util-модели.

Эмбеддинги путают слова с механикой: «агент в Docker выполняет README» и «агенты-персонажи
проходят сайт» люди в pairs.csv считают одной идеей, а косинус ≈ 0. Поэтому эмбеддинги
только отбирают кандидатов, решает судья."""
import asyncio

from engine import llm

VERDICTS = ("да", "частично", "нет")


async def same(a: str, b: str) -> str:
    r = await llm.complete(llm.UTIL_MODEL, llm.prompt("judge", a=a, b=b), temperature=0, max_tokens=5)
    word = r.strip(" .!«»\"'\n").lower().split()[0] if r.strip() else ""
    return word if word in VERDICTS else "нет"


async def many(a: str, bs: list[str]) -> list[str]:
    return await asyncio.gather(*(same(a, b) for b in bs))

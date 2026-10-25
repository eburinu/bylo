"""Идея → механика одной фразой (prompts/normalize.txt, normalize_user.txt) через util-модель."""
from engine import llm


async def mechanic(title: str, pitch: str) -> str:
    text = llm.prompt("normalize", title=title, pitch=pitch)
    return (await llm.complete(llm.UTIL_MODEL, text, temperature=0, max_tokens=80)).strip(" «»\"'\n")


async def user_idea(text: str) -> dict:
    """Свободный текст пользователя → {"title", "mechanic"}."""
    r = await llm.complete_json(llm.UTIL_MODEL, llm.prompt("normalize_user", text=text),
                                temperature=0, max_tokens=120)
    return {"title": str(r.get("title", "")).strip(), "mechanic": str(r.get("mechanic", "")).strip()}

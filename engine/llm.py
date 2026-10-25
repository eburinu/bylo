"""Тонкая обёртка над OpenAI-совместимым API (OpenRouter и т.п.).

Всё берётся из .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODELS, LLM_UTIL_MODEL.
Проверка ключа и моделей: python -m engine.llm
"""
import asyncio
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]
UTIL_MODEL = os.getenv("LLM_UTIL_MODEL", "").strip() or (MODELS[0] if MODELS else "")
CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "12"))
RETRIES = 5

_client: AsyncOpenAI | None = None
_sems: dict[int, asyncio.Semaphore] = {}  # семафор на event loop: asyncio.run можно звать много раз


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not os.getenv("LLM_API_KEY"):
            raise RuntimeError("LLM_API_KEY пуст — впиши ключ в .env")
        _client = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"),
                              timeout=90, max_retries=0)
    return _client


def _sem() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _sems:
        _sems[loop_id] = asyncio.Semaphore(CONCURRENCY)
    return _sems[loop_id]


def prompt(name: str, **kw) -> str:
    """Подставляет {ключи} в prompts/<name>.txt. Не str.format: в шаблонах есть фигурные скобки JSON."""
    text = (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    return text


async def chat(model: str, messages: list[dict], temperature: float = 1.0, max_tokens: int = 2000) -> str:
    for attempt in range(RETRIES):
        try:
            async with _sem():
                r = await client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            text = (r.choices[0].message.content or "").strip() if r.choices else ""
            if text:
                return text
            err: Exception = RuntimeError(f"{model}: пустой ответ")
        except APIStatusError as e:
            if e.status_code != 429 and e.status_code < 500:
                raise
            err = e
        except (APIConnectionError, APITimeoutError) as e:
            err = e
        await asyncio.sleep(min(30, 2 ** attempt + random.random()))
    raise err


async def complete(model: str, text: str, **kw) -> str:
    return await chat(model, [{"role": "user", "content": text}], **kw)


def parse_json(text: str):
    """Достаёт JSON из ответа: снимает ```-ограды и болтовню вокруг."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError("в ответе нет JSON")
    return json.loads(m.group(0))


async def complete_json(model: str, text: str, **kw):
    """Как complete, но возвращает разобранный JSON; при мусоре один раз просит «только JSON»."""
    messages = [{"role": "user", "content": text}]
    answer = await chat(model, messages, **kw)
    try:
        return parse_json(answer)
    except ValueError:
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": "Верни только валидный JSON, без пояснений и без ```."}]
        return parse_json(await chat(model, messages, **kw))


async def _ping():
    models = list(dict.fromkeys(MODELS + [UTIL_MODEL]))
    if not models:
        print("LLM_MODELS и LLM_UTIL_MODEL пусты — заполни .env")
        return

    async def one(m):
        try:
            r = await complete_json(m, 'Ответь строго JSON: {"ok": true, "model": "<твоё название>"}',
                                    temperature=0, max_tokens=50)
            return m, "ok", r
        except Exception as e:  # noqa: BLE001 — это диагностика
            return m, "FAIL", f"{type(e).__name__}: {e}"[:200]

    for m, status, info in await asyncio.gather(*(one(m) for m in models)):
        print(f"{status:4}  {m:45}  {info}")


if __name__ == "__main__":
    asyncio.run(_ping())

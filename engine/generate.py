"""Генератор соседей: модели × персоны × тон → data/pool/<brief>.json.

python -m engine.generate --brief fkn2026 --n 20
Дописывает пул по одному соседу: можно прервать и запустить снова с большим --n.
"""
import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path

from engine import llm, normalize

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool"
TONE_PROMPT = {"formal": "generator", "casual": "generator_casual", "lowercase_short": "generator_short"}


def load_pool(brief: str) -> list[dict]:
    path = POOL / f"{brief}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_pool(brief: str, ideas: list[dict]):
    POOL.mkdir(parents=True, exist_ok=True)
    tmp = POOL / f"{brief}.json.tmp"
    tmp.write_text("[\n" + ",\n".join(json.dumps(i, ensure_ascii=False) for i in ideas) + "\n]\n", encoding="utf-8")
    tmp.replace(POOL / f"{brief}.json")


def plan(n: int, models: list[str], personas: list[dict]) -> list[tuple[str, str, dict]]:
    """n соседей: (sim_id, модель, персона). Перебор модель×персона в перемешанном порядке —
    так любые первые 20 уже покрывают все семейства."""
    combos = list(itertools.product(models, personas))
    random.Random(4).shuffle(combos)
    return [(f"s{i:03d}", *combos[i % len(combos)]) for i in range(n)]


def brief_prompt(brief_text: str, persona: dict, k: int) -> str:
    tone = persona.get("tone", "formal")
    text = brief_text.lower() if tone == "lowercase_short" else brief_text
    return llm.prompt(TONE_PROMPT[tone], persona=persona["text"], brief=text, k=k).strip()


async def one_sim(sim_id, model, persona, brief, brief_text, k) -> list[dict]:
    r = await llm.complete_json(model, brief_prompt(brief_text, persona, k), temperature=1.0, max_tokens=2500)
    raw = r.get("ideas", []) if isinstance(r, dict) else r
    ideas = []
    for x in raw[:k]:
        if not isinstance(x, dict) or not x.get("title"):
            continue
        try:
            theme = int(x.get("theme") or 0)
        except (TypeError, ValueError):
            theme = 0
        ideas.append({"sim_id": sim_id, "model": model, "persona": persona["id"], "brief": brief,
                      "title": str(x["title"]).strip(), "theme": theme,
                      "pitch": str(x.get("pitch", "")).strip(), "demo": str(x.get("demo", "")).strip()})
    mechs = await asyncio.gather(*(normalize.mechanic(i["title"], i["pitch"]) for i in ideas))
    for i, m in zip(ideas, mechs):
        i["mechanic"] = m
    return ideas


async def run(brief: str, n: int, k: int):
    brief_text = (ROOT / "data" / "briefs" / f"{brief}.md").read_text(encoding="utf-8").strip()
    personas = json.loads((ROOT / "data" / "personas.json").read_text(encoding="utf-8"))
    if not llm.MODELS:
        raise SystemExit("LLM_MODELS пуст — заполни .env")
    pool = load_pool(brief)
    done = {i["sim_id"] for i in pool}
    todo = [p for p in plan(n, llm.MODELS, personas) if p[0] not in done]
    print(f"в пуле {len(done)} соседей, генерирую ещё {len(todo)}")
    lock = asyncio.Lock()
    failed = 0

    async def worker(sim_id, model, persona):
        nonlocal failed
        try:
            ideas = await one_sim(sim_id, model, persona, brief, brief_text, k)
        except Exception as e:  # noqa: BLE001 — один упавший сосед не валит прогон
            failed += 1
            print(f"FAIL {sim_id} {model} {persona['id']}: {type(e).__name__}: {str(e)[:120]}")
            return
        async with lock:
            pool.extend(ideas)
            save_pool(brief, pool)
            print(f"{sim_id} {model:35} {persona['id']:12} {len(ideas)} идей  (всего соседей {len({i['sim_id'] for i in pool})})")

    await asyncio.gather(*(worker(*p) for p in todo))
    print(f"готово: {len(pool)} идей, {len({i['sim_id'] for i in pool})} соседей, упало {failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="fkn2026")
    ap.add_argument("--n", type=int, default=20, help="сколько соседей должно быть в пуле")
    ap.add_argument("--k", type=int, default=5, help="идей на соседа")
    a = ap.parse_args()
    asyncio.run(run(a.brief, a.n, a.k))

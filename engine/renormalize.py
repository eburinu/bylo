"""Переписать механики пула одной util-моделью (prompts/normalize.txt).

Бесплатные соседи писали механику сами (--inline): разный стиль, часть по-английски.
Эмбеддинги тогда кластеризуют стиль и язык, а не идею. Старая механика остаётся
в поле mechanic_raw.

python -m engine.renormalize [--brief fkn2026] [--only-raw]
"""
import argparse
import asyncio

from engine import generate, normalize


async def run(brief: str, only_raw: bool, free: bool = False):
    pool = generate.load_pool(brief)
    todo = [i for i in pool if (not only_raw or "mechanic_raw" not in i)
            and (not free or i["model"].endswith(":free"))]
    print(f"переписываю {len(todo)} механик из {len(pool)}")
    done = failed = 0

    async def one(i):
        nonlocal done, failed
        try:
            m = await normalize.mechanic(i["title"], i.get("pitch", ""))
        except Exception:  # noqa: BLE001 — оставляем старую механику
            failed += 1
            return
        i.setdefault("mechanic_raw", i["mechanic"])
        i["mechanic"] = m
        done += 1

    await asyncio.gather(*(one(i) for i in todo))
    generate.save_pool(brief, pool)
    print(f"готово: {done}, упало {failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="fkn2026")
    ap.add_argument("--only-raw", action="store_true", help="только те, что ещё не переписаны")
    ap.add_argument("--free", action="store_true", help="только соседи на :free моделях (писали механику сами)")
    a = ap.parse_args()
    asyncio.run(run(a.brief, a.only_raw, a.free))

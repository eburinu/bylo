"""Метрика покрытия: какая доля реальных проектов хакатона нашлась бы у наших соседей.

python -m validation.coverage --real <id> --pool <бриф этого хакатона> --control <чужой бриф>

Покрыт — у проекта есть хотя бы одна идея соседа с похожестью ≥ порога (тот же порог,
что у /api/check). Контроль: соседи по *чужому* брифу должны покрывать заметно меньше,
иначе покрытие — просто «все идеи про ИИ похожи».
Честность сравнения:
- оба пула сравниваются на одинаковом числе соседей (подвыборки, --reps повторов),
  потому что покрытие растёт с размером пула;
- центрирование эмбеддингов общее для обоих пулов (среднее их средних), чтобы ни один
  пул не получил преимущества от «своей» системы координат.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from engine import embed_cluster as ec
from engine import llm, normalize
from engine.api import load_ideas, threshold

ROOT = Path(__file__).resolve().parent.parent
VALID = ROOT / "data" / "validation"
CACHE = ROOT / "data" / "cache"
CURVE = (5, 10, 20, 50, 100)


def load_real(vid: str) -> list[dict]:
    path = VALID / f"{vid}.json"
    if not path.exists():
        raise SystemExit(f"нет {path.relative_to(ROOT)} — сначала python -m validation.devpost")
    return [p for p in json.loads(path.read_text(encoding="utf-8")) if p.get("title")]


async def _mechanics(projects: list[dict], cache: dict) -> None:
    async def one(p):
        key = p.get("url") or p["title"]
        if key not in cache:
            try:
                cache[key] = await normalize.mechanic(p["title"], p.get("tagline", ""))
            except Exception as e:  # noqa: BLE001 — один проект не валит прогон
                print(f"FAIL {p['title']}: {type(e).__name__}", file=sys.stderr)
    await asyncio.gather(*(one(p) for p in projects))


def real_texts(vid: str, projects: list[dict]) -> tuple[list[str], bool]:
    """Механики реальных проектов (как у пула: prompts/normalize.txt), кэш в data/cache/.
    Без ключа — «название. tagline»: по стилю это не механика, цифра будет занижена."""
    path = CACHE / f"valmech_{vid}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if llm.UTIL_MODEL and os.getenv("LLM_API_KEY"):
        asyncio.run(_mechanics(projects, cache))
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    texts, normalized = [], True
    for p in projects:
        m = cache.get(p.get("url") or p["title"])
        if not m:
            normalized = False
            m = f"{p['title']}. {p.get('tagline', '')}".strip(" .")
        texts.append(m)
    return texts, normalized


class Pool:
    def __init__(self, brief: str):
        self.brief = brief
        self.ideas = load_ideas(brief)
        if not self.ideas:
            raise SystemExit(f"нет пула для брифа {brief} — python -m engine.generate --brief {brief}")
        self.raw = ec.embed_cached([i["mechanic"] for i in self.ideas], f"emb_{brief}")
        self.sim_ids = sorted({i["sim_id"] for i in self.ideas})
        idx = {s: n for n, s in enumerate(self.sim_ids)}
        self.owner = np.array([idx[i["sim_id"]] for i in self.ideas])

    def best_per_sim(self, real: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """real × соседи: лучшая похожесть среди идей каждого соседа."""
        s = ec.center(real, mean) @ ec.center(self.raw, mean).T
        out = np.full((len(real), len(self.sim_ids)), -1.0)
        for j, o in enumerate(self.owner):
            np.maximum(out[:, o], s[:, j], out=out[:, o])
        return out


def coverage(best: np.ndarray, t: float, n: int, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Средняя и разброс доли покрытых проектов на подвыборках по n соседей."""
    k = best.shape[1]
    if n >= k:
        return float((best.max(1) >= t).mean()), 0.0
    vals = [(best[:, rng.choice(k, n, replace=False)].max(1) >= t).mean() for _ in range(reps)]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="id в data/validation/")
    ap.add_argument("--pool", required=True, help="бриф этого хакатона (data/pool/<бриф>.json)")
    ap.add_argument("--control", required=True, help="бриф чужого хакатона — контроль")
    ap.add_argument("--threshold", type=float, help="по умолчанию — как у /api/check")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--show", type=int, default=0, help="напечатать N проектов с лучшим совпадением")
    a = ap.parse_args()

    t = a.threshold if a.threshold is not None else threshold()
    projects = load_real(a.real)
    texts, normalized = real_texts(a.real, projects)
    real = ec.embed_cached(texts, f"emb_val_{a.real}")
    main_pool, ctrl = Pool(a.pool), Pool(a.control)
    mean = (main_pool.raw.mean(0) + ctrl.raw.mean(0)) / 2
    bm, bc = main_pool.best_per_sim(real, mean), ctrl.best_per_sim(real, mean)
    n = min(bm.shape[1], bc.shape[1])
    rng = np.random.default_rng(4)

    cm, sm = coverage(bm, t, n, a.reps, rng)
    cc, sc = coverage(bc, t, n, a.reps, rng)
    curve = [{"n_sims": k, "pool": round(coverage(bm, t, k, a.reps, rng)[0], 3),
              "control": round(coverage(bc, t, k, a.reps, rng)[0], 3)} for k in CURVE if k < n] + \
            [{"n_sims": n, "pool": round(cm, 3), "control": round(cc, 3)}]

    print(f"{len(projects)} реальных проектов «{a.real}», порог {t:.2f}"
          + ("" if normalized else "  [без нормализации: проекты как «название. tagline» — цифры занижены]"))
    print(f"пул «{a.pool}»: {bm.shape[1]} соседей, контроль «{a.control}»: {bc.shape[1]} — сравниваем по {n}")
    print(f"покрытие своим брифом:  {cm:.1%} ± {sm:.1%}")
    print(f"покрытие чужим брифом:  {cc:.1%} ± {sc:.1%}")
    print(f"разница: {cm - cc:+.1%}" + ("   ← контроль не отличается, цифре верить нельзя" if cm - cc < 2 * max(sm, sc, 0.02) else ""))
    print("кривая (соседей → покрытие свой / чужой): "
          + ", ".join(f"{c['n_sims']}: {c['pool']:.0%}/{c['control']:.0%}" for c in curve))

    if a.show:
        s = ec.center(real, mean) @ ec.center(main_pool.raw, mean).T
        for r in np.argsort(-s.max(1))[:a.show]:
            j = int(np.argmax(s[r]))
            print(f"  {s[r, j]:.2f}  {projects[r]['title']}  ↔  {main_pool.ideas[j]['title']}: {main_pool.ideas[j]['mechanic']}")

    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:8]
    out = CACHE / f"coverage_{a.real}__{a.pool}__{a.control}_{key}.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "real": a.real, "pool": a.pool, "control": a.control, "n_projects": len(projects),
        "normalized": normalized, "threshold": t, "n_sims": n, "reps": a.reps,
        "coverage": round(cm, 3), "coverage_std": round(sm, 3),
        "control_coverage": round(cc, 3), "control_std": round(sc, 3), "curve": curve,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

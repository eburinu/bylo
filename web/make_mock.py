"""Моки для страницы без сервера: web/mock.js из seed и закэшированной карты.

python web/make_mock.py
Страница берёт моки, только если /api/map не ответил (открыта как файл или API лежит).
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "seed" / "sims_2026-10-25.json"


def main():
    ideas = json.loads(SEED.read_text(encoding="utf-8"))
    maps = sorted((ROOT / "data" / "cache").glob("map_fkn2026_*.json"), key=lambda p: p.stat().st_mtime)
    m = next((json.loads(p.read_text(encoding="utf-8")) for p in reversed(maps)
              if len(json.loads(p.read_text(encoding="utf-8"))["labels"]) == len(ideas)), None)
    if m is None:
        raise SystemExit("нет data/cache/map_fkn2026_*.json под seed — подними API на seed один раз")
    sims = defaultdict(set)
    for i, c in zip(ideas, m["labels"]):
        sims[c].add(i["sim_id"])
    n_sims = len({i["sim_id"] for i in ideas})
    # та же форма, что у GET /api/map (docs/SCHEMA.md)
    mock_map = {
        "n_sims": n_sims,
        "points": [{"x": round(x, 3), "y": round(y, 3), "cluster": c, "sim_id": i["sim_id"], "title": i["title"]}
                   for i, (x, y), c in zip(ideas, m["xy"], m["labels"])],
        "clusters": [{"id": c, "label": m["cluster_label"].get(str(c), ""), "size_sims": len(s),
                      "share": round(len(s) / n_sims, 3)}
                     for c, s in sorted(sims.items(), key=lambda kv: -len(kv[1]))],
    }
    mock_ideas = [{"sim_id": i["sim_id"], "model": i["model"], "title": i["title"], "mechanic": i["mechanic"],
                   "cluster": c} for i, c in zip(ideas, m["labels"])]
    out = ROOT / "web" / "mock.js"
    out.write_text("// сгенерировано web/make_mock.py — не править руками\n"
                   f"window.MOCK_MAP = {json.dumps(mock_map, ensure_ascii=False)};\n"
                   f"window.MOCK_IDEAS = {json.dumps(mock_ideas, ensure_ascii=False)};\n", encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {len(ideas)} идей, {n_sims} соседей, {len(sims)} кластеров")


if __name__ == "__main__":
    main()

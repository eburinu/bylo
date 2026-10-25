"""FastAPI: /api/map и /api/check строго по docs/SCHEMA.md.

uvicorn engine.api:app --reload
Пока нет data/pool/fkn2026.json, бриф fkn2026 отдаёт seed (68 идей, 8 соседей).
"""
import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import embed_cluster as ec
from engine import generate, llm, normalize, twist

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
THRESHOLD_FILE = ROOT / "data" / "gold" / "threshold.json"
N_CLONES = 5
LABEL_TOP = 20     # сколько крупнейших кластеров подписывать через LLM
TWIST_CLUSTERS = 8  # сколько кластеров показывать генератору поворота


def threshold() -> float:
    """Порог «та же идея» в центрированном косинусе. Калибруется по data/gold/pairs.csv
    (инвариант 3); пока разметки нет — env или лучший F1 по парам из seed."""
    if THRESHOLD_FILE.exists():
        return float(json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))["threshold"])
    return float(os.getenv("COLLISION_SIM", "0.24"))


def load_ideas(brief: str) -> list[dict]:
    ideas = generate.load_pool(brief)
    if not ideas and brief in ("fkn2026", "seed"):
        ideas = json.loads(ec.SEED.read_text(encoding="utf-8"))
    return [i for i in ideas if i.get("mechanic")]


@dataclass
class Space:
    """Всё, что нужно для ответа по одному брифу: пул, эмбеддинги, кластеры, карта."""
    ideas: list[dict]
    mean: np.ndarray
    emb: np.ndarray           # центрированные и нормированные
    labels: np.ndarray        # номер кластера у каждой идеи
    xy: np.ndarray
    cluster_label: dict[int, str]
    cluster_sims: dict[int, int]
    n_sims: int

    def sims(self, mechanics: list[str]) -> np.ndarray:
        return ec.center(ec.embed(mechanics), self.mean) @ self.emb.T

    def collision(self, s: np.ndarray, t: float) -> int:
        """Сколько соседей дали хотя бы одну идею с похожестью ≥ t (инвариант 1)."""
        return len({self.ideas[j]["sim_id"] for j in np.nonzero(s >= t)[0]})


_spaces: dict[str, Space] = {}
_lock = asyncio.Lock()


async def space(brief: str) -> Space:
    async with _lock:
        if brief in _spaces:
            return _spaces[brief]
        ideas = load_ideas(brief)
        if not ideas:
            raise HTTPException(404, f"нет пула для брифа {brief}")
        mechanics = [i["mechanic"] for i in ideas]
        raw = await asyncio.to_thread(ec.embed_cached, mechanics, f"emb_{brief}")
        mean = raw.mean(0)
        emb = ec.center(raw, mean)
        key = hashlib.sha1(("\n".join(mechanics) + f"|{ec.CLUSTER_DIST}").encode()).hexdigest()[:12]
        map_path = CACHE / f"map_{brief}_{key}.json"
        if map_path.exists():
            m = json.loads(map_path.read_text(encoding="utf-8"))
            labels, xy = np.array(m["labels"]), np.array(m["xy"])
            cluster_label = {int(k): v for k, v in m["cluster_label"].items()}
        else:
            labels = await asyncio.to_thread(ec.cluster, emb)
            xy = await asyncio.to_thread(ec.project2d, emb)
            cluster_label = await ec.label_clusters(ideas, labels, LABEL_TOP)
            CACHE.mkdir(parents=True, exist_ok=True)
            map_path.write_text(json.dumps({"labels": labels.tolist(), "xy": xy.round(4).tolist(),
                                            "cluster_label": cluster_label}, ensure_ascii=False), encoding="utf-8")
        sp = Space(ideas, mean, emb, labels, xy, cluster_label, ec.sim_counts(ideas, labels),
                   len({i["sim_id"] for i in ideas}))
        _spaces[brief] = sp
        return sp


def _norm_title(t: str) -> str:
    return re.sub(r"[^\w\s]", "", t.lower().replace("ё", "е")).strip()


def name_matches(title: str, ideas: list[dict]) -> list[dict]:
    """Совпадение названий — отдельный показатель (инвариант 2): точное после нормализации
    или однословное название, совпавшее со словом чужого (≥4 букв)."""
    me = _norm_title(title)
    if not me:
        return []
    out, seen = [], set()
    for i in ideas:
        other = _norm_title(i["title"])
        hit = me == other or (" " not in me and len(me) >= 4 and me in other.split()) \
            or (" " not in other and len(other) >= 4 and other in me.split())
        if hit and i["sim_id"] not in seen:
            seen.add(i["sim_id"])
            out.append({"sim_id": i["sim_id"], "title": i["title"]})
    return out


def _fallback_idea(text: str) -> dict:
    """Без сети: название — первый кусок до точки/двоеточия, механика — весь текст."""
    head = re.split(r"[.:—\n]", text.strip(), maxsplit=1)[0]
    return {"title": head[:40].strip(), "mechanic": text.strip()}


app = FastAPI(title="Было")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/map")
async def get_map(brief: str = "fkn2026"):
    sp = await space(brief)
    points = [{"x": round(float(x), 3), "y": round(float(y), 3), "cluster": int(c),
               "sim_id": i["sim_id"], "title": i["title"]}
              for i, (x, y), c in zip(sp.ideas, sp.xy, sp.labels)]
    clusters = [{"id": c, "label": sp.cluster_label.get(c, ""), "size_sims": n,
                 "share": round(n / sp.n_sims, 3)}
                for c, n in sorted(sp.cluster_sims.items(), key=lambda kv: -kv[1])]
    return {"n_sims": sp.n_sims, "points": points, "clusters": clusters}


class CheckIn(BaseModel):
    brief: str = "fkn2026"
    idea: str
    twists: bool = True


def _cache_path(brief: str, idea: str) -> Path:
    key = hashlib.sha1(f"{brief}\n{' '.join(idea.lower().split())}".encode()).hexdigest()[:16]
    return CACHE / f"check_{brief}_{key}.json"


@app.post("/api/check")
async def check(req: CheckIn):
    if not req.idea.strip():
        raise HTTPException(400, "пустая идея")
    path = _cache_path(req.brief, req.idea)
    if path.exists():  # инвариант 5: заготовленные идеи отвечают без сети
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}

    sp = await space(req.brief)
    online = bool(llm.UTIL_MODEL and os.getenv("LLM_API_KEY"))
    idea = _fallback_idea(req.idea)
    if online:
        try:
            idea = await normalize.user_idea(req.idea)
        except Exception:  # noqa: BLE001 — сеть упала: считаем по сырому тексту
            online = False
    t = threshold()
    s = (await asyncio.to_thread(sp.sims, [idea["mechanic"]]))[0]
    n_coll = sp.collision(s, t)
    rate = round(n_coll / sp.n_sims, 3)

    clones, seen = [], set()
    for j in np.argsort(-s):
        i = sp.ideas[j]
        if i["sim_id"] in seen:
            continue
        seen.add(i["sim_id"])
        clones.append({"sim_id": i["sim_id"], "model": i["model"], "title": i["title"],
                       "mechanic": i["mechanic"], "similarity": round(float(s[j]), 3)})
        if len(clones) == N_CLONES:
            break
    top = int(sp.labels[int(np.argmax(s))])

    twists = []
    if req.twists and online and rate > 0:
        big = sorted(sp.cluster_sims, key=lambda c: -sp.cluster_sims[c])[:TWIST_CLUSTERS]
        try:
            cand = await twist.propose(idea, [sp.cluster_label.get(c, "") for c in big])
        except Exception:  # noqa: BLE001 — поворот необязателен
            cand = []
        if cand:
            st = await asyncio.to_thread(sp.sims, [c["mechanic"] for c in cand])
            for c, row in zip(cand, st):
                r = round(sp.collision(row, t) / sp.n_sims, 3)
                if r < rate:  # инвариант 4: только перепроверенные и реже исходной
                    twists.append({**c, "collision_rate": r})
            twists.sort(key=lambda x: x["collision_rate"])

    out = {"idea": idea, "n_sims": sp.n_sims, "collision_sims": n_coll, "collision_rate": rate,
           # null, если ни один сосед не выше порога: иначе «ближайший» кластер вводит в заблуждение
           "cluster": {"id": top, "label": sp.cluster_label.get(top, ""), "size_sims": sp.cluster_sims[top]}
           if n_coll else None,
           "clones": clones, "name_matches": name_matches(idea["title"], sp.ideas),
           "twists": twists, "threshold": t, "cached": False}
    if online:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if (ROOT / "web" / "index.html").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

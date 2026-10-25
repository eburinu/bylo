"""Эмбеддинги механик, кластеры, 2D-карта, подписи кластеров.

Проверка на seed без сети: python -m engine.embed_cluster
(ARI с manual_cluster при разных порогах; ниже ~0.4 — что-то сломано).
"""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine import llm

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
SEED = ROOT / "data" / "seed" / "sims_2026-10-25.json"
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# Расстояния и пороги — в центрированном пространстве (см. center), не в сыром косинусе e5
CLUSTER_DIST = float(os.getenv("CLUSTER_DIST", "0.8"))

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Нормированные эмбеддинги. e5 обучен с префиксом «query: » — без него сходства плывут."""
    return model().encode([f"query: {t}" for t in texts], normalize_embeddings=True,
                          batch_size=64, show_progress_bar=False)


def embed_cached(texts: list[str], name: str) -> np.ndarray:
    """embed с кэшем на диске: data/cache/<name>.npy, ключ — хэш всех текстов."""
    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
    path = CACHE / f"{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = embed(texts)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def center(emb: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Вычитает средний вектор пула и нормирует. У e5 все косинусы 0.8–0.9 (анизотропия),
    после центрирования внутри кластера ~0.2, между ~0: ARI на seed 0.37 → 0.46."""
    c = emb - mean
    return c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-9)


def cluster(emb: np.ndarray, dist: float = CLUSTER_DIST) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=dist).fit_predict(emb)


def project2d(emb: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    if len(emb) < 5:
        return np.zeros((len(emb), 2))
    xy = TSNE(n_components=2, metric="cosine", init="pca", random_state=4,
              perplexity=min(30, max(5, len(emb) // 4))).fit_transform(emb)
    return (xy - xy.mean(0)) / (xy.std(0) + 1e-9)


async def label_clusters(ideas: list[dict], labels: np.ndarray, top: int) -> dict[int, str]:
    """Подписи для `top` крупнейших кластеров через util-модель; без ключа — механика первой идеи."""
    members = defaultdict(list)
    for idea, c in zip(ideas, labels):
        members[int(c)].append(idea["mechanic"])
    biggest = sorted(members, key=lambda c: -len(members[c]))[:top]
    out = {c: m[0] for c, m in members.items()}
    if not llm.UTIL_MODEL or not os.getenv("LLM_API_KEY"):
        return out

    # один запрос на все кластеры: дешевле и не ест лимит бесплатных моделей
    blocks = "\n\n".join(f"Кластер {c}:\n" + "\n".join(f"- {m}" for m in members[c][:8]) for c in biggest)
    try:
        r = await llm.complete_json(llm.UTIL_MODEL, llm.prompt("label", clusters=blocks),
                                    temperature=0, max_tokens=60 * len(biggest))
        for k, v in (r.get("labels", {}) if isinstance(r, dict) else {}).items():
            if str(k).strip().isdigit() and int(k) in members and str(v).strip():
                out[int(k)] = str(v).strip(" «»\"'.\n")
    except Exception:  # noqa: BLE001 — подпись не критична, остаётся механика
        pass
    return out


def sim_counts(ideas: list[dict], labels: np.ndarray) -> dict[int, int]:
    """Сколько разных соседей (sim_id) в каждом кластере — инвариант 1."""
    sims = defaultdict(set)
    for idea, c in zip(ideas, labels):
        sims[int(c)].add(idea["sim_id"])
    return {c: len(s) for c, s in sims.items()}


def _ari_report():
    from sklearn.metrics import adjusted_rand_score, f1_score
    ideas = json.loads(SEED.read_text(encoding="utf-8"))
    # «other» — свалка уникальных идей: каждой свой ярлык, иначе ARI награждает за слипание
    truth = [i["manual_cluster"] if i["manual_cluster"] != "other" else f"other_{n}" for n, i in enumerate(ideas)]
    raw = embed([i["mechanic"] for i in ideas])
    emb = center(raw, raw.mean(0))
    print(f"{len(ideas)} идей, {len({i['sim_id'] for i in ideas})} соседей, ручных кластеров {len(set(truth))}")
    for d in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0):
        lab = cluster(emb, d)
        print(f"dist={d:.2f}  кластеров={len(set(lab)):3}  ARI={adjusted_rand_score(truth, lab):.3f}")
    s = emb @ emb.T
    iu = np.triu_indices(len(ideas), 1)
    same = np.array([[a == b for b in truth] for a in truth])[iu]
    print(f"косинус: внутри ручного кластера {s[iu][same].mean():.3f}, между {s[iu][~same].mean():.3f}")
    # временный порог коллизии, пока нет data/gold/pairs.csv: лучший F1 по парам из seed
    best = max((f1_score(same, s[iu] >= t), t) for t in np.arange(0.1, 0.9, 0.02))
    print(f"порог по seed-парам: {best[1]:.2f} (F1={best[0]:.3f})")


if __name__ == "__main__":
    _ari_report()

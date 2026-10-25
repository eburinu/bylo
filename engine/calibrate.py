"""Калибровка по человеческой разметке data/gold/pairs.csv (инвариант 3).

python -m engine.calibrate [--brief fkn2026]
- порог косинуса (запасной путь без сети) — лучший F1 по «да» → data/cache/threshold.json
- если есть ключ — согласие судьи с людьми по тем же парам
"""
import argparse
import asyncio
import csv
import json
import os

import numpy as np

from engine import embed_cluster as ec
from engine import judge, llm
from engine.api import ROOT, THRESHOLD_FILE, load_ideas

PAIRS = ROOT / "data" / "gold" / "pairs.csv"


def f1(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    tp = (pred & y).sum()
    p, r = tp / max(1, pred.sum()), tp / max(1, y.sum())
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


async def main(brief: str):
    rows = [r for r in csv.DictReader(PAIRS.open(encoding="utf-8-sig")) if r["same"].strip()]
    lab = np.array([r["same"].strip().lower() for r in rows])
    y = lab == "да"
    print(f"{len(rows)} пар: да {y.sum()}, частично {(lab == 'частично').sum()}, нет {(lab == 'нет').sum()}")

    mean = ec.embed([i["mechanic"] for i in load_ideas(brief)]).mean(0)
    a = ec.center(ec.embed([r["a_mechanic"] for r in rows]), mean)
    b = ec.center(ec.embed([r["b_mechanic"] for r in rows]), mean)
    s = (a * b).sum(1)
    for name in ("да", "частично", "нет"):
        print(f"  косинус у «{name}»: {np.round(np.sort(s[lab == name]), 2)}")
    # для порога «да» и «частично» вместе: «да» всего ~6 штук, и F1 по ним выбирает одну
    # высокую точку (0.72), офлайн-режим тогда не видит даже явных клонов
    # Ложный клон на демо хуже пропущенного: берём самый точный порог, который ещё
    # ловит треть настоящих совпадений (по F1 выходил 0.23 и «пицца = RAG по чату»).
    y_emb = lab != "нет"
    scored = [(f1(s >= t, y_emb), t) for t in np.arange(0.0, 0.9, 0.01)]
    (bf, bp, br), bt = max(((m, t) for m, t in scored if m[2] >= 1 / 3), key=lambda x: (x[0][1], -x[1]))
    print(f"эмбеддинги (да+частично): порог {bt:.2f}  F1={bf:.2f} P={bp:.2f} R={br:.2f}")
    THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps({"threshold": round(float(bt), 2), "f1": round(float(bf), 3),
                                          "n_pairs": len(rows), "brief": brief}), encoding="utf-8")
    print(f"→ {THRESHOLD_FILE.relative_to(ROOT)}")

    if not (llm.UTIL_MODEL and os.getenv("LLM_API_KEY")):
        print("ключа нет — судью не проверяю")
        return
    verdicts = np.array(await asyncio.gather(*(judge.same(r["a_mechanic"], r["b_mechanic"]) for r in rows)))
    jf, jp, jr = f1(verdicts == "да", y)
    print(f"судья {llm.UTIL_MODEL}: F1={jf:.2f} P={jp:.2f} R={jr:.2f}, "
          f"точное совпадение трёх классов {(verdicts == lab).mean():.0%}")
    for r, v in zip(rows, verdicts):
        if (v == "да") != (r["same"].strip().lower() == "да"):
            print(f"  #{r['pair_id']} люди={r['same']} судья={v}: {r['a_title']} ↔ {r['b_title']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="fkn2026")
    asyncio.run(main(ap.parse_args().brief))

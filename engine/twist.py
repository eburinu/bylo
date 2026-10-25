"""Поворот: увести идею из кластеров. Инвариант 4 — каждый поворот перепроверяется
по пулу, показываем только те, что сталкиваются реже исходной идеи."""
from engine import llm


async def propose(idea: dict, cluster_labels: list[str]) -> list[dict]:
    text = llm.prompt("twist", idea=f"{idea['title']}: {idea['mechanic']}",
                      clusters="\n".join(f"- {c}" for c in cluster_labels))
    r = await llm.complete_json(llm.UTIL_MODEL, text, temperature=0.9, max_tokens=800)
    twists = r.get("twists", []) if isinstance(r, dict) else r
    return [{"title": str(t.get("title", "")).strip(), "mechanic": str(t.get("mechanic", "")).strip()}
            for t in twists if isinstance(t, dict) and t.get("mechanic")]

"""Статическая версия страницы для GitHub Pages: без сервера, на настоящих данных.

python -m engine.static_demo <папка>
Кладёт index.html (копия web/index.html с правками ниже) и mock.js: карта и пул на 100
соседей из data/cache и data/pool, готовые ответы /api/check для демо-идей. Идеи вне кэша
страница сравнивает по словам (как в web/ без сервера) и честно это подписывает.
"""
import hashlib
import json
import sys
from pathlib import Path

from engine import api
from engine.warm import DEMO

BRIEF = "fkn2026"


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def replace_once(s: str, a: str, b: str) -> str:
    assert s.count(a) == 1, f"не нашёл в web/index.html: {a[:60]}"
    return s.replace(a, b)


def main(out: Path):
    ideas = api.load_ideas(BRIEF)
    mechanics = [i["mechanic"] for i in ideas]
    key = hashlib.sha1(("\n".join(mechanics) + f"|{api.ec.CLUSTER_DIST}").encode()).hexdigest()[:12]
    m = json.loads((api.CACHE / f"map_{BRIEF}_{key}.json").read_text(encoding="utf-8"))
    labels = m["labels"]
    sims = {}
    for i, c in zip(ideas, labels):
        sims.setdefault(c, set()).add(i["sim_id"])
    n_sims = len({i["sim_id"] for i in ideas})
    mock_map = {  # форма GET /api/map
        "n_sims": n_sims,
        "points": [{"x": round(x, 3), "y": round(y, 3), "cluster": c, "sim_id": i["sim_id"], "title": i["title"]}
                   for i, (x, y), c in zip(ideas, m["xy"], labels)],
        "clusters": [{"id": c, "label": m["cluster_label"].get(str(c), ""), "size_sims": len(s),
                      "share": round(len(s) / n_sims, 3)}
                     for c, s in sorted(sims.items(), key=lambda kv: -len(kv[1]))],
    }
    mock_ideas = [{"sim_id": i["sim_id"], "model": i["model"], "title": i["title"], "mechanic": i["mechanic"],
                   "cluster": c} for i, c in zip(ideas, labels)]
    checks = {}
    for idea in DEMO:
        p = api._cache_path(BRIEF, idea, True)
        if p.exists():
            checks[norm(idea)] = {**json.loads(p.read_text(encoding="utf-8")), "cached": True}

    html = (api.ROOT / "web" / "index.html").read_text(encoding="utf-8")
    # готовый ответ из кэша, если идея из демо; иначе — сравнение по словам
    html = replace_once(html, "if (MOCK) { await new Promise(r => setTimeout(r, 600)); res = mockCheck(text); }",
                        "if (MOCK) { await new Promise(r => setTimeout(r, 600)); "
                        "res = (window.MOCK_CHECKS || {})[text.toLowerCase().split(/\\s+/).join(\" \")] || mockCheck(text); }")
    html = replace_once(html, '$("verdictNote").textContent = MOCK ? "демо без сервера · сравнение по словам"',
                        '$("verdictNote").textContent = MOCK && !r.judged ? "статическая версия · сравнение по словам — для настоящей проверки запустите сервер"')
    html = replace_once(html, 'title="API не ответил — показываю seed и грубое сравнение по словам">демо без сервера',
                        'title="GitHub Pages: карта и примеры — настоящие, новые идеи сравниваются по словам">статическая версия')
    # все демо-идеи — кнопками
    examples = html[html.index("const EXAMPLES = ["):html.index("];", html.index("const EXAMPLES = [")) + 2]
    html = html.replace(examples, "const EXAMPLES = " + json.dumps(DEMO[:-1], ensure_ascii=False, indent=2) + ";")

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "mock.js").write_text(
        "// сгенерировано engine/static_demo.py\n"
        f"window.MOCK_MAP = {json.dumps(mock_map, ensure_ascii=False)};\n"
        f"window.MOCK_IDEAS = {json.dumps(mock_ideas, ensure_ascii=False)};\n"
        f"window.MOCK_CHECKS = {json.dumps(checks, ensure_ascii=False)};\n", encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"{out}: {len(ideas)} идей, {n_sims} соседей, {len(checks)} готовых ответов")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Парсер Devpost: проекты хакатона → data/validation/<id>.json [{"title", "tagline", "url"}].

python -m validation.devpost <хакатон> [--id <id>]
  <хакатон> — поддомен (`hackmit2024`) или любая ссылка на него (`https://hackmit2024.devpost.com/...`).
python -m validation.devpost --html page1.html page2.html --id <id>
  если сеть режет Devpost: сохранить страницы галереи из браузера (Ctrl+S) и разобрать локально.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "validation"
UA = "Mozilla/5.0 (bylo validation; hackathon research)"
MAX_PAGES = 50


def slug(arg: str) -> str:
    """`hackmit2024`, `hackmit2024.devpost.com` или полная ссылка → `hackmit2024`."""
    host = urlparse(arg if "//" in arg else "//" + arg).hostname or arg
    return host.removesuffix(".devpost.com")


def parse_gallery(html: str) -> list[dict]:
    """Карточки проектов со страницы project-gallery. Карточка — ссылка на /software/<имя>
    с заголовком (h5) и подзаголовком (p.tagline)."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="devpost.com/software/"], a[href^="/software/"]'):
        title = a.select_one("h5") or a.select_one("h4, h3")
        if not title:
            continue
        tagline = a.select_one(".tagline, p")
        url = a["href"] if a["href"].startswith("http") else "https://devpost.com" + a["href"]
        out.append({"title": " ".join(title.get_text(" ").split()),
                    "tagline": " ".join(tagline.get_text(" ").split()) if tagline else "",
                    "url": url.split("?")[0]})
    return out


def fetch(hackathon: str, pause: float = 1.0) -> list[dict]:
    import httpx
    base = f"https://{hackathon}.devpost.com/project-gallery"
    projects, seen = [], set()
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as http:
        for page in range(1, MAX_PAGES + 1):
            r = http.get(base, params={"page": page})
            r.raise_for_status()
            new = [p for p in parse_gallery(r.text) if p["url"] not in seen]
            if not new:  # за последней страницей Devpost отдаёт пустую галерею или повтор
                break
            seen.update(p["url"] for p in new)
            projects += new
            print(f"стр. {page}: +{len(new)} (всего {len(projects)})", file=sys.stderr)
            time.sleep(pause)
    return projects


def dedup(projects: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in projects:
        if p["url"] not in seen and p["title"]:
            seen.add(p["url"])
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hackathon", nargs="?", help="поддомен или ссылка на хакатон")
    ap.add_argument("--id", help="имя файла в data/validation/ (по умолчанию — поддомен)")
    ap.add_argument("--html", nargs="+", type=Path, help="разобрать сохранённые страницы вместо сети")
    a = ap.parse_args()
    if a.html:
        projects = dedup([p for f in a.html for p in parse_gallery(f.read_text(encoding="utf-8"))])
    elif a.hackathon:
        projects = dedup(fetch(slug(a.hackathon)))
    else:
        ap.error("нужен хакатон или --html")
    vid = a.id or (slug(a.hackathon) if a.hackathon else None)
    if not vid:
        ap.error("с --html нужен --id")
    if not re.fullmatch(r"[\w-]+", vid):
        ap.error(f"плохой id: {vid}")
    no_tag = sum(not p["tagline"] for p in projects)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{vid}.json"
    path.write_text(json.dumps(projects, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {len(projects)} проектов, без tagline {no_tag}")
    if len(projects) < 40:
        print("меньше 40 проектов — для валидации мало, возьми хакатон побольше", file=sys.stderr)


if __name__ == "__main__":
    main()

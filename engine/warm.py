"""Прогрев кэша для демо (инвариант 5): карта и ответы /api/check для заготовленных идей
считаются заранее и лежат в data/cache/ — на защите работают без сети.

python -m engine.warm [--brief fkn2026]
"""
import argparse
import asyncio
import time

from engine import api

# первые три — дословно примеры со страницы (web/index.html, EXAMPLES): ключ кэша — текст
DEMO = [
    "Алиби: по фото и подписи сверяем тени с положением солнца и архивом погоды, выдаём список противоречий",
    "Агент проходит README проекта в чистом Docker и присылает PR с исправленной инструкцией",
    "Второй мозг для семьи: бот собирает рецепты, чеки и напоминания из чата и отвечает на вопросы",
    "Детектор дипфейков: загружаешь видео, нейросеть ищет артефакты и выдаёт вероятность подделки",
    "AR-гид по кампусу: наводишь камеру, получаешь подсказки, куда идти и что делать новичку",
    "Проверка что на видеозвонке живой человек: случайный челлендж мимикой и подписанный токен",
    "Второй мозг: заметки и документы индексируются, по вопросу находим связанный контекст со ссылками",
    "Было: генерируем сотню ИИ-соседей с тем же заданием и показываем, сколько из них придумали вашу идею",
    "Бот для заказа пиццы в общагу",
]


async def run(brief: str):
    t = time.time()
    await api.space(brief)
    print(f"карта готова за {time.time() - t:.0f} с")

    async def one(idea):
        api._cache_path(brief, idea).unlink(missing_ok=True)
        t = time.time()
        r = await api.check(api.CheckIn(brief=brief, idea=idea))
        print(f"{time.time() - t:5.1f}s  {r['collision_sims']:3}/{r['n_sims']} judged={r['judged']} "
              f"повороты={len(r['twists'])}  {idea[:50]}")

    await asyncio.gather(*(one(i) for i in DEMO))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="fkn2026")
    asyncio.run(run(ap.parse_args().brief))

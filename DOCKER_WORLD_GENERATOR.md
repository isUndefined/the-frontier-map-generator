# The Frontier — ChatGPT art handoff (без API)

`docker compose` **не делает платных запросов** к OpenAI и не вызывает локальную
AI-модель. Он готовит один ZIP для генерации изображения вручную в ChatGPT.
Остальные стадии (проверка соседей, склейка, экспорт) автоматизированы.

## Новый чанк

```bash
docker compose build generator
docker compose run --rm generator expand --direction west --provider chatgpt
```

Результат — `jobs/<job>/chatgpt_bundle.zip`. Внутри:

- `CHATGPT_INSTRUCTIONS.md` — как генерировать, фиксированный размер и основной prompt;
- `input/generation_canvas.png`, `generation_mask.png`, `prompt.txt`, `context_board.png`;
- все доступные референсы соседей и диагоналей из `input/`;
- `job_info.json` — координаты и размер холста без абсолютных путей.

Загрузи ZIP в ChatGPT. Попроси нарисовать **одну новую карту** в размере,
указанном в инструкциях, строго без изменения закреплённых пикселей.
ChatGPT не гарантирует точный размер вывода или пиксельную неизменность;
проверь результат. Docker **не увеличивает и не уменьшает** PNG молча.

Сохрани PNG как `jobs/<job>/generated.png`. Затем:

```bash
docker compose run --rm generator finalize --job <job> --draft
docker compose run --rm generator finalize --job <job>
```

`--draft` создаёт черновик, но не публикует. Обычный `finalize` проверяет
разрешение, стык и записывает независимый `tile.png` и WebP-экспорт.
При несовпадении размера CLI возвращает ошибку; существующая карта не меняется.

## Для уже подготовленного tile_x-2_y+0

Не запускай `expand` снова. Подготовленные изображения остаются пригодными:

```bash
docker compose run --rm generator pack --job tile_x-2_y+0
```

Загрузи `jobs/tile_x-2_y+0/chatgpt_bundle.zip` сюда. После генерации
помести `generated.png` в `jobs/tile_x-2_y+0/` и запусти `finalize`.

## Export без огромной склеенной карты

```bash
docker compose run --rm generator export --format webp
```

Выход: `jobs/_export/manifest.json` и отдельные `jobs/_export/tiles/*.webp`.
Никаких запросов OpenAI API, ключей или платы за API нет. Генерация арта через
ChatGPT остаётся ручным шагом.

## Tests

```bash
python -m unittest discover -s tests -v
```

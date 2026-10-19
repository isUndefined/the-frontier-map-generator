# SHARED OVERLAP — рабочая схема расширения региона

## Файлы
- `world_prepare_v3_shared_overlap.py`
- `stitch_shared_overlap.py`

## Идея
Генерация идёт не по схеме:

`[ GENERATED ][ SOURCE ]`

а по схеме:

`[ NEW_CORE ][ SHARED_OVERLAP ][ SOURCE_CONTEXT ]`

После генерации сборка идёт так:

- для `left`: `[ NEW_CORE ][ MERGED_OVERLAP ][ SOURCE_TAIL ]`
- для `right`: `[ SOURCE_HEAD ][ MERGED_OVERLAP ][ NEW_CORE ]`

## 1. Prepare
Пример для расширения влево:

```bash
python world_prepare_v3_shared_overlap.py prepare \
  --source bg-ravefall-valley.png \
  --job jobs/west_02 \
  --direction left \
  --add-width 1040 \
  --native-height 853 \
  --overlap 256 \
  --context 700
```

После этого появятся:

- `jobs/west_02/input/generation_canvas.png`
- `jobs/west_02/input/generation_mask.png`
- `jobs/west_02/input/source_region_native.png`
- `jobs/west_02/job.json`

## 2. Генерация в модели
В модель отправляешь:

- `generation_canvas.png`
- `generation_mask.png`
- твой prompt

Важно: `generated.png` должен вернуться **точно того же размера**, что и `generation_canvas.png`.

## 3. Stitch

```bash
python stitch_shared_overlap.py stitch \
  --job jobs/west_02 \
  --generated jobs/west_02/generated.png \
  --feather 4
```

На выходе:

- `jobs/west_02/stitched/final_native.png`
- `jobs/west_02/stitched/seam_preview.png`
- `jobs/west_02/stitched/final_native_preview.png`

## Пояснение параметров
- `add-width` — ширина реально новой зоны в native-height пространстве
- `overlap` — общая overlap-зона, внутри которой ищется лучший seam
- `context` — сколько пикселей существующей карты даём модели как якорь

## Рекомендуемые стартовые значения
- `native-height = 853`
- `overlap = 256`
- `context = 700`
- `feather = 4`

## Что важно
Это уже лучше, чем жёсткая склейка по одной вертикали, потому что seam выбирается внутри overlap-зоны по минимальной разнице между `generated` и `source`.

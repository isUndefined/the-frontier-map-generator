# Docker world generator: automatic OpenAI Image Edit

Incremental map extension publishes local tiles, never a giant panorama.
The existing prepare/stitch pipeline remains authoritative.

## Setup

```sh
cp .env.example .env
# Set OPENAI_API_KEY in the local, untracked .env
docker compose build generator
docker compose run --rm generator status
docker compose run --rm generator expand --direction west --dry-run
```

The API is paid separately from ChatGPT. API key and model availability are
required. Default model: `gpt-image-2.5-sunburst`, quality: `high`.
Alternative: `gpt-image-2.5-flare`.

## Expand in one command

```sh
docker compose run --rm generator expand --direction west
```

The command prepares neighboring visual context, mask and prompt,
calls OpenAI Images Edit, validates the result, restores fixed anchor pixels,
then stitches, publishes and exports standalone browser tiles plus manifest.
If the request or validation fails, the job stays pending.

## Resume the already prepared job

You previously created `jobs/tile_x-2_y+0/input/`. Do NOT run expand again:
the target location is already pending. Run:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0
```

This calls the provider if `generated.png` is absent, and then publishes.
For a non-publishing visual inspection:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft
```

If a generated.png is already present, finalize reuses it. A failed API
call leaves the prepared job for retry. No API call happens with --dry-run.

## Native-size policy

The original generation canvas is **never resized or resampled**.
Only enough extra pixels are appended to make its dimensions multiples of 16.
For example 2372×941 becomes 2384×944. The response must be exactly that
size or it is rejected. The padding is cropped (not resized) back to the
original dimensions. Black Frontier mask pixels are restored exactly from
the original source. Areas larger or smaller than allowed by the model are
rejected rather than rescaled.

The model may still fail to continue a road, river or cliff correctly, even
when the fixed pixels match; validate the art before treating it as navigation.

## Export

```sh
docker compose run --rm generator export --format webp
```

Results are independent lossless WebP tiles in `jobs/_export/tiles/` and
`jobs/_export/manifest.json`. Nothing modifies `main` or the old tiles.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use mocked API responses: no paid API calls. Full Docker build and
real API end-to-end execution have not been run in this environment.

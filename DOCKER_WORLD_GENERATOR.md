# The Frontier: incremental art-world generator

This Docker CLI wraps the existing prepare/stitch pipeline. It does not generate
a giant cumulative world PNG; each published region remains a standalone tile.
The image-model provider is deliberately optional and is NOT implemented here.

## Prerequisites

- Docker with Compose V2
- jobs/center/region.png, jobs/layout.json and existing completed jobs, as in
  the current repository
- Optional image generation provider executable (see below)

## Quick start

    docker compose build generator
    docker compose run --rm generator status
    docker compose run --rm generator expand --direction west --dry-run
    docker compose run --rm generator expand --direction west

The CLI selects the outermost published tile on the requested frontier.
The final command creates a new job and writes a generation canvas, preserve
mask (black = unchanged), prompt and neighboring reference images under:

    jobs/tile_x-2_y+0/input/

The job name is an example; the actual coordinates come from the existing map.
Without an image model the command stops here safely, without inventing art.
Generate the image externally and place it at:

    jobs/<job>/generated.png

Then validate the fixed overlap and publish the new standalone tile:

    docker compose run --rm generator finalize --job tile_x-2_y+0

To view a draft before publishing:

    docker compose run --rm generator finalize --job tile_x-2_y+0 --draft

## Choose a specific existing edge

    docker compose run --rm generator expand --direction south --from west_01
    docker compose run --rm generator expand --direction east --from '0,0'

The --from argument accepts a published tile ID or x,y grid coordinates.
If a tile is already published or pending, the CLI refuses to overwrite it.
Prepare, stitch and export honor existing geometry and dependencies.

## One command with a real provider

A provider is an external command installed in the container or mounted inside
./providers. It must write exactly one output image at FRONTIER_OUTPUT,
preserve the masked anchors and keep the canvas aspect ratio.

Available environment variables:

    FRONTIER_CANVAS      input/generation_canvas.png
    FRONTIER_MASK        input/generation_mask.png
    FRONTIER_PROMPT      input/prompt.txt
    FRONTIER_CONTEXT     input/context_board.png
    FRONTIER_OUTPUT      where the provider must write the output PNG
    FRONTIER_JOB         prepared job directory
    FRONTIER_TARGET_X    world grid x of new chunk
    FRONTIER_TARGET_Y    world grid y of new chunk

Example with your own implemented providers/my_generator.py:

    export FRONTIER_GENERATOR_COMMAND='python /app/providers/my_generator.py'
    docker compose run --rm generator expand --direction north

The CLI prepares inputs, invokes the provider, runs the existing stitch
verification and exports separate browser tiles. On failure it leaves the job
pending for correction. No model credentials or API adapters are shipped.
The provider runs without shell=True, so provide a command, not a pipeline.

## Browser tiles: no mega-image

    docker compose run --rm generator export --format webp
    docker compose run --rm generator export --format png --force

Published assets are saved to:

    jobs/_export/tiles/x+0_y+0.webp
    jobs/_export/tiles/x-1_y+0.webp
    jobs/_export/manifest.json

The manifest stores grid positions, world pixel positions, tile size,
overlap and a dependency-first render order. Render tiles in manifest order;
overlaps are deliberate and newer tiles paint over their source neighbors.
Default WebP compression is lossless to avoid new seam artifacts. Use
--lossy for smaller images only when acceptable.

The old build_grid_from_jobs.py remains available for an optional complete
preview but is never executed by this CLI.

## Tests

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v

Caveats: Docker does not supply an image model. A model may fail to respect
the preserve mask, and stitching rejects modified anchors. Semantic
segmentation and world-state navigation are separate future stages.
Docker build/runtime must be tested on a host with Docker installed.

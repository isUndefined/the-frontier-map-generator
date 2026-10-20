# Docker — original map generator from main

This branch keeps **all original Python map logic unchanged**. Docker only
provides the Python environment and a writable mount of local `jobs/`.
No API keys, image services, or feature-branch generation pipelines are used.

Build once:

```sh
docker compose build generator
```

The container runs the same original scripts with the same arguments:

```sh
# Prepare a new tile immediately west of west_01, at grid (-2, 0).
# The example name can be changed; check your current jobs first.
docker compose run --rm generator prepare_region_grid.py \
  --jobs jobs --x -2 --y 0 --primary east --job west_02

# Send the generated canvas, mask and prompt to ChatGPT manually.
# Save the returned PNG as jobs/west_02/generated.png.

# Inspect the draft without publishing.
docker compose run --rm generator stitch_region_grid.py \
  --jobs jobs --job west_02 --draft

# Finalize only after visual inspection.
docker compose run --rm generator stitch_region_grid.py \
  --jobs jobs --job west_02

# Optional: rebuild the combined world preview only when needed.
docker compose run --rm generator build_grid_from_jobs.py --jobs jobs

# Run the unmodified original tests.
docker compose run --rm generator -m unittest discover -s tests -v
```

Source art and job inputs are stored in `./jobs` on the host and persist
between container runs. The entry point is the standard Python interpreter;
this is not a new frontend or tile-generation CLI.

## Repository cleanup

Removed only regenerateable/obsolete tracked artifacts: `tmp/`, Python
bytecode caches, `jobs/_grid/`, and the tracked `jobs/*/draft/`.
Published art `jobs/*/tile.png`, metadata `region.json`, original
`generated.png`, preparation `input/`, and source
`jobs/center/region.png` remain intact.

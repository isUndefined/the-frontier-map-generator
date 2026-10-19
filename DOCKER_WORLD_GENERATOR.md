# The Frontier: World Atlas (ChatGPT handoff + visual QA)

No paid OpenAI API calls. The generator prepares source art and ZIP for manual ChatGPT use.

## Open your whole world in the browser

After pulling this branch, run once:

```sh
docker compose build generator
docker compose run --rm generator export --format webp
docker compose up -d viewer
```

**Open http://localhost:8000/** on the computer running Docker. If the Docker
host is remote, the service binds to 127.0.0.1 by default; use SSH port forwarding
(`ssh -L 8000:127.0.0.1:8000 <your-server>`) to view the same URL securely.
The viewer serves only `jobs/_export`, not source jobs or secrets.

The page draws all tiles from `manifest.json` in their dependency-first order,
using `pixel_x/pixel_y`. It never needs a giant stitched raster.

## Tools

- **Show Grid (G):** logical tile cell outlines, using step width/height
- **Show Overlap (O):** highlight exact overlap strips of direct neighbors
- **Tile Bounds (B):** show each full source image boundary
- **Tile Coordinates (L):** show tile names and grid coordinates
- **Highlight Latest:** outline last tile in export paint order
- **Pan / Zoom:** drag, wheel, +/- buttons, Fit (F)
- **Select tile:** click to inspect coordinates, dependencies and open the original
- **Refresh:** reread manifest after another export

The viewer is auto-copied to `jobs/_export/index.html` whenever the generator
publishes or exports. To inspect your newest tile without regenerating artwork,
run `docker compose run --rm generator export --format webp` then refresh the browser.

## Continue the existing ChatGPT-assisted workflow

```sh
docker compose run --rm generator expand --direction west --provider chatgpt
# Or for already prepared job:
docker compose run --rm generator pack --job tile_x-2_y+0
```

Upload the resulting `jobs/<job>/chatgpt_bundle.zip` to ChatGPT, save the PNG
at `jobs/<job>/generated.png` with the exact canvas dimensions, then:

```sh
docker compose run --rm generator finalize --job <job> --draft
docker compose run --rm generator finalize --job <job>
```

Existing local tiles remain independent. Docker doesn't resize artwork and
the API-free workflow doesn't require any credits.

## Run tests

```sh
python -m unittest discover -s tests -v
```

## Quality gate / fixing a bad published tile

```sh
docker compose run --rm generator audit --job tile_x-2_y+0
```

The report is stored in `jobs/tile_x-2_y+0/quality_report.json`.
Large near-identical terrain copied **outside the physical overlap** is a
hard failure, even if stitch accepted the generated image.
Weak ChatGPT alignment is also blocked by default. Uncertain alignment can
be accepted only with the explicit `--allow-weak-alignment` option, after
visual inspection; that option NEVER bypasses a terrain-duplication rejection.

To fix an already published tile, generate a **new** PNG at the exact native
canvas size and place it at `jobs/tile_x-2_y+0/generated.png`. Inspect the draft:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft
```

The draft writes `draft/quality_report.json` and does not affect the world.
If the report passes, publish the replacement safely:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --replace
```

Old tile and metadata are saved under `backups/<UTC timestamp>/`.
A failed replacement restores the previous publication. Published descendants
which already used the replaced area are not regenerated automatically: inspect
their boundaries and re-render if necessary.

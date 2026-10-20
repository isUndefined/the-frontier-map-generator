# The Frontier: World Atlas (ChatGPT handoff + visual QA)

No paid OpenAI API calls. The generator prepares canvas, mask and prompt for manual ChatGPT use. ZIP is optional.

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

Upload `input/generation_canvas.png`, `input/generation_mask.png` and
`input/prompt.txt` to ChatGPT (or use the optional ZIP), save the full PNG
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

## Rebuild an existing region against ALL touching neighbors

```sh
docker compose build generator
docker compose run --rm generator rebuild-job --job tile_x-2_y+0
```

This stages inputs, then backs up old `input/`, metadata, old `generated.png`
and drafts under `jobs/tile_x-2_y+0/backups/input_<UTC timestamp>/`.
The published `tile.png` and `region.json` remain unchanged.

Send **three separate files** to ChatGPT (no ZIP needed):

- `jobs/tile_x-2_y+0/input/generation_canvas.png`
- `jobs/tile_x-2_y+0/input/generation_mask.png`
- `jobs/tile_x-2_y+0/input/prompt.txt`

`neighbor_debug_preview.png` shows where neighboring terrain was locked.
The lock uses the same world-coordinate geometry and dependency-first paint
order as the viewer, including all diagonal intersections. For new prepare v2 jobs the generated image
must be the full 2056×1325 px canvas; the published tile remains 1672×941 px. Save it at `jobs/tile_x-2_y+0/generated.png`.

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft
docker compose run --rm generator finalize --job tile_x-2_y+0 --replace
```

The stitcher restores black-mask pixels **after** any alignment/side merging
from the locked visible-world composite. Newly expanded jobs produce the same
three files automatically, without an auto-generated ZIP. The optional old
`pack` command remains for compatibility.

## Prepare v2: world context without moving geography

New `expand` and `rebuild-job` inputs use `chatgpt_world_canvas_v3` with
`prepare_version: 2`. Existing prepared jobs keep their workflow until rebuilt.
The published layout remains 1672×941 with overlap 256. The generation canvas
adds a native 192px halo on all four sides: **2056×1325**. There is no scaling.
All surrounding source crops use actual world-space intersections in exactly
the atlas dependency paint order. The old target is excluded from context.

For target (-2,0), world origin is (-2832,0); canvas origin is (-3024,-192).
The exact output tile crop is **[192,192,1864,1133)** (XYXY, half-open).
The east neighbor `west_01` contributes source [0,0,448,941), of which only
[0,0,256,941) overlaps the tile. The remaining 192px stays OUTSIDE the target.
`north_02` contributes [0,493,448,941), with a real shared 256×256 corner
[0,685,256,941). `south_01` contributes [0,0,448,448), with shared corner
[0,0,256,256). Later sources own intersections, as in the world viewer.

All existing world pixels are locked. The 192px transition is in new editable
territory adjacent to known world, including diagonal corner boundaries;
other new territory is free. Unknown padding outside the tile is also locked.
No overlap geography is repeated or moved to make the transition.

In `input/`:

- `generation_canvas.png`, `generation_mask.png`, `prompt.txt`: the only uploads.
- `zones.json`: world/canvas origins, explicit crop, all source crops and paint order.
- `generation_zones.png`: exact pixel partition (0 padding, 1 lock, 2 transition, 3 free).
- `prepare_debug_overlay.png`: red lock, yellow transition, green free, gray padding;
  white rectangle is the published tile crop. Do not upload this debug image.
- `neighbor_map.json`: all actual coordinate neighbors.
- `neighbor_lock.png`, `neighbor_mask.png`: immutable tile-sized world snapshot,
  retained for stitch/finalize and exact overlap restoration.

ChatGPT must return the FULL canvas at the dimensions in the prompt. If it
returns a different size, regenerate: both finalize and direct stitch reject
it rather than resize or guess the crop. Prepare v2 never automatically warps
world geometry. Anchor QA ignores unknown black padding, checks real context
and shared tile pixels, and rejects changed anchors by default. No artistic
continuity is guaranteed by geometry alone; inspect the strict draft.
Seam QA checks every cardinal neighbor. The old adaptive overlap handoff and
its tone ramp remain available for already prepared legacy jobs; they do not
repaint prepare v2's locked overlaps.

Update and safely rebuild an existing published job:

```sh
git switch feature/docker-world-generator
git pull --ff-only origin feature/docker-world-generator
docker compose build generator
docker compose run --rm generator rebuild-job --job tile_x-2_y+0
```

Rebuild stages the new inputs before replacing old inputs and archives stale
generated art, draft reports and optional ZIP. Failure during installation
restores the old inputs. Published tile/region metadata stays unchanged.
Send the three new inputs to ChatGPT; save the full output as
`jobs/tile_x-2_y+0/generated.png`, then:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft --strict
# After inspecting the draft and its QA report:
docker compose run --rm generator finalize --job tile_x-2_y+0 --replace --strict
```

## Whole-world draft QA

`docker compose run --rm generator finalize --job tile_x-2_y+0 --draft`

Check **`jobs/tile_x-2_y+0/draft/world_preview.png`**, not only the old
`primary_pair_preview.png`. The new image shows the candidate in the same
`dependencies_first` ordering as the atlas, with every neighboring tile
(including both diagonal corners) and no changes to published art.
`draft/world_preview.json` includes the covered world coordinates, the
candidate pixel rectangle and tile render order. A dependency cycle fails
instead of rendering a misleading preview.

## Seam QA

Run `docker compose run --rm generator finalize --job tile_x-2_y+0 --draft --strict`.
Inspect `draft/world_preview.png`, `draft/seam_east_preview.png`, `draft/seam_ne_preview.png`, `draft/seam_se_preview.png`, and `draft/quality_report.json`. Strict mode rejects persistent significant gradient discontinuities. Pixels of existing neighbors remain unchanged. Once draft is acceptable, use `finalize --job tile_x-2_y+0 --replace --strict`.

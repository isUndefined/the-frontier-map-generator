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
order as the viewer, including all diagonal intersections. For new jobs, each generated image is a native 1536×1024 local pass
(1024×1536 for north/south); the assembled tile remains 1672×941 px.
See the native-pass instructions below before finalizing. Save it at `jobs/tile_x-2_y+0/generated.png`.

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft
docker compose run --rm generator finalize --job tile_x-2_y+0 --replace
```

The stitcher restores black-mask pixels **after** any alignment/side merging
from the locked visible-world composite. Newly expanded jobs produce the same
three files automatically, without an auto-generated ZIP. The optional old
`pack` command remains for compatibility.

## Native local outpainting: current default

The large whole-tile generation was insufficient in practice: image generators
could resize the canvas and invent an unrelated valley against a narrow existing
mountain edge. New `expand` and `rebuild-job` inputs now use **native local passes**
(`prepare_version: 3`, `chatgpt_native_patches_v1`). Existing prepared jobs retain
their format until rebuilt. Layout, published pixels and overlap are unchanged.

Each request uses a standard **1536×1024** canvas (1024×1536 for north/south),
with the real 256px overlap and up to 700px of real adjacent context. Only a
local band of new territory is editable. Work proceeds from the existing seam
outwards. Accepted pixels become exact context for the next pass. No pixel is
resized, rotated, shifted or stretched. Diagonal intersections and source paint
order still come from the same coordinate-based world compositor as the atlas.

For the west tile (-2,0), the default plan has three passes:

| Pass | Editable target x pixels | Canvas x interval relative to tile |
| --- | --- | --- |
| 1 | [836,1416) | [836,2372) |
| 2 | [256,836) | [256,1792) |
| 3 | [0,256) | [0,1536) |

Each canvas spans target y [-42,982). Its black padding stays outside the
1672×941 target. Pass 1 contains source [0,0,956,941) from `west_01` and the
real intersections of `north_02` and `south_01`. Geography is never moved into
the new band. The first 192px next to known terrain is marked transition in
`generation_zones.png`; other editable pixels are free.

Update and rebuild a published job:

```sh
git switch feature/docker-world-generator
git pull --ff-only origin feature/docker-world-generator
docker compose build generator
docker compose run --rm generator rebuild-job --job tile_x-2_y+0
```

For EACH pass, send these three new files to ChatGPT, without a ZIP:

- `jobs/tile_x-2_y+0/input/generation_canvas.png`
- `jobs/tile_x-2_y+0/input/generation_mask.png`
- `jobs/tile_x-2_y+0/input/prompt.txt`

Save its **1536×1024** PNG as `jobs/tile_x-2_y+0/generated.png`. Inspect the
actual terrain connection; black/white mask boundaries must not become cliffs
or separate compositions. Accept passes 1 and 2 with:

```sh
docker compose run --rm generator advance-job --job tile_x-2_y+0 --strict
```

`advance-job` archives the current three inputs, generated image and acceptance
report, then installs the next handoff atomically. It removes the stale root
`generated.png` into a backup so it cannot be reused for the next pass. A failed
installation restores all prior inputs and output. Published art is untouched.
`draft/pass_preview.png` and `draft/pass_quality_report.json` show rejected
strict passes. Regenerate that same pass instead of advancing it.

After generating pass 3, assemble and inspect the whole tile:

```sh
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft --strict
# After inspection:
docker compose run --rm generator finalize --job tile_x-2_y+0 --replace --strict
```

Finalize refuses an incomplete tile. For other layouts/neighbors the pass count
can differ; `zones.json` and command output identify the current pass. Re-running
`advance-job` on the last pass leaves inputs unchanged and tells you to finalize.

The final image is still **1672×941**. Only new white-mask pixels are taken
from generated passes. Original locks and accepted previous bands remain exact.
Anchor validation uses native pixel error first; if texture/color changed, it
requires at least 60 distributed consistent landmark matches, with 90% within
2px and fitted drift <=2px. This is validation only: **no transform is applied**.
A result with the former ~7px shift is rejected. Existing duplication QA remains
mandatory; strict seam QA also checks joins between local passes. These checks
are useful proxies, not proof of artistic quality; inspect the full world draft.

`input/zones.json` records the exact current canvas/world/edit rectangles,
source crops, actual neighbors and accepted passes. `prepare_debug_overlay.png`
shows red known pixels, yellow transition, green free and gray padding.
`input/patch_state.json`, `accepted_tile.png` and `remaining_mask.png` retain
progress; `input/passes/` contains accepted handoffs. A rebuild archives ALL of
this progress before starting over. `neighbor_lock.png` and `neighbor_mask.png`
remain the immutable tile-sized source snapshot.

### Historical whole-canvas jobs

Already prepared `prepare_version: 2` jobs still use their exact original
2056×1325 canvas and explicit tile crop. Existing v1 adaptive overlap jobs also
retain their old masks and handoff behavior. No old job is silently converted
or rescaled. To use native passes, run `rebuild-job` and generate new art.

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

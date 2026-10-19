# The Frontier — Map Generator + World Atlas

Develop the world as independent art tiles and inspect them together in an
interactive browser map with **Show Grid, Show Overlap, Tile Bounds**, pan and zoom.

```sh
docker compose build generator
docker compose run --rm generator export --format webp
docker compose up -d viewer
```

Visit **http://localhost:8000/** on the Docker host. When the host is remote,
use SSH port forwarding (see [documentation](DOCKER_WORLD_GENERATOR.md)).
The viewer is exported as `jobs/_export/index.html` with the current manifest.

New art is prepared for ChatGPT; there are no paid image-API calls:

```sh
docker compose run --rm generator expand --direction west --provider chatgpt
# existing pending job:
docker compose run --rm generator pack --job tile_x-2_y+0
# after placing generated.png:
docker compose run --rm generator finalize --job tile_x-2_y+0 --draft
docker compose run --rm generator finalize --job tile_x-2_y+0
```

See [DOCKER_WORLD_GENERATOR.md](DOCKER_WORLD_GENERATOR.md) for the viewer controls
and workflow. The original scripts remain available; no mega-image is required.

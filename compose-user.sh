#!/bin/sh
# Run Docker Compose using the current host user's UID and GID.
set -eu

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
export HOST_UID HOST_GID

exec docker compose "$@"

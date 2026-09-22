#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export UV_ENV_FILE="$PWD/.env"

git submodule update --init

docker build -t chia-exo:latest -f dockerfiles/ExoDockerfile dockerfiles/
docker build -t chia-evolver:latest -f dockerfiles/EvolverDockerfile .

uv run chia down -y cluster.yaml || true
uv run chia up -y cluster.yaml

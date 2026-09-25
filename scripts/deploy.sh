#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export UV_ENV_FILE="$PWD/.env"

docker build -t chia-exo:latest -f dockerfiles/ExoDockerfile dockerfiles/
# The evolver image installs private forks over SSH: forward the ssh-agent if
# one is running, else the default GitHub key.
docker build --ssh "default=${SSH_AUTH_SOCK:-$HOME/.ssh/id_ed25519}" \
    -t chia-evolver:latest -f dockerfiles/EvolverDockerfile dockerfiles/

uv run chia down -y cluster.yaml || true
uv run chia up -y cluster.yaml

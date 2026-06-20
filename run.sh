#!/usr/bin/env bash
# One command to run EFIRAS locally: Redis + API, both in Docker Compose.
# Usage: ./run.sh [up|down|logs|restart]

set -euo pipefail
cd "$(dirname "$0")"

ACTION="${1:-up}"

case "$ACTION" in
  up)
    touch data/visits.db
    docker compose up --build -d
    echo "Waiting for API to become healthy..."
    for i in $(seq 1 30); do
      if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ 2>/dev/null | grep -q "200"; then
        echo "Ready: http://localhost:8080"
        exit 0
      fi
      sleep 2
    done
    echo "API did not become ready in time - check: ./run.sh logs"
    exit 1
    ;;
  down)
    docker compose down
    ;;
  restart)
    docker compose down
    "$0" up
    ;;
  logs)
    docker compose logs -f
    ;;
  *)
    echo "Usage: $0 [up|down|logs|restart]"
    exit 1
    ;;
esac

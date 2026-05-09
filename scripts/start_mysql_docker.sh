#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="customer-service-agent-mysql"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=customer_service \
  -e MYSQL_USER=appuser \
  -e MYSQL_PASSWORD=apppass \
  -p 3306:3306 \
  -v "${PROJECT_ROOT}/sql:/docker-entrypoint-initdb.d" \
  mysql:8.4

echo "Waiting for MySQL to become ready..."
for _ in $(seq 1 60); do
  if docker exec "${CONTAINER_NAME}" mysqladmin ping -h 127.0.0.1 -prootpass >/dev/null 2>&1; then
    echo "MySQL is ready."
    exit 0
  fi
  sleep 2
done

echo "MySQL failed to become ready in time." >&2
exit 1


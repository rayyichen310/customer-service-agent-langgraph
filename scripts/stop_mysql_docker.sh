#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="customer-service-agent-mysql"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "MySQL container stopped."


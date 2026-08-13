#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="vinrobotics"
REMOTE_ROOT="~/Dung_Workspace/testing"
DIRECTORIES=(docs notebooks outputs src)

for command in rsync ssh; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

for directory in "${DIRECTORIES[@]}"; do
  if [[ ! -d "${PROJECT_ROOT}/${directory}" ]]; then
    echo "Missing local directory: ${PROJECT_ROOT}/${directory}" >&2
    exit 1
  fi
done

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_ROOT}"

for directory in "${DIRECTORIES[@]}"; do
  echo "Syncing ${directory}/..."
  rsync -a --partial --human-readable --info=stats2 \
    "${PROJECT_ROOT}/${directory}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/"
done

for directory in "${DIRECTORIES[@]}"; do
  local_stats="$(find "${PROJECT_ROOT}/${directory}" -type f -printf '%s\\n' | awk '{count += 1; total += $1} END {printf "%d files, %d bytes", count, total}')"
  remote_stats="$(ssh "${REMOTE_HOST}" "find ${REMOTE_ROOT}/${directory} -type f -printf '%s\\n' | awk '{count += 1; total += \$1} END {printf \"%d files, %d bytes\", count, total}'")"

  if [[ "${local_stats}" != "${remote_stats}" ]]; then
    echo "Verification failed for ${directory}: local=${local_stats}; remote=${remote_stats}" >&2
    exit 1
  fi

  echo "Verified ${directory}: ${local_stats}"
done

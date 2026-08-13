#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="vinrobotics"
REMOTE_ROOT="~/Dung_Workspace/testing"
DIRECTORIES=(docs)

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

  if ! ssh "${REMOTE_HOST}" "test -d ${REMOTE_ROOT}/${directory}"; then
    echo "Missing remote directory: ${REMOTE_ROOT}/${directory}" >&2
    exit 1
  fi
done

for directory in "${DIRECTORIES[@]}"; do
  echo "Pulling ${directory}/..."
  rsync -a --partial --human-readable --info=stats2 \
    "${REMOTE_HOST}:${REMOTE_ROOT}/${directory}/" \
    "${PROJECT_ROOT}/${directory}/"
done

for directory in "${DIRECTORIES[@]}"; do
  pending_changes="$(rsync -ani \
    "${REMOTE_HOST}:${REMOTE_ROOT}/${directory}/" \
    "${PROJECT_ROOT}/${directory}/")"

  if [[ -n "${pending_changes}" ]]; then
    echo "Verification failed for ${directory}; remote changes remain:" >&2
    echo "${pending_changes}" >&2
    exit 1
  fi

  echo "Verified ${directory}: synchronized"
done

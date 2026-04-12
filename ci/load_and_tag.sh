#!/bin/bash
set -euo pipefail

# 用法: ./ci/load_and_tag.sh [TAR_FILE] [TARGET_REGISTRY_TAG]
# 示例: ./ci/load_and_tag.sh uploader_latest.tar my-registry.local/vectorsphere/uploader:latest

TAR_FILE=${1:-uploader_latest.tar}
TARGET_TAG=${2:-}

if [ ! -f "${TAR_FILE}" ]; then
  echo "Tar file not found: ${TAR_FILE}" >&2
  exit 2
fi

echo "Loading image from ${TAR_FILE}..."
docker load -i "${TAR_FILE}"

if [ -n "${TARGET_TAG}" ]; then
  # find first image id from tar (best-effort), retag
  # we assume the tar contains tag we expect; retag the first matching repo:tag
  ORIGINAL_TAG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep vectorsphere/uploader | head -n1 || true)
  if [ -n "${ORIGINAL_TAG}" ]; then
    docker tag "${ORIGINAL_TAG}" "${TARGET_TAG}"
    echo "Tagged ${ORIGINAL_TAG} -> ${TARGET_TAG}"
    if command -v docker >/dev/null 2>&1; then
      echo "Pushing ${TARGET_TAG} to registry..."
      docker push "${TARGET_TAG}"
    fi
  else
    echo "Warning: could not find original tag to retag. You may tag manually." >&2
  fi
fi

echo "Load completed." 

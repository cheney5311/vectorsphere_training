#!/bin/bash
set -euo pipefail

# 用法: ./ci/build_and_export.sh [IMAGE_TAG] [OUT_TAR]
# 示例: ./ci/build_and_export.sh vectorsphere/uploader:latest uploader_latest.tar

IMAGE_TAG=${1:-vectorsphere/uploader:latest}
OUT_TAR=${2:-uploader_latest.tar}

echo "Building image ${IMAGE_TAG}..."
docker build -t "${IMAGE_TAG}" .

echo "Saving image to ${OUT_TAR}..."
docker save "${IMAGE_TAG}" -o "${OUT_TAR}"

echo "Done: ${OUT_TAR} contains ${IMAGE_TAG}" 

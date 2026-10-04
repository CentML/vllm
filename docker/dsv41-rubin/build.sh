#!/usr/bin/env bash
# Build the DeepSeek-V4.1-Flash Rubin (VR200, aarch64) serving image from a
# checkout of this vLLM branch, in two steps:
#   1. target vllm-openai of docker/Dockerfile with vllm-build-args.txt
#      -> <image>-vllm (local only)
#   2. Dockerfile.dynamo: ai-dynamo on top, vLLM/torch/transformers pinned
#      -> <image>
# Needs native arm64 docker + buildx (any Grace/Vera host). Full instructions: README.md in this directory.
#
# usage: build.sh <image:tag> [--push]
# env:
#   VLLM_SRC        clean git checkout to build (default: this repository). It
#                   must reach a vLLM release tag (GIT_REPO_CHECK=1 and the
#                   wheel version use `git describe`), e.g.
#                   git tag v0.20.2rc0 e6ff3e9c83a6520c3793f4e0511ac8591a07c243
#   DYNAMO_VERSION  ai-dynamo version (default 1.6.0.dev20260924)
#   BUILD_ARGS_FILE override vllm-build-args.txt
#   BUILDX_FLAGS    extra flags for both `docker buildx build` calls, e.g. "--no-cache --pull"
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
image=${1:?usage: build.sh <image:tag> [--push]}
push=${2:-}
src=${VLLM_SRC:-$(git -C "$here" rev-parse --show-toplevel)}
dynamo_version=${DYNAMO_VERSION:-1.6.0.dev20260924}
args_file=${BUILD_ARGS_FILE:-$here/vllm-build-args.txt}
buildx_flags=${BUILDX_FLAGS:-}

git -C "$src" diff --quiet HEAD || { echo "ERROR: $src is dirty" >&2; exit 1; }
git -C "$src" describe --tags --match 'v[0-9]*' >/dev/null || {
  echo "ERROR: no vLLM release tag reachable from HEAD in $src" >&2; exit 1; }
commit=$(git -C "$src" rev-parse HEAD)
mapfile -t args < <(grep -vE '^[[:space:]]*(#|$)' "$args_file")
echo "[build] src=$src commit=$commit describe=$(git -C "$src" describe --tags --match 'v[0-9]*')"
echo "[build] args: ${args[*]}"

t0=$(date +%s)
docker buildx build --platform linux/arm64 --progress=plain $buildx_flags \
  --target vllm-openai \
  "${args[@]/#/--build-arg=}" \
  --build-arg "VLLM_BUILD_COMMIT=$commit" \
  --build-arg "VLLM_IMAGE_TAG=$image" \
  --tag "$image-vllm" \
  -f "$src/docker/Dockerfile" "$src"
t1=$(date +%s)
echo "[build] vllm-openai built in $((t1 - t0)) s"

docker buildx build --platform linux/arm64 --progress=plain ${buildx_flags/--pull/} \
  --build-arg "VLLM_IMAGE=$image-vllm" \
  --build-arg "DYNAMO_VERSION=$dynamo_version" \
  --label "ai.vllm.build.commit=$commit" \
  --tag "$image" \
  -f "$here/Dockerfile.dynamo" "$here"
echo "[build] dynamo layer built in $(( $(date +%s) - t1 )) s"

if [[ $push == --push ]]; then
  docker push "$image"
  docker image inspect --format '{{json .RepoDigests}}' "$image"
fi

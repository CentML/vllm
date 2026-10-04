#!/usr/bin/env bash
# Check that every input of the DeepSeek-V4.1-Flash Rubin image resolves from PUBLIC sources: no credentials, no
# proxy, no internal mirror. See README.md "Public-sources manifest".
#
# usage: check_public_sources.sh [--full] [--freeze FILE]
#   default  base images (by digest), git pins (each commit fetched from its public repo), the explicitly pinned
#            wheels on their public index, and fixed-URL downloads (NVSHMEM, Triton's LLVM and NVIDIA redists,
#            Rust toolchain, uv, get-pip, apt/dnf repositories).
#   --full   also every `name==version` of the image's pip freeze (image-pip-freeze.txt next to this script, or
#            --freeze FILE) on PyPI or one of the public extra indexes the build uses, and every crates.io crate of
#            rust/Cargo.lock.
# Needs bash, curl, git, python3. Exit status: number of items that did not resolve (0 = all public).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
FULL=0; FREEZE=$HERE/image-pip-freeze.txt
while [ $# -gt 0 ]; do case $1 in --full) FULL=1;; --freeze) FREEZE=$2; FULL=1; shift;; *) echo "unknown arg $1" >&2; exit 64;; esac; shift; done
# Public access only: drop proxies, package-index overrides and the user's git config (insteadOf rewrites, tokens).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY \
      PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false
CURL=(curl -sS --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 120 -A check_public_sources)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAIL=0
ok()  { printf 'OK    %-8s %s\n' "$1" "$2"; }
bad() { printf 'FAIL  %-8s %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }

# ---------------------------------------------------------------- base images (anonymous registry v2 API)
ACCEPT='application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'
image() {  # <registry host> <repository> <tag> <digest>
  local host=$1 repo=$2 tag=$3 digest=$4 tok
  case $host in
    registry-1.docker.io) tok=$("${CURL[@]}" "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$repo:pull") ;;
    nvcr.io) tok=$("${CURL[@]}" "https://nvcr.io/proxy_auth?scope=repository:$repo:pull") ;;
  esac
  tok=$(printf '%s' "$tok" | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])' 2>/dev/null)
  local code
  code=$("${CURL[@]}" -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $tok" -H "Accept: $ACCEPT" \
         "https://$host/v2/$repo/manifests/$digest")
  if [ "$code" = 200 ]; then ok image "$host/$repo:$tag@$digest"; else bad image "$host/$repo:$tag@$digest (HTTP $code)"; fi
  local cur
  cur=$("${CURL[@]}" -I -H "Authorization: Bearer $tok" -H "Accept: $ACCEPT" "https://$host/v2/$repo/manifests/$tag" \
        | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:"{print $2}')
  [ "$cur" = "$digest" ] || echo "NOTE  image    $repo:$tag now points to ${cur:-?}; the build pins $digest"
}
echo "== base images"
image registry-1.docker.io pytorch/manylinuxaarch64-builder cuda13.4 \
  sha256:487d5f86222afdfda9426da96332ca2603ff0c5024c56bd1b363241240ef7a8e
image nvcr.io nvidia/cuda-dl-base 26.08-cuda13.4-devel-ubuntu24.04 \
  sha256:843a5e81ed49ad2a2787aa10e48e926dc1f9440522a122ec2a637c27039fc77d

# ---------------------------------------------------------------- git pins (commit fetched by hash, no checkout)
gitpin() {  # <url> <commit> [label]
  local d="$TMP/g$RANDOM$RANDOM"
  git init -q "$d" && if git -C "$d" fetch -q --depth=1 --filter=tree:0 "$1" "$2" 2>/dev/null \
     && [ "$(git -C "$d" rev-parse FETCH_HEAD)" = "$2" ]; then ok git "$1@$2 ${3:-}"; else bad git "$1@$2 ${3:-}"; fi
  rm -rf "$d"
}
gittag() {  # <url> <tag> <commit>
  local got; got=$(git ls-remote "$1" "refs/tags/$2^{}" "refs/tags/$2" 2>/dev/null | awk 'NR==1{print $1}')
  if [ "$got" = "$3" ]; then ok tag "$1 $2 = $3"; else bad tag "$1 $2 = $3 (got ${got:-nothing})"; fi
}
echo "== git"
GH=https://github.com
# this branch and its public forks (the vendored cmake patches reproduce the fork branches)
gitpin $GH/nvzhihanj/vllm.git 7cf5542c25a83d229abfe6779b99d42aa0b5c9bc "image source (dsv41-flash-rubin-opt)"
gittag $GH/vllm-project/vllm.git v0.20.2rc0 e6ff3e9c83a6520c3793f4e0511ac8591a07c243
gitpin $GH/nvzhihanj/DeepGEMM.git c50dea4ff3d4f3bba9d6441c33f0b5422aeea230 "fork dsv41-rubin-megamoe-sm107"
gitpin $GH/nvzhihanj/FlashMLA.git 43876b77cb6d2dfbb5eac4ca853a316672d91542 "fork dsv41-rubin-sm107"
# cmake FetchContent pins (CMakeLists.txt, cmake/external_projects/*.cmake) and their submodules
gittag $GH/NVIDIA/cutlass.git v4.7.1 cb4247394dd82148787aed73e5dc7cef33cbf862
gittag $GH/NVIDIA/cutlass.git v4.8.0 098de2a652cf8f00fd70b2df54051c7eccbb855a
gitpin $GH/vllm-project/FlashMLA.git 0eee43b12f034b657133cf2afca6a72ebb6efccf
gitpin $GH/NVIDIA/cutlass.git 147f5673d0c1c3dcf66f78d677fd647e4a020219 "FlashMLA csrc/cutlass"
gitpin $GH/vllm-project/DeepGEMM.git e1f418c2a4f20818221f6b0e578b4c2f634d4c3f
gitpin $GH/NVIDIA/cutlass.git f3fde58372d33e9a5650ba7b80fc48b3b49d40c8 "DeepGEMM third-party/cutlass"
gitpin $GH/deepseek-ai/DeepJIT.git e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16 "DeepGEMM third-party/deep_jit"
gitpin $GH/vllm-project/flash-attention.git 9cd61de38763d712bb6ce56e2a02cc2bf718c89f
gitpin $GH/NVIDIA/cutlass.git 62750a2b75c802660e4894434dc55e839f322277 "flash-attention csrc/cutlass"
gitpin $GH/ROCm/composable_kernel.git c56c6750d0fc54ed771d532cc92c316423449614 "flash-attention csrc/composable_kernel"
gitpin $GH/ROCm/aiter.git 9bab8388c35936814a659b4ebd245c491e1b940a "flash-attention third_party/aiter"
gitpin $GH/ROCm/composable_kernel.git af7118e342580ecd3f71edce7b1d0ba465012ecf "aiter 3rdparty/composable_kernel"
gitpin $GH/vllm-project/DeepSelect.git d96d33afe1fab0d6066da49cdc91e64c2bee65ea
gitpin $GH/NVIDIA/cutlass.git ae6bccf341fb4410241f696ba06873023d5ce4ed "DeepSelect csrc/3rdparty/cutlass"
gitpin $GH/vllm-project/FlashKDA.git 17a037d98da546deb4591e967cf961a43c034d8b
gitpin $GH/NVIDIA/cutlass.git 5c149f52a436782210263fb2f19b354443a61c6a "FlashKDA cutlass"
gitpin $GH/vllm-project/MSA.git f355c37eb4e1413f21ee2ad8bbad25079e6bef9d "fmha_sm100"
gitpin $GH/NVIDIA/cutlass.git eb61c911471867a5fd2466bfd8f29306cea6ebf8 "MSA python/fmha_sm100/cutlass"
gitpin $GH/vllm-project/tml-fa4.git 75765e76a9c2c012c1f6ecd64577eb646eb4d303
gitpin $GH/IST-DASLab/qutlass.git e74319e3405ce6d71965732880f5dc1f52371f64
gitpin $GH/NVIDIA/cutlass.git b2ca083d2bb96c41d9b3c5a930637c641f6669bf "qutlass third_party/cutlass"
gittag $GH/triton-lang/triton.git v3.5.1 0add68262ab0a2e33b84524346cb27cbb2787356 # triton_kernels
# docker build stages
gitpin $GH/triton-lang/triton.git 3f6e41132b5edf639bfb872ad73d4688765e08b8 "Triton from source"
gitpin $GH/deepseek-ai/DeepEP.git d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb "DeepEP (tools/ep_kernels)"
gitpin $GH/fmtlib/fmt.git a4c7e17133ee9cb6a2f45545f6e974dd3c393efa "DeepEP third-party/fmt"
# source of the PyTorch nightly wheels (torch.version.git_version etc. in the image), the fallback if the nightlies
# are pruned from download.pytorch.org (README "PyTorch nightlies")
gitpin $GH/pytorch/pytorch.git 3d2b4c7639df58a777529a4699efa74b00619d90 "torch 2.15.0.dev20260908 (nightly release commit)"
gitpin $GH/pytorch/vision.git add1dd9ec5d33b983b22b163130c21c01b7dc9fa "torchvision 0.30.0.dev20260909 (nightly release commit)"
gitpin $GH/pytorch/audio.git 9b89f15d387b85f740a934c7683482ab39de6c30 "torchaudio 2.11.0.dev20260911 (nightly release commit)"
# Rust frontend git crates (rust/Cargo.lock)
gitpin $GH/oss-harmony/harmony.git 76e849426cc092f84509e31a17027755f67d662a "crate harmony v0.0.11"
gitpin $GH/smg-project/llm-multimodal.git f0985ef65967615db2c79279aa07818499301bfd "crate llm-multimodal"

# ---------------------------------------------------------------- explicitly pinned wheels
# wheel <name> <version> <index>: index = pypi, or a PEP 503 simple-index base URL. Requires a cp312/abi3/py3
# wheel for aarch64 (or any platform) of exactly that version.
wheel_in_index() {  # <name> <version> <index-base>
  local norm; norm=$(printf '%s' "$1" | tr 'A-Z_.' 'a-z--')
  "${CURL[@]}" -f "$3/$norm/" 2>/dev/null | python3 -c '
import re, sys, urllib.parse
name, ver = sys.argv[1], sys.argv[2]
stem = re.sub(r"[-_.]+", "_", name).lower() + "-" + ver.lower() + "-"
for href in re.findall(r"href=\"([^\"]+)\"", sys.stdin.read()):
    f = urllib.parse.unquote(href.split("#")[0].rsplit("/", 1)[-1]).lower()
    if f.startswith(stem) and f.endswith(".whl") and ("aarch64" in f or "none-any" in f):
        sys.exit(0)
sys.exit(1)' "$1" "$2"
}
wheel_on_pypi() {  # <name> <version>
  "${CURL[@]}" -f "https://pypi.org/pypi/$1/$2/json" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
ok = any(u["filename"].endswith(".tar.gz") or "aarch64" in u["filename"] or "none-any" in u["filename"] for u in d["urls"])
sys.exit(0 if ok else 1)'
}
wheel() {
  if [ "$3" = pypi ]; then wheel_on_pypi "$1" "$2"; else wheel_in_index "$1" "$2" "$3"; fi \
    && ok wheel "$1==$2 ($3)" || bad wheel "$1==$2 ($3)"
}
echo "== pinned wheels"
PTN=https://download.pytorch.org/whl/nightly/cu134
FI=https://flashinfer.ai/whl/cu134
wheel torch 2.15.0.dev20260908+cu134 $PTN
wheel torchvision 0.30.0.dev20260909+cu134 $PTN
wheel torchaudio 2.11.0.dev20260911+cu134 $PTN
wheel flashinfer-python 0.7.0.post1 pypi
wheel flashinfer-cubin 0.7.0.post1 https://flashinfer.ai/whl
wheel flashinfer-jit-cache 0.7.0.post1+cu134 $FI
wheel nvidia-cutlass-dsl 4.8.0.dev0 pypi
wheel quack-kernels 0.6.5 pypi
wheel cuda-python 13.4.1 pypi
wheel cuda-bindings 13.4.1 pypi
wheel nixl 1.4.1 pypi
wheel nvidia-nccl-cu13 2.30.7 pypi
wheel ai-dynamo 1.6.0.dev20260924 pypi
wheel ai-dynamo-runtime 1.6.0.dev20260924 pypi

# ---------------------------------------------------------------- fixed-URL downloads
url() {  # <url> [label]
  local code; code=$("${CURL[@]}" -o /dev/null -w '%{http_code}' -L -r 0-0 "$1")
  case $code in 200|206) ok url "$1 ${2:-}";; *) bad url "$1 ${2:-} (HTTP $code)";; esac
}
echo "== downloads"
NV=https://developer.download.nvidia.com/compute
url $NV/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.24_cuda13-archive.tar.xz "NVSHMEM for DeepEP"
url https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-5f07f818-almalinux-arm64-1.tar.gz "Triton LLVM"
for p in cuda_nvcc:12.9.86 cuda_cuobjdump:13.1.80 cuda_nvdisasm:13.1.80 cuda_crt:13.1.80 cuda_cudart:13.1.80 \
         cuda_cupti:12.8.90 cuda_cupti:13.3.35; do
  n=${p%%:*}; v=${p#*:}
  url $NV/cuda/redist/$n/linux-sbsa/$n-linux-sbsa-$v-archive.tar.xz "Triton NVIDIA tool"
done
url $GH/nlohmann/json/releases/download/v3.11.3/include.zip "Triton json"
url https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl "torch dependency"
url https://sh.rustup.rs "rustup"
url https://static.rust-lang.org/dist/channel-rust-1.95.toml "Rust 1.95 (rust-toolchain.toml)"
url https://astral.sh/uv/install.sh "uv installer"
url https://bootstrap.pypa.io/get-pip.py
url http://ports.ubuntu.com/ubuntu-ports/dists/noble/Release "Ubuntu apt"
url https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/noble/Release "deadsnakes apt"
url https://repo.almalinux.org/almalinux/8/BaseOS/aarch64/os/repodata/repomd.xml "manylinux_2_28 dnf"

# ---------------------------------------------------------------- --full: every installed wheel and crate
if [ $FULL = 1 ]; then
  echo "== image pip freeze: $FREEZE"
  export -f wheel_on_pypi wheel_in_index; export TMP
  INDEXES="$PTN https://download.pytorch.org/whl/cu134 $FI https://flashinfer.ai/whl https://pypi.nvidia.com"
  export INDEXES CURL_ARGS="${CURL[*]}"
  one() {
    CURL=($CURL_ARGS)
    local line=$1 name ver
    case $line in
      *" @ file://"*) printf 'SRC   wheel    %s (built from source in docker/Dockerfile)\n' "${line%% @*}"; return 0;;
      *==*) name=${line%%==*}; ver=${line#*==};;
      *) return 0;;
    esac
    if [[ $ver != *+* ]] && wheel_on_pypi "$name" "$ver"; then printf 'OK    wheel    %s==%s (pypi)\n' "$name" "$ver"; return 0; fi
    for ix in $INDEXES; do
      if wheel_in_index "$name" "$ver" "$ix"; then printf 'OK    wheel    %s==%s (%s)\n' "$name" "$ver" "$ix"; return 0; fi
    done
    printf 'FAIL  wheel    %s==%s (not on PyPI or the public extra indexes)\n' "$name" "$ver"; return 1
  }
  export -f one
  grep -vE '^\s*(#|$)' "$FREEZE" | xargs -P 16 -I{} bash -c 'one "$@"' _ {} > "$TMP/freeze.out"
  sort -k3 "$TMP/freeze.out"; FAIL=$((FAIL + $(grep -c '^FAIL' "$TMP/freeze.out")))
  echo "== crates.io (rust/Cargo.lock)"
  python3 - "$REPO/rust/Cargo.lock" > "$TMP/crates.txt" <<'PY'
import re, sys
for blk in open(sys.argv[1]).read().split("[[package]]")[1:]:
    if 'source = "registry+https://github.com/rust-lang/crates.io-index"' in blk:
        n = re.search(r'^name = "([^"]+)"', blk, re.M).group(1); v = re.search(r'^version = "([^"]+)"', blk, re.M).group(1)
        print(n, v)
PY
  crate() { local c; c=$(curl -sS -o /dev/null -w '%{http_code}' -L -r 0-0 --retry 3 "https://static.crates.io/crates/$1/$1-$2.crate"); case $c in 200|206) echo "OK";; *) echo "FAIL  crate    $1 $2 (HTTP $c)";; esac; }
  export -f crate
  xargs -P 16 -n 2 bash -c 'crate "$@"' _ < "$TMP/crates.txt" > "$TMP/crates.out"
  n_ok=$(grep -c '^OK' "$TMP/crates.out"); n_bad=$(grep -c '^FAIL' "$TMP/crates.out")
  grep '^FAIL' "$TMP/crates.out"; printf '%s   crates   %d of %d crates.io crates resolve\n' "$([ $n_bad = 0 ] && echo OK || echo FAIL)" "$n_ok" "$((n_ok + n_bad))"
  FAIL=$((FAIL + n_bad))
fi
echo "== $([ $FAIL = 0 ] && echo "ALL PUBLIC: every pinned input resolved anonymously" || echo "$FAIL item(s) did NOT resolve")"
exit $FAIL

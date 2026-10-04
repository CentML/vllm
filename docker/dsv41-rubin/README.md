# DeepSeek-V4.1-Flash on NVIDIA Vera Rubin (VR200): vLLM + Dynamo image

This directory is the canonical build recipe for the DeepSeek-V4.1-Flash serving image for VR200 (sm_107, aarch64,
CUDA 13.4). It covers:
- what this branch adds to upstream;
- where every input comes from (all public);
- how to rebuild the image from scratch;
- how to validate it;
- the known issues.

Files in this directory:

| file | purpose |
|---|---|
| `vllm-build-args.txt` | `docker/Dockerfile` build arguments (CUDA 13.4.1, arch 10.0, Rubin prerelease wheels, base images pinned by digest) |
| `build.sh` | builds target `vllm-openai` of `docker/Dockerfile`, then `Dockerfile.dynamo`; optional push |
| `Dockerfile.dynamo` | adds `ai-dynamo` to the vLLM image, with vLLM, torch and transformers pinned |
| `check_public_sources.sh` | checks that every pinned input resolves anonymously from its public source |
| `image-pip-freeze.txt` | `pip freeze` of the reference image (the full Python manifest) |
| `validate_image.py` | inventory and one-GPU smoke check, run inside the image |

## 1. Prebuilt image

| | |
|---|---|
| image | `gitlab-master.nvidia.com:5005/mlpinf/mlperf-endpoints/centml-rubin-vllm:zhihanj-dsv41-rubin-7cf5542c25` |
| digest | `sha256:c52590f1f94b63454f65e5d2f40c813d3eea9a57cc3d3030a94908f7f2608ea2` |
| source | `github.com/nvzhihanj/vllm`, branch `dsv41-flash-rubin-opt` @ `7cf5542c25` |
| vLLM | `0.20.2rc1.dev6106+g7cf5542c2.d20261003.cu134` |
| stack | torch `2.15.0.dev20260908+cu134`, Triton `3.8.0+git3f6e4113`, FlashInfer `0.7.0.post1`, nvidia-cutlass-dsl `4.8.0.dev0`, transformers `5.18.0`, ai-dynamo `1.6.0.dev20260924`, CUDA 13.4.1, Python 3.12 |
| arch | linux/arm64; `TORCH_CUDA_ARCH_LIST=10.0` (sm_100f family code); FlashMLA `_flashmla_C` also sm_107a |

Commits after `7cf5542c25` that touch only `docker/dsv41-rubin/` do not change the image's contents: docs, the check
script, and pinning the base images to the digests already used. A rebuild from them differs only in the version
metadata (`dev6107+g<sha>`, `VLLM_BUILD_COMMIT`).

The registry above is NVIDIA-internal. Everything that goes into the image is public, so anyone can rebuild it with
section 5.

```bash
# Slurm + Pyxis/Enroot: import once into a squashfs and use it as --container-image
enroot import -o dsv41-rubin-7cf5542c25.sqsh \
  'docker://gitlab-master.nvidia.com:5005#mlpinf/mlperf-endpoints/centml-rubin-vllm@sha256:c52590f1f94b63454f65e5d2f40c813d3eea9a57cc3d3030a94908f7f2608ea2'
# Docker
docker pull gitlab-master.nvidia.com:5005/mlpinf/mlperf-endpoints/centml-rubin-vllm@sha256:c52590f1f94b63454f65e5d2f40c813d3eea9a57cc3d3030a94908f7f2608ea2
```

Notes on the import:
- `enroot import` needs registry credentials in `~/.config/enroot/.credentials`:
  `machine gitlab-master.nvidia.com:5005 login <user> password <token>`.
- Run it on a compute node with local scratch. The squashfs is ~15 GB, and the import takes ~2-3 min on a VR200
  node.
- The image's entrypoint is `vllm serve`. Dynamo runs with the image's `python3`, e.g. `python3 -m dynamo.frontend`
  and `python3 -m dynamo.vllm`. No separate venv is needed.

## 2. What this branch contains

The lineage, from upstream down:
1. `vllm-project/vllm`.
2. CentML's DeepSeek-V4.1 branch `CentML/vllm:mlperf-end-multiturn-v1.0-dsv4.1-flash`. That is the model, DSpark
   speculative decoding, Engram and the Rubin bring-up fixes, at `cbc89c23`, later merged up to `c91ad8b7`.
3. 7 GB300 commits.
4. The Rubin commits below.

Every Rubin change that can alter model outputs or memory use is **opt-in by environment variable**. Without the
variables, the model runs the GB300 branch's code paths. The default-on changes (CPU scheduler paths, pure copies,
sampler kernels, the FlashMLA build) are bitwise identical for the ops DSV4.1 uses.

The GB300 commits:
- inverse-RoPE + FP8 quant kernel;
- `VLLM_DP_RANK_LOCAL_OVERRIDE`;
- CED candidate bound;
- decoder SWA bounded replay (`VLLM_DSV41_GRAPH_BOUNDED_REPLAY`);
- deferred full GC (`VLLM_ENGINE_DEFER_FULL_GC`);
- a residual-mass IMA fix;
- the Engram all-to-all (`VLLM_DSV41_ENGRAM_ALL_TO_ALL`).

Rubin commits, oldest first. "Default on" means no flag is needed. Results are VR200 microbenchmarks unless marked
e2e. The e2e numbers are same-node serving pairs at the DSV4.1 C2048 point.

| commit | change | flag | result |
|---|---|---|---|
| `61275492e0` | MXFP8 FlashInfer per-M backend switch | `VLLM_MXFP8_FI_LARGE_M_BACKEND=cutlass`, `..._THRESHOLD` | CUTLASS 3-22% faster for M >= 3072. e2e: no gain, keep off; superseded by `=rubin` below |
| `c3f198a63a` | mega-attention prefill: plan chunks once per step | default on | identical values; worker CPU only (e2e within noise) |
| `4326ff5d1d` | DSpark: step-major draft logits (contiguous Markov-bias add) | default on | removes ~0.94 ms/step of strided adds; bitwise identical logits |
| `c4d7812f46` | MegaMoE: Rubin SM107 DeepGEMM kernel (K=64 block-scaled UMMA, packed FP4 weights) | `VLLM_DSV41_MEGAMOE_SM107=1` | 1.03-1.08x per call at T/rank 1024-8192; 99.98% of outputs bitwise equal (FP32 accumulation order) |
| `558d068796` | MegaMoE: die-local routed weights + per-die task queues (Rubin locality domains) | `VLLM_DSV41_MEGAMOE_PERDIE=routed` (needs `..._SM107=1`) | another 1-5% per call (1.04-1.10x total); bitwise equal to the SM107 kernel |
| `7f5598c731` ... `76c3d91232` (13 commits) | scheduler / KV-manager CPU fast paths (`[Core][perf]`) | default on | identical scheduling decisions; DSV4.1 C2048 CPU harness: `schedule()` 5.52 -> 2.95 ms/step, `update_from_output()` 3.03 -> ~2.0 ms/step |
| `0e6958e3d8` | top-k/top-p kernel: dynamic row hand-out, 2 programs/SM | default on | 1.33-1.51x; bitwise identical |
| `20d8cf4d01` | Gumbel sampling: screened noise, exact recompute; fused DSpark bias add | default on | 146 -> 81 us; bitwise identical tokens |
| `aa7e14d54e` | rejection sampler: screened resample, register cap | default on | 163 -> 83 us; identical tokens |
| `9a82f3e630` | vectorized hc-stream broadcast and replay-row scatter | default on | 148 -> 15 us, 116 -> 14 us; pure copies |
| `18d647b175` | build: FlashMLA `_flashmla_C` with CUTLASS 4.8, `--use_fast_math`, extra sm_107a cubin | build-time | fused attention bitwise identical; decode -4..-11%, prefill -7..-13% per call |
| `42e9d85204` | build: vendored DeepGEMM gets the SM107 MegaMoE kernel + per-die execution | build-time | provides the kernels for `c4d7812f46` / `558d068796` |
| `e17ac6c330` | build: this directory (`vllm-build-args.txt`, `Dockerfile.dynamo`, `build.sh`) | - | the image serves the same as the overlay stack it replaces (e2e 24,394 vs 24,500 tok/s/GPU) |
| `ec0c9ab349` | Triton BF16->MXFP8 activation quant (F8_128x4 scales) | `VLLM_MXFP8_TRITON_QUANT=1` | 1.3-1.7x; bit-identical values and scales |
| `20197d93ea` | per-shape MXFP8 GEMM dispatch: Sm107 cute-dsl / cuBLASLt / SM100 cute-dsl (primes FlashInfer's cute-dsl alpha cache per device before graph capture) | `VLLM_MXFP8_FI_LARGE_M_BACKEND=rubin` | <= 1 bf16 ulp vs cute-dsl, identical error vs an FP32 reference |
| `0f9b7b1611`, `7cf5542c25` | exact fused candidate-block top-k for the indexer (radix select = `torch.topk` order) | `VLLM_DSV41_FAST_CANDIDATE_TOPK=1` | decode scores + top-k 792 -> 383-445 us; bitwise identical to `torch.topk` |

The three dense flags (`VLLM_MXFP8_TRITON_QUANT=1`, `VLLM_MXFP8_FI_LARGE_M_BACKEND=rubin`,
`VLLM_DSV41_FAST_CANDIDATE_TOPK=1`) together gave **+2.0% e2e** in a same-node pair.

## 3. Forks and how they map to the vendored patches

The build fetches only upstream repositories and applies patches kept in this tree (`cmake/patches/`). The forks
hold the same changes as commits, for browsing and review. **The build does not use the forks.**

| component | upstream pin (fetched by cmake) | patches applied at build time | equivalent fork branch |
|---|---|---|---|
| DeepGEMM | `vllm-project/DeepGEMM@e1f418c` + submodules (`cmake/external_projects/deepgemm.cmake`) | `deepgemm-sm107-sm100-family.patch`, then `deepgemm-megamoe-sm107-perdie.patch` | [nvzhihanj/DeepGEMM `dsv41-rubin-megamoe-sm107`](https://github.com/nvzhihanj/DeepGEMM/tree/dsv41-rubin-megamoe-sm107) @ `c50dea4`: `e1f418c` + `01eae73` (the sm100-family patch as a commit) + the Rubin MegaMoE commits |
| FlashMLA | `vllm-project/FlashMLA@0eee43b` (`cmake/external_projects/flashmla.cmake`) | `flashmla-cutlass48-sm107.patch` | [nvzhihanj/FlashMLA `dsv41-rubin-sm107`](https://github.com/nvzhihanj/FlashMLA/tree/dsv41-rubin-sm107) @ `43876b7`: `0eee43b` + one commit |
| CUTLASS for SM107 | `NVIDIA/cutlass@v4.8.0` (`cmake/external_projects/cutlass_sm107.cmake`) | none | - |

How the patched trees are used:
- **FlashMLA.** `_flashmla_C` is compiled against the CUTLASS v4.8.0 headers, not FlashMLA's `csrc/cutlass` submodule
  (147f567, which has no SM107). `_flashmla_extension_C` (SM90) keeps the submodule.
- **DeepGEMM.** The CUTLASS 4.8 headers are also installed as `deep_gemm/include/sm107_cutlass`. Only the opt-in
  `DG_MEGA_MOE_SM107_ARCH=107a` JIT target uses them.

Both mappings are exact; the patched tree equals the fork commit with an empty diff:

```bash
V=$PWD   # this vLLM checkout
(git clone -q https://github.com/nvzhihanj/DeepGEMM /tmp/dg && cd /tmp/dg && git checkout -q --detach 01eae73 &&
 git apply $V/cmake/patches/deepgemm-megamoe-sm107-perdie.patch && git add -A && git diff --cached c50dea4 --stat)  # prints nothing
(git clone -q https://github.com/nvzhihanj/FlashMLA /tmp/fmla && cd /tmp/fmla && git checkout -q --detach 0eee43b &&
 git apply $V/cmake/patches/flashmla-cutlass48-sm107.patch && git add -A && git diff --cached 43876b7 --stat)  # prints nothing
```

## 4. Public-sources manifest

`check_public_sources.sh` checks every item below anonymously. It unsets proxies and package-index variables, ignores
the user's git config, and uses no registry credentials. It exits non-zero if anything does not resolve. `--full` also
checks every entry of `image-pip-freeze.txt` and every crate of `rust/Cargo.lock`. That takes ~30 s from a host with
internet access.

**Base images.** Both are pinned by digest in `vllm-build-args.txt`.

| image | digest | public source |
|---|---|---|
| `pytorch/manylinuxaarch64-builder:cuda13.4` (build stages) | `sha256:487d5f86222afdfda9426da96332ca2603ff0c5024c56bd1b363241240ef7a8e` | Docker Hub, `registry-1.docker.io` |
| `nvcr.io/nvidia/cuda-dl-base:26.08-cuda13.4-devel-ubuntu24.04` (runtime) | `sha256:843a5e81ed49ad2a2787aa10e48e926dc1f9440522a122ec2a637c27039fc77d` | NGC, anonymous pull |

The Docker Hub tag `cuda13.4` was re-pushed after the reference build, to `sha256:e629a13f...`. Use the digest.

**Git sources.** All are on github.com. "fc" means fetched by cmake FetchContent; submodules are fetched recursively
unless noted.

| what | repository | pin |
|---|---|---|
| this branch | nvzhihanj/vllm | `7cf5542c25a83d229abfe6779b99d42aa0b5c9bc` |
| release tag for `git describe` | vllm-project/vllm | tag `v0.20.2rc0` = `e6ff3e9c83a6520c3793f4e0511ac8591a07c243` (an ancestor of this branch) |
| CUTLASS (vLLM kernels, fc) | NVIDIA/cutlass | `v4.7.1` = `cb4247394dd8` |
| CUTLASS for SM107 (fc) | NVIDIA/cutlass | `v4.8.0` = `098de2a652cf` |
| FlashMLA (fc) | vllm-project/FlashMLA | `0eee43b12f03`, submodule `csrc/cutlass` NVIDIA/cutlass `147f5673d0c1` |
| DeepGEMM (fc) | vllm-project/DeepGEMM | `e1f418c2a4f2`, submodules NVIDIA/cutlass `f3fde58372d3`, deepseek-ai/DeepJIT `e5bdee2bc4ca` |
| vllm-flash-attn (fc) | vllm-project/flash-attention | `9cd61de38763`, submodules NVIDIA/cutlass `62750a2b75c8`, ROCm/composable_kernel `c56c6750d0fc`, ROCm/aiter `9bab8388c359` (and its ROCm/composable_kernel `af7118e34258`) |
| DeepSelect (fc) | vllm-project/DeepSelect | `d96d33afe1fa`, submodule NVIDIA/cutlass `ae6bccf341fb` |
| FlashKDA (fc) | vllm-project/FlashKDA | `17a037d98da5`, submodule NVIDIA/cutlass `5c149f52a436` |
| fmha_sm100 (fc) | vllm-project/MSA | `f355c37eb4e1`, submodule NVIDIA/cutlass `eb61c9114718` |
| tml-fa4 (fc) | vllm-project/tml-fa4 | `75765e76a9c2` |
| qutlass (fc) | IST-DASLab/qutlass | `e74319e3405c`, submodule NVIDIA/cutlass `b2ca083d2bb9` |
| triton_kernels (fc) | triton-lang/triton | tag `v3.5.1` = `0add68262ab0` |
| Triton wheel (built from source) | triton-lang/triton | `3f6e41132b5edf639bfb872ad73d4688765e08b8` |
| DeepEP wheel (built from source) | deepseek-ai/DeepEP | `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb`, submodule fmtlib/fmt `a4c7e17133ee` |
| Rust frontend git crates | oss-harmony/harmony, smg-project/llm-multimodal | `v0.0.11` = `76e849426cc0`, `f0985ef65967` |

**Python packages.** There are 319 packages in `image-pip-freeze.txt`:
- 302 from **PyPI**;
- 3 from the **PyTorch nightly index** `https://download.pytorch.org/whl/nightly/cu134`;
- 11 from **FlashInfer**: `https://flashinfer.ai/whl/cu134` and `https://flashinfer.ai/whl`;
- 3 built from source in `docker/Dockerfile`: vllm, triton and deep-ep, with the git pins above.

The build also resolves through `https://download.pytorch.org/whl/cu134` and `https://pypi.nvidia.com`.

The explicit pins are in `requirements/rubin-prerelease.txt`, `requirements/cuda.txt` and `Dockerfile.dynamo`:
- `torch==2.15.0.dev20260908+cu134` (aarch64), `torchvision==0.30.0.dev20260909+cu134`,
  `torchaudio==2.11.0.dev20260911+cu134`;
- `flashinfer-python==0.7.0.post1`, `flashinfer-cubin==0.7.0.post1`, `flashinfer-jit-cache==0.7.0.post1+cu134`;
- `nvidia-cutlass-dsl[cu13]==4.8.0.dev0`, `quack-kernels==0.6.5`;
- `cuda-python==13.4.1`, `cuda-bindings==13.4.1`;
- `nixl==1.4.1`, `nvidia-nccl-cu13==2.30.7`;
- `ai-dynamo==1.6.0.dev20260924`, `ai-dynamo-runtime==1.6.0.dev20260924`.

**Other downloads.**

| download | public URL |
|---|---|
| NVSHMEM 3.3.24 for DeepEP | `developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.24_cuda13-archive.tar.xz` |
| Triton's LLVM, `cmake/llvm-info.json` (sha256-checked) | `oaitriton.blob.core.windows.net/public/llvm-builds/llvm-5f07f818-almalinux-arm64-1.tar.gz` |
| Triton's NVIDIA tools, `cmake/nvidia-toolchain-version.json` | `developer.download.nvidia.com/compute/cuda/redist/`: `cuda_nvcc` 12.9.86, `cuda_cuobjdump` / `cuda_nvdisasm` / `cuda_crt` / `cuda_cudart` 13.1.80, `cuda_cupti` 12.8.90 and 13.3.35 (linux-sbsa) |
| Triton's json headers | `github.com/nlohmann/json/releases/download/v3.11.3/include.zip` |
| Rust toolchain 1.95 (`rust-toolchain.toml`) | `sh.rustup.rs`, `static.rust-lang.org` |
| 623 crates (`rust/Cargo.lock`) | crates.io |
| uv, pip bootstrap | `astral.sh/uv/install.sh`, `bootstrap.pypa.io/get-pip.py` |
| OS packages | `ports.ubuntu.com` (noble), `ppa.launchpadcontent.net/deadsnakes`, the AlmaLinux 8 repositories of the manylinux image |

**Floating inputs.** These are resolved at build time from the public sources above and are not pinned by this recipe:
- apt and dnf packages;
- `uv` and `get-pip.py`;
- Python dependencies that the requirements files leave unpinned.

`image-pip-freeze.txt` records the Python set of the reference image. The no-cache verification build of 2026-10-04
reproduced it except for 3 such floating packages (see "Verification" below). For a bit-identical Python set, install
from `image-pip-freeze.txt` as constraints; this recipe does not wire that in.

**PyTorch nightlies.** `download.pytorch.org` prunes old nightly wheels. If torch, torchvision or torchaudio at the
dates above disappear, rebuild them from source at the commits recorded inside the wheels. All three are public, and
`check_public_sources.sh` checks them.

| wheel | source commit (`torch.version.git_version` / `torchvision.version.git_version` / `torchaudio.version.git_version`) |
|---|---|
| `torch 2.15.0.dev20260908+cu134` | pytorch/pytorch `3d2b4c7639df58a777529a4699efa74b00619d90` (nightly-release commit of 2026-09-08 for main `1c9a207306db`; `version.txt` = `2.15.0a0`) |
| `torchvision 0.30.0.dev20260909+cu134` | pytorch/vision `add1dd9ec5d33b983b22b163130c21c01b7dc9fa` |
| `torchaudio 2.11.0.dev20260911+cu134` | pytorch/audio `9b89f15d387b85f740a934c7683482ab39de6c30` |

To read the commits from the image:
```bash
docker run --rm --entrypoint python3 <image> -c \
  'import torch, torchvision, torchaudio as a; print(torch.version.git_version, torchvision.version.git_version, a.version.git_version)'
```
To rebuild:
- Build CUDA 13.4 aarch64 wheels with PyTorch's manylinux builder image (the same `BUILD_BASE_IMAGE`), and
  `USE_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" python setup.py bdist_wheel` at that commit, with
  `PYTORCH_BUILD_VERSION=2.15.0.dev20260908+cu134 PYTORCH_BUILD_NUMBER=1`. Build torchvision and torchaudio the same
  way against it.
- Serve the three wheels from a local directory index and point `requirements/rubin-prerelease.txt`'s
  `--extra-index-url` at it.
- Simplest of all: archive the three wheels while the index still serves them:
  ```bash
  pip download --no-deps --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.12 -d wheels \
    --index-url https://download.pytorch.org/whl/nightly/cu134 \
    torch==2.15.0.dev20260908+cu134 torchvision==0.30.0.dev20260909+cu134 torchaudio==2.11.0.dev20260911+cu134
  ```

**Mirrors and proxies.**
- The dlcluster hosts used for the reference build have dockerd `registry-mirrors` pointing at NVIDIA's internal
  Docker Hub pull-through caches (`/etc/docker/daemon.json`). The Docker Hub base image of the reference build was
  therefore most likely pulled through a mirror. Pulls are content-addressed, so the digest is the same.
- No proxy, `PIP_INDEX_URL`/`UV_*INDEX*` or other index override was set in the build.
- nvcr.io, PyPI, the PyTorch and FlashInfer indexes, GitHub and the NVIDIA redist downloads were reached directly.

To bypass a Docker Hub mirror, name the registry host explicitly; mirrors apply only to `docker.io` references.
Section 5 shows how.

### Verification (2026-10-04)

A no-cache build was run on a fresh dlcluster VR200 node, with:
- a public `git clone` and no user git config;
- an empty `DOCKER_CONFIG`, i.e. anonymous pulls;
- `BUILD_BASE_IMAGE=registry-1.docker.io/pytorch/manylinuxaarch64-builder:cuda13.4@sha256:487d5f86...` (no mirror);
- `--no-cache --pull`;
- no push.

Result:
- The build succeeded: 2391 s for `vllm-openai`, plus 75 s for Dynamo.
- Compared with the reference image, every vLLM Python file is identical (hash of the tree), and so are the compiled
  extension set, the 584 dpkg packages and the torch, torchvision and torchaudio git commits.
- The two pip freezes have the same 319 entries. They differ in 3 floating dependencies (`uv` 0.12.22 -> 0.12.23,
  `websockets` 17.1 -> 17.2, `zipp` 4.1.0 -> 4.1.1) and in vLLM's `.d<date>` suffix.
- Image size is 18.44 GB in both.

## 5. Build

**Host:**
- native linux/arm64: Grace or Vera, e.g. a VR200 or GB200/GB300 node;
- Docker >= 24 with the buildx plugin (BuildKit; the Dockerfile uses cache, secret and bind mounts);
- outbound internet access;
- >= 200 GB free in the docker root. The final image is ~18.4 GB uncompressed; the build stages and cache take
  several times that.
- Reference: a 176-core / 706 GB Vera node, `max_jobs=32 nvcc_threads=2`, ~40-45 min cold, with no GPU needed. On
  smaller hosts, lower `max_jobs` in `vllm-build-args.txt`.

```bash
# 1. Source. Use --no-tags: the fork also carries upstream tags such as v0.30.1rc0, which git describe would pick,
#    and the wheel version would become 0.30.1rc1.dev573. Fetch only the release tag the version is based on.
git clone --no-tags --branch dsv41-flash-rubin-opt https://github.com/nvzhihanj/vllm.git vllm-dsv41
cd vllm-dsv41
git checkout --detach <commit>          # 7cf5542c25 = the reference image, or the README commit
git fetch --no-tags https://github.com/vllm-project/vllm.git refs/tags/v0.20.2rc0:refs/tags/v0.20.2rc0
git describe --tags --match 'v[0-9]*'   # v0.20.2rc0-6106-g7cf5542c25 for the reference image

# 2. Optional: every input is public (~20 s; --full ~30 s)
docker/dsv41-rubin/check_public_sources.sh

# 3. Build (and optionally push). The tag is yours. BUILDX_FLAGS="--no-cache --pull" forces a from-scratch build.
VLLM_SRC=$PWD docker/dsv41-rubin/build.sh <registry>/<repo>:<tag> [--push]
```

What `build.sh` runs:
1. `docker buildx build --platform linux/arm64 --target vllm-openai --build-arg=<each line of vllm-build-args.txt>
   --build-arg VLLM_BUILD_COMMIT=<sha> -f docker/Dockerfile .` produces `<tag>-vllm`, local only.
2. `docker buildx build --build-arg VLLM_IMAGE=<tag>-vllm -f docker/dsv41-rubin/Dockerfile.dynamo docker/dsv41-rubin`
   produces `<tag>`.
3. `docker push <tag>`, only with `--push`.

`build.sh` refuses a dirty tree, or a tree where no `v[0-9]*` tag is reachable. `GIT_REPO_CHECK=1` needs the tag, and
setuptools-scm derives the wheel version from it.

Building from a later README-only commit gives the same image except the version metadata, `dev6107+g<sha>` and
`VLLM_BUILD_COMMIT`.

The `.d<date>` suffix of the version is expected. The Rubin path of the `build` stage rewrites tracked requirement
files (`use_existing_torch.py` and a `sed` of the flashinfer/cutlass-dsl pins) before `bdist_wheel`, so setuptools-scm
sees a dirty tree.

**Docker Hub mirror on the build host:** use a copy of the args file with the registry spelled out:
```bash
sed 's#^BUILD_BASE_IMAGE=#BUILD_BASE_IMAGE=registry-1.docker.io/#' docker/dsv41-rubin/vllm-build-args.txt > /tmp/args.txt
BUILD_ARGS_FILE=/tmp/args.txt VLLM_SRC=$PWD docker/dsv41-rubin/build.sh <tag>
```

**Arch choice.** Everything is built for arch `10.0`, which is family-portable sm_100f code that runs on sm_107.
`torch_cuda_arch_list` containing `10.7` fails ptxas on `attn_res_kernel` (tcgen05 on sm_107). Rubin-native code is
added per kernel instead:
- FlashMLA's `_flashmla_C` gets a 10.7a cubin;
- DeepGEMM JIT-compiles at runtime, for sm_100f by default (`deepgemm-sm107-sm100-family.patch`).

**Stage timings of the reference build:**

| stage | time |
|---|---|
| base image pulls | ~2-4 min |
| base Python requirements, incl. z3-solver and tilelang sdist builds | ~8-9 min |
| `csrc-build` | ~19 min |
| Triton from source | ~4-6 min |
| DeepEP | ~1.5 min |
| wheel | ~1 min |
| runtime stages | ~6 min |
| Dynamo layer | ~1 min |
| push | ~3 min |

Do not count on the BuildKit cache surviving between builds on shared hosts.

**Pre-flight for cmake changes (~3 min).** If you change anything under `cmake/`, configure and build only the touched
targets inside an existing image of this recipe before the full build:
```bash
docker run --rm -v $PWD:/src:ro --entrypoint bash <image> -c '
  cp -r /src /tmp/v && cd /tmp/v && rm -rf .git && pip install -q cmake ninja &&
  TORCH_CUDA_ARCH_LIST=10.0 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH=$(python3 -c "import sys; print(\":\".join(sys.path))") \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc &&
  cmake --build build --target _flashmla_C _deep_gemm_C'
```
Two notes:
- The runtime image lacks the elfutils (libdw) headers that DeepGEMM's `_C` includes. Install `libdw-dev`, or add the
  headers via `CPLUS_INCLUDE_PATH`, for `_deep_gemm_C`.
- The configure step prints `FlashMLA _flashmla_C CUDA architectures: 10.0f;10.7a` and
  `DeepGEMM SM107 JIT target: SM100 family; Rubin MegaMoE kernel: opt-in`.

## 6. Validate

**6.1 Inventory and GPU smoke** (any VR200 node, one GPU, no `PYTHONPATH` overlay). Run it from outside the source tree
so that `import vllm` resolves to the image's copy:
```bash
docker run --rm --gpus all -v $PWD:/src:ro -w /tmp --entrypoint python3 <image> /src/docker/dsv41-rubin/validate_image.py
```
It checks:
- the vLLM, torch, Triton, FlashInfer, cutlass-dsl, transformers and Dynamo versions;
- cc (10, 7);
- `_flashmla_C` cubins `{sm_100: 51, sm_107a: 51}`, and that FlashMLA mega/sparse attention is supported;
- that the vendored deep_gemm has the 6 `sm107_*` bindings, the SM107 MegaMoE header and CUTLASS 4.8 in
  `include/sm107_cutlass`;
- 2 locality domains with a balanced SM map (106 + 106 on VR200), and a die-local weight copy whose content is
  preserved;
- that the dense opt-in modules import, and that Dynamo imports.

Expected last line: `== ALL OK`.

On Slurm with Pyxis, the equivalent is
`srun --container-image=<sqsh> --container-mounts=$PWD:/src --container-workdir=/tmp python3 /src/docker/dsv41-rubin/validate_image.py`.

**6.2 Kernel unit tests** for the Rubin paths. The image has no pytest; install it in a throwaway container:
```bash
docker run --rm --gpus all --ipc=host -v $PWD:/src:ro -w /tmp --entrypoint bash <image> -c '
  pip install -q pytest tblib && mkdir /tmp/r && cp -r /src/tests /tmp/r/ && cd /tmp/r &&
  python3 -m pytest -q -p no:cacheprovider -k "not flashinfer" \
    tests/kernels/attention/test_flashmla_sparse.py tests/kernels/attention/test_dsv41_candidate_topk.py \
    tests/kernels/quantization/test_mxfp8_rubin_gemm.py tests/kernels/quantization/test_mxfp8_triton_swizzled_quant.py'
```
Reference result: 67 passed.
- The 9 `test_flashinfer_*` cases in `test_flashmla_sparse.py`, deselected above, fail on any Rubin image.
  FlashInfer's `trtllm_batch_decode_sparse_mla_dsv4` supports SM100/SM103/SM12x only. That is also why DSV4.1 on
  Rubin uses the `FLASHMLA_MEGA_ATTN_DSV41` backend.
- Run from a copy of `tests/` outside the source tree, as above. Otherwise the tests import the uncompiled source
  `vllm`.

**6.3 Which FlashMLA cubin runs (optional).** On Rubin the driver should run the sm_107a cubin. Its decode epilogue uses
`tcgen05.ld.red`, which the SASS shows as `LDTM.STAT`; the sm_100f cubin has none. Profile one mega-attention decode
launch with ncu (the image ships `/usr/local/bin/ncu`):
- `--kernel-name-base demangled -k regex:core_attn --launch-count 1 --section SourceCounters`
- then grep the SASS source page for `LDTM.STAT`.

Reference: 8 rows with this image, 0 with an sm_100f-only build.

**6.4 Serving smoke.** Run DSV4.1-Flash, DEP4 on one 4-GPU VR200 node, C2048, with a short dataset run. Use the
mlperf-endpoints VR200 serving config for DeepSeek-V4.1-Flash, without any vLLM overlay, and set the flags in
section 7.

Reference results (DATASET_REPEATS=2, memclk pinned, NUMA-bound workers):
- capture succeeds on all 4 workers;
- worker logs show `DeepGEMM MegaMoE: Rubin SM107 kernel enabled (DG_MEGA_MOE_SM107=1)`,
  `per-die execution with die-local routed expert weights`, and `vllm_mxfp8_sm107_sm107a_cute_dsl` kernels compiling;
- 8084/8084 turns, 0 failed;
- inline accuracy 0.535, mean OSL 882. The official gates are inline >= 52.36 and OSL in [793, 970].

Two-repeat runs are noisy. Full-length official runs of this image (C4096, 16 repeats) measured inline 53.2.

Reference A/B results (VR200, same node):
- FlashMLA in this image vs the sm_100f-only production build: fused mega-attention outputs bitwise identical. Only
  the unused ops differ, from fast-math logf: sparse lse <= 9.5e-7, dense FMHA out <= 9.8e-4. Decode -4..-11% and
  prefill -7..-13% per call.
- MegaMoE (EP4, shared expert): base / SM107 / SM107 + per-die at T/rank 1024: 239.5 / 230.4 / 220.8 us; at 8192:
  940.4 / 917.8 / 904.7 us.

## 7. Runtime configuration (DSV4.1-Flash, DEP4 on 4x VR200)

- **Workers.** These enable the Rubin paths; all are opt-in:
  ```
  VLLM_DSV41_GRAPH_BOUNDED_REPLAY=1 VLLM_ENGINE_DEFER_FULL_GC=1 VLLM_DSV41_ENGRAM_ALL_TO_ALL=1
  VLLM_DSV41_MEGAMOE_SM107=1 VLLM_DSV41_MEGAMOE_PERDIE=routed
  VLLM_MXFP8_FI_LARGE_M_BACKEND=rubin VLLM_MXFP8_TRITON_QUANT=1 VLLM_DSV41_FAST_CANDIDATE_TOPK=1
  ```
  - `DG_JIT_SM107_USE_SM100F=1` (as in the GB300 recipe) is harmless.
  - vLLM warns `Unknown vLLM environment variable detected` for the last three dense flags. That is expected: those
    modules read the variables directly.
- **Attention.** `--attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41",...}'`. `FLASHINFER_MLA_SPARSE_DSV41` does
  not support sm_107.
- **MoE.** `--kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'`.
- **Dynamo frontend.** `DYN_TOKENIZER_CACHE_BYTES=17179869184 DYN_TOKENIZER_CACHE_EXTEND=1`. At C2048+ the frontend
  tokenizer otherwise competes for CPU.
- **Host.**
  - Bind each DP rank (dynamo.vllm, EngineCore, workers) to its GPU's Vera socket with `numactl --cpunodebind
    --membind`. GPU0/1 are on node 0 and GPU2/3 on node 1.
  - Pin the HBM clock: `nvidia-smi -lmc 4752,4752` on the host.

## 8. Known issues

- **Per-die MegaMoE.**
  - Not compatible with EPLB: `get_expert_weights` raises.
  - Needs ~1.7 GB per rank transiently at load, while the routed weights are re-homed.
  - `VLLM_DSV41_MEGAMOE_PERDIE=1` also localizes the shared expert. That is no faster, and costs ~35 MiB per layer.
- **Overlays.** Do not put a python overlay in front of this image that carries compiled files (`_flashmla_C`,
  `third_party/deep_gemm`) from another image. It silently brings back the old FlashMLA and DeepGEMM. Check `md5sum`
  of `_flashmla_C.abi3.so` against the image's copy.
- **`sm107_cutlass` headers.** `setup.py` `package_data` ships only `*.h/*.hpp/*.cuh` under `deep_gemm/include`, so
  the 45 CUTLASS `.inl` files (collective builders) are missing from `sm107_cutlass`, as from the default vendored
  CUTLASS. DeepGEMM kernels do not include them.
- **Dynamo constraints.** `Dockerfile.dynamo` pins torch and transformers by version. vLLM is installed from a local
  wheel (`vllm @ file://...`), so `uv pip freeze` cannot pin it the same way. Plain `ai-dynamo` does not depend on
  vllm, and the post-install check confirms torch and transformers are unchanged. The install upgrades protobuf to
  7.36.2.
- **Benign log noise.** `import vllm` prints a `nixl_ep_cu13 ... ModuleNotFoundError` traceback for the optional
  nixl EP module, and in-container torchrun can print deep_ep "Duplicate NCCL runtime". Both are benign.
- **DeepGEMM JIT.** DeepGEMM compiles at runtime with the image's CUDA toolkit, cached in `DG_JIT_CACHE_DIR`, by
  default `~/.deep_gemm`. Use a cache directory per image, so kernels compiled from another image's headers are never
  reused.

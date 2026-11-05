# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# FLASHINFER_PATCH embeds C++ source lines verbatim.
# ruff: noqa: E501
"""Single-CTA post-top-K routing permutation for FlashInfer's trtllm fused MoE.

FlashInfer's trtllm fused MoE routes from logits in two steps for more than 16
tokens: a score kernel writes packed ``(bf16 score, int16 expert)`` top-K words
and a permutation kernel turns them into the grouped-GEMM metadata (per-expert
counts, CTA tables, permutation maps). For 17-256 tokens the permutation is a
cluster kernel (8 CTAs x 1024 threads, cluster barriers and DSMEM reads), above
that a cooperative kernel (grid-wide sync) or a histogram + offsets pair.

:data:`FLASHINFER_PATCH` replaces them with one cluster-free 1024-thread CTA
(SMEM histogram, block scan, parallel CTA-table fill) for unpacked ids
(``mPtrTopKIds``) and for packed top-K, both in the
post-top-K pipeline (17-256 tokens) and after the large-batch score kernels
(> 256 tokens), as long as ``numTokens * topK <= GS2_ROUTE_MAXN``. All routing
metadata except the within-expert row order (nondeterministic in the stock
atomics-based kernels too) is identical and the packed scores are copied to the
expert weights as before, so MoE outputs are unchanged. The kernel waits on its
producer before touching global memory and triggers its dependents after its
last write, which makes it PDL-safe (unlike the cluster / cooperative launches
FlashInfer PR #4806 disabled MoE PDL for on SM107).

When enabled, the patch is applied to the installed FlashInfer sources and
FlashInfer's trtllm fused-MoE JIT module is built from them under a renamed
module name (``<stock name>_<GS2_ROUTE_TAG>``), so neither the stock JIT build
directory nor the prebuilt AOT module of the stock name is reused.

Environment:

* ``GS2_ROUTE`` (default ``0``): any other value enables the patched module.
  The patched C++ also reads it on every call (``GS2_ROUTE=0`` at runtime
  selects the stock kernels inside the patched module).
* ``GS2_ROUTE_TAG`` (default ``exact_routing``): suffix of the module name.
* ``GS2_ROUTE_PREBUILT``: path of the patched module's ``.so`` built from the
  same FlashInfer install (e.g. once per image); loaded instead of JIT-building.
* ``VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE``: optional persistent directory
  for the JIT-built module (ignored with ``GS2_ROUTE_PREBUILT``). FlashInfer
  builds JIT modules inside its workspace, which serving launchers often make
  fresh per process (minutes of nvcc per start for this module). With this
  set, the built ``.so`` is copied to ``<dir>/<module>-<input hash>/<module>.so``
  and later processes whose inputs hash the same (source bytes, compiler/linker
  flags, include dirs, FlashInfer version, device capability) load it instead
  of rebuilding. Start-up only: the loaded code is the code the build would
  produce.
* Read by the patched C++: ``GS2_ROUTE_MAXN`` (max ``numTokens * topK``,
  default 8192 = 1024 tokens at top-8, capped at 32768; above
  that the single CTA was slower than the cooperative kernel on VR), ``GS2_ROUTE_MINTOK`` (default 17),
  ``GS2_ROUTE_PACKED`` (default 1: packed top-K input), ``GS2_ROUTE_LARGE``
  (default 1: also after the large-batch score kernels), ``GS2_ROUTE_AGG``
  (warp-aggregated atomics), ``GS2_ROUTE_LOG``.
* ``VLLM_MOE_PDL_FC`` (default ``0``): ``1`` additionally builds the module
  with :data:`PDL_FC_PATCH` applied to ``trtllm_fused_moe_kernel_launcher.cu``
  (module name suffix ``_pdlfc``): every ``routing_runner.run`` call passes
  ``enable_pdl=false``, so routing / permutation kernels are never
  PDL-launched and never wait on or trigger a programmatic edge, while FC1 /
  FC2 keep the caller's ``enable_pdl``. FlashInfer's sm_107 MoE PDL clamp
  (``_device_support_moe_pdl``) is bypassed only after that patched module has
  been generated; until then (or if the patch does not apply) MoE PDL stays
  off. Launch attributes only: numerics are unchanged. Works with or without
  ``GS2_ROUTE``; ignored (with a warning) when
  ``VLLM_FI_SM107_MOE_PDL_MAX_TOKENS > 0`` (per-call MoE PDL) or
  ``GS2_ROUTE_PREBUILT`` is set (that module lacks the launcher patch).
* ``VLLM_H82_ROUTE_EARLY`` (default ``0``): non-zero
  with ``GS2_ROUTE`` and no ``GS2_ROUTE_PREBUILT`` also applies
  :data:`H82_ROUTING_PATCH` (module name suffix ``_h82``). A mask of the
  routing kernels that trigger their PDL dependent right after their own
  grid-dependency wait instead of after their last write (FlashInfer PR #6449
  does this for the static block kernel): ``2`` the block-per-token score
  kernel (17-256 tokens; dependent: the single-CTA permutation), ``1`` the
  static / dynamic block kernels and the single-CTA permutation (dependent:
  FC1), ``3`` both. Launch order only: every dependent waits on the grid
  before reading its outputs, so outputs are unchanged. The patched C++ reads
  it on every call and copies it to a device flag outside graph capture.
* ``VLLM_H158_LEAN_ROUTE`` (default ``0``; H158): non-zero
  together with an ``_h82`` module also applies :data:`H158_ROUTING_PATCH`
  (module name suffix ``_h158``): logits-routed calls of
  ``VLLM_H158_LEAN_ROUTE_MIN`` (default 1) to ``VLLM_H158_LEAN_ROUTE_MAX``
  (default and at most 32) tokens run one lean single-CTA kernel instead of
  the dyn-block kernel (<= 16 tokens) or the block-per-token score kernel plus
  the single-CTA permutation (17+). Its top-K is the dyn-block kernel's code
  (one warp per token, same ``ExpertSelectPolicy::apply``), so ids and
  weights are bit-identical to both stock paths, and its permutation keeps the
  dyn-block kernel's token order. The patched C++ re-reads
  ``VLLM_H158_LEAN_ROUTE`` on every call (``0`` selects the stock kernels).

FlashInfer adapter: FlashInfer builds the module through the module-level
function ``flashinfer.fused_moe.core.gen_trtllm_gen_fused_moe_sm100_module``
(cached by ``_get_trtllm_moe_sm100_module_impl``) and offers no way to pass
different sources, so :func:`maybe_install` replaces that function by one that
returns the renamed spec with the patched sources and clears the cache. It must
run before the first trtllm fused-MoE call; vLLM calls it when MoE layers are
constructed. The proper home of the change is the FlashInfer source itself.
"""

import dataclasses
import hashlib
import os
import tempfile
from pathlib import Path

import regex as re

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("GS2_ROUTE", "0") != "0"
TAG = os.environ.get("GS2_ROUTE_TAG", "exact_routing")
PREBUILT = os.environ.get("GS2_ROUTE_PREBUILT")

# Mirrors of the C++ defaults (trtllm_fused_moe_routing_common.cu, gs2::Cfg) and of
# FlashInfer's routingCustom::run dispatch (trtllm_fused_moe_routing_custom.cu).
_MAX_ENTRIES_CAP = 32 * 1024
_MAX_EXPERTS = 1024
_SINGLE_CLUSTER_MAX_TOKENS = 256  # MaxNumTokensSingleClusterScores
_SPLIT_TOPK_MIN_EXPERTS = 160  # NumExperts160Experts


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def single_cta_routing_covers(num_tokens: int, num_experts: int, top_k: int) -> bool:
    """Whether the loaded patched module runs this logits-routed
    (routingCustom, e.g. Renormalize) MoE call's permutation as the single-CTA
    kernel, i.e. no Cluster/cooperative routing kernel runs.

    1-16 tokens use FlashInfer's fused Block/DynBlock kernels (no separate
    permutation): False. 17-256 tokens reach the patched post-top-K pipeline
    only on FlashInfer's split-top-K path (>= 160 experts, or forced by
    FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN=1/on; never with =0/off);
    otherwise the fused score + permutation Cluster kernel runs. Above 256
    tokens the large-batch hook covers every expert count <= 1024.
    """
    if not _installed or _env_int("GS2_ROUTE", 1) == 0 or top_k <= 0:
        return False
    if _env_int("GS2_ROUTE_PACKED", 1) == 0 or not 0 < num_experts <= _MAX_EXPERTS:
        return False
    if num_tokens < max(_env_int("GS2_ROUTE_MINTOK", 17), 17):
        return False
    max_n = min(_env_int("GS2_ROUTE_MAXN", 8 * 1024), _MAX_ENTRIES_CAP)
    if num_tokens * top_k > max_n:
        return False
    if num_tokens > _SINGLE_CLUSTER_MAX_TOKENS:
        return _env_int("GS2_ROUTE_LARGE", 1) != 0
    force = os.environ.get("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN", "")
    if force in ("0", "off", "OFF"):
        return False
    return num_experts >= _SPLIT_TOPK_MIN_EXPERTS or force in ("1", "on", "ON")


# sha256 of the FlashInfer 0.6.18.post1 sources the patch was generated against.
_BASE_SOURCE_SHA256 = {
    "trtllm_fused_moe_routing_common.cu": (
        "182f493e45feea4d7fd3cc2c20f76195d6477123934e1dd3d1561ebaaf5342df"
    ),
    "trtllm_fused_moe_routing_custom.cu": (
        "a04638539d3deb9d7e224ab8f7673a6a87eadfedf59204e6b21c39885f77cd6a"
    ),
}

# Routing-never-PDL launcher (VLLM_MOE_PDL_FC=1): FC1 / FC2 PDL-launched,
# routing never.
LAUNCHER_SOURCE_NAME = "trtllm_fused_moe_kernel_launcher.cu"
# sha256 of the FlashInfer 0.6.18.post1 launcher PDL_FC_PATCH was generated against.
_LAUNCHER_BASE_SHA256 = "7b7adacc9117eb64869bb68b96818af9588a85651afd83cb4a27abaae54152bf"
_PDL_MARK = b"[vllm moe-pdl-fc] routing never PDL"
# Set once gen() has produced the launcher-patched module spec; the clamp
# bypass returns False until then, so routing can never run PDL-launched from
# an unpatched module.
_PDL_MODULE_READY = False

_HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_RE = re.compile(rb"^\+\+\+ b/(\S+)")

# Whether the GS2_ROUTE module is installed (single_cta_routing_covers).
_installed = False
_pdl_fc_installed = False
_h82_installed = False
_h158_installed = False


def h158_lean_route() -> bool:
    """``VLLM_H158_LEAN_ROUTE``: build the module with :data:`H158_ROUTING_PATCH`
    (lean single-CTA routing kernel; needs the ``_h82`` module)."""
    return os.environ.get("VLLM_H158_LEAN_ROUTE", "0") not in ("", "0")


def h82_route_early() -> int:
    """``VLLM_H82_ROUTE_EARLY`` mask (0..3): early PDL trigger in the routing
    kernels (1: those whose dependent is FC1; 2: the score kernel)."""
    try:
        return int(os.environ.get("VLLM_H82_ROUTE_EARLY", "0")) & 3
    except ValueError:
        return 0


def _split_lines(data: bytes) -> list[bytes]:
    parts = data.split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def split_patch(diff: bytes) -> dict[str, bytes]:
    """Split a multi-file unified diff into ``{basename: single-file diff}``."""
    out: dict[str, list[bytes]] = {}
    cur: list[bytes] | None = None
    for line in _split_lines(diff):
        m = _FILE_RE.match(line)
        if m is not None:
            cur = out.setdefault(Path(m.group(1).decode()).name, [])
            continue
        if cur is not None and not line.startswith(b"--- "):
            cur.append(line)
    return {k: b"".join(v) for k, v in out.items()}


def apply_unified_diff(original: bytes, diff: bytes) -> bytes:
    """Apply a single-file unified diff to ``original``; context must match
    exactly (no fuzz, no offsets). Text before the first hunk is ignored.
    """
    src = _split_lines(original)
    lines = _split_lines(diff)
    out: list[bytes] = []
    pos = 0
    i = 0
    n = len(lines)
    while i < n and not lines[i].startswith(b"@@"):
        i += 1
    if i == n:
        raise ValueError("patch contains no hunks")
    while i < n:
        m = _HUNK_RE.match(lines[i])
        if m is None:
            raise ValueError(f"unexpected patch line {i + 1}: {lines[i]!r}")
        old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
        new_len = int(m.group(4) or 1)
        start = old_start - 1 if old_len > 0 else old_start
        if start < pos or start > len(src):
            raise ValueError(f"hunk at patch line {i + 1} is out of order")
        out.extend(src[pos:start])
        pos = start
        i += 1
        old_rem, new_rem = old_len, new_len
        while old_rem > 0 or new_rem > 0:
            if i >= n:
                raise ValueError("truncated hunk")
            line = lines[i]
            i += 1
            tag, body = line[:1], line[1:]
            if tag in (b" ", b"-"):
                if pos >= len(src) or src[pos] != body:
                    raise ValueError(f"patch context mismatch at source line {pos + 1}")
                pos += 1
                old_rem -= 1
                if tag == b" ":
                    out.append(body)
                    new_rem -= 1
            elif tag == b"+":
                out.append(body)
                new_rem -= 1
            else:
                raise ValueError(f"unsupported patch line {i}: {line!r}")
    out.extend(src[pos:])
    return b"".join(out)


def _write_if_changed(path: Path, data: bytes) -> None:
    # Keep the file (and its mtime) when the content is unchanged so the JIT
    # build stays incremental; replace atomically otherwise.
    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def patched_sources(
    stock_sources: list[Path],
    out_dir: Path,
    extra_patch: str | tuple[str, ...] | None = None,
) -> dict[Path, Path]:
    """Return ``{stock source: patched source}`` for every file the patch
    touches, generating the patched files in ``out_dir``. ``extra_patch`` (a
    diff against the patched files, e.g. :data:`H82_ROUTING_PATCH`, or a tuple
    of diffs applied in order, each against the result of the previous one) is
    applied in memory on top, so each file is written once (unchanged content
    keeps its mtime and the JIT build incremental).
    """
    per_file = split_patch(FLASHINFER_PATCH.encode())
    if isinstance(extra_patch, str):
        extra_patch = (extra_patch,)
    extras = [split_patch(p.encode()) for p in extra_patch or ()]
    by_name = {Path(p).name: Path(p) for p in stock_sources}
    missing = set(per_file) - set(by_name)
    for extra in extras:
        missing |= set(extra) - set(per_file)
    if missing:
        raise RuntimeError(f"exact MoE routing: {sorted(missing)} not in spec sources")
    result: dict[Path, Path] = {}
    for name, diff in per_file.items():
        stock = by_name[name]
        original = stock.read_bytes()
        if hashlib.sha256(original).hexdigest() != _BASE_SOURCE_SHA256.get(name):
            logger.warning(
                "exact MoE routing: %s differs from the FlashInfer source the "
                "patch was generated against; applying it with exact context "
                "matching",
                stock,
            )
        out = out_dir / name
        data = apply_unified_diff(original, diff)
        for extra in extras:
            if name in extra:
                data = apply_unified_diff(data, extra[name])
        _write_if_changed(out, data)
        result[stock] = out
    return result


def patched_launcher_source(stock_source: Path, out_dir: Path) -> Path:
    """Return the path of the launcher with :data:`PDL_FC_PATCH` applied
    (routing never PDL), generating it in ``out_dir`` from ``stock_source``.
    Raises if the patch does not apply exactly or does not cover every
    ``routing_runner.run`` call.
    """
    original = stock_source.read_bytes()
    if hashlib.sha256(original).hexdigest() != _LAUNCHER_BASE_SHA256:
        logger.warning(
            "[moe-pdl-fc] %s differs from the FlashInfer source the launcher "
            "patch was generated against; applying it with exact context matching",
            stock_source,
        )
    patched = apply_unified_diff(original, PDL_FC_PATCH.encode())
    n_sites = patched.count(b"routing_runner.run(")
    n_marked = patched.count(_PDL_MARK)
    if n_sites == 0 or n_marked != n_sites:
        raise RuntimeError(
            f"[moe-pdl-fc] launcher patch covers {n_marked} of {n_sites} "
            "routing_runner.run call sites; refusing to enable MoE PDL"
        )
    out = out_dir / LAUNCHER_SOURCE_NAME
    _write_if_changed(out, patched)
    return out


def _moe_pdl_fc_supported(device) -> bool:
    """Replacement for FlashInfer's ``_device_support_moe_pdl`` under
    VLLM_MOE_PDL_FC=1: PDL for the MoE runner (FC1 / FC2) wherever the device
    supports PDL, but only once the routing-never-PDL module is in use."""
    if not _PDL_MODULE_READY:
        return False
    from flashinfer.fused_moe import core

    return bool(core.device_support_pdl(device))


def _module_cache_key(spec) -> str:
    import flashinfer
    import torch
    from flashinfer.jit import env as jit_env

    # Paths inside the (per-process) FlashInfer workspace are normalised, so the
    # key does not depend on where the workspace lives; the generated headers
    # there are a function of the FlashInfer version and the module flags.
    ws = str(jit_env.FLASHINFER_WORKSPACE_DIR)
    h = hashlib.sha256()
    cap = torch.cuda.get_device_capability()
    h.update(f"{spec.name}|{flashinfer.__version__}|{cap}".encode())
    for p in spec.sources:
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    groups = (
        "extra_cflags",
        "extra_cuda_cflags",
        "extra_ldflags",
        "extra_include_dirs",
    )
    for group in groups:
        for item in getattr(spec, group, None) or []:
            h.update(f"{group}={str(item).replace(ws, '<ws>')}".encode())
    return h.hexdigest()[:16]


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(src.read_bytes())
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _with_module_cache(spec):
    """Return ``spec`` whose build output is shared through
    ``VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE`` (see the module docstring).
    """
    cache_dir = envs.VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE
    if not cache_dir:
        return spec
    cached = (
        Path(cache_dir) / f"{spec.name}-{_module_cache_key(spec)}" / f"{spec.name}.so"
    )

    class _CachedSpec(type(spec)):
        def try_load(self):
            if cached.is_file():
                try:
                    # Materialise the cached build at the module's JIT path:
                    # FlashInfer also opens that path directly (cubin loader).
                    dst = Path(self.jit_library_path)
                    _copy_atomic(cached, dst)
                    module = self.load(dst)
                    logger.info(
                        "exact MoE routing: loaded %s from %s", self.name, cached
                    )
                    return module
                except Exception as e:  # corrupt / incompatible copy: rebuild
                    logger.warning(
                        "exact MoE routing: cannot load %s (%r); rebuilding", cached, e
                    )
            return super().try_load()

        def build(self, *args, **kwargs):
            super().build(*args, **kwargs)
            try:
                _copy_atomic(Path(self.jit_library_path), cached)
                logger.info("exact MoE routing: cached %s at %s", self.name, cached)
            except Exception as e:  # the module still loads from the workspace
                logger.warning(
                    "exact MoE routing: could not cache %s (%r)", self.name, e
                )

    out = object.__new__(_CachedSpec)
    out.__dict__.update(spec.__dict__)
    return out


def _pdl_fc_enabled() -> bool:
    """``VLLM_MOE_PDL_FC``, unless a setting it cannot compose with is set."""
    if not envs.VLLM_MOE_PDL_FC:
        return False
    if envs.VLLM_FI_SM107_MOE_PDL_MAX_TOKENS > 0:
        # Both control FlashInfer's _device_support_moe_pdl: per call for
        # VLLM_FI_SM107_MOE_PDL_MAX_TOKENS, globally here.
        logger.warning_once(
            "[moe-pdl-fc] VLLM_MOE_PDL_FC ignored: "
            "VLLM_FI_SM107_MOE_PDL_MAX_TOKENS > 0 selects the per-call MoE PDL"
        )
        return False
    if ENABLED and PREBUILT:
        logger.warning_once(
            "[moe-pdl-fc] VLLM_MOE_PDL_FC ignored: the GS2_ROUTE_PREBUILT "
            "module is built without the routing-never-PDL launcher"
        )
        return False
    return True


def maybe_install() -> None:
    """Build (or load the prebuilt) FlashInfer trtllm fused-MoE module from the
    patched routing sources (``GS2_ROUTE``) and / or the routing-never-PDL
    launcher (``VLLM_MOE_PDL_FC=1``). No-op unless one of them is enabled;
    idempotent.
    """
    global _installed, _pdl_fc_installed, _h82_installed, _h158_installed
    if _installed or _pdl_fc_installed:
        return
    pdl_fc = _pdl_fc_enabled()
    if not (ENABLED or pdl_fc):
        return
    from flashinfer.fused_moe import core
    from flashinfer.jit import env as jit_env

    prebuilt = Path(PREBUILT) if ENABLED and PREBUILT else None
    h82 = ENABLED and h82_route_early() != 0
    if h82 and prebuilt is not None:
        logger.warning(
            "H82 routing: VLLM_H82_ROUTE_EARLY ignored: the GS2_ROUTE_PREBUILT "
            "module is built without H82_ROUTING_PATCH"
        )
        h82 = False
    h158 = h82 and h158_lean_route()
    if h158_lean_route() and not h82:
        logger.warning(
            "H158 routing: VLLM_H158_LEAN_ROUTE ignored: it needs the _h82 module "
            "(GS2_ROUTE, VLLM_H82_ROUTE_EARLY non-zero, no GS2_ROUTE_PREBUILT)"
        )
    if prebuilt is not None and envs.VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE:
        logger.warning(
            "exact MoE routing: GS2_ROUTE_PREBUILT is set; ignoring "
            "VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE"
        )
    if prebuilt is not None and not prebuilt.is_file():
        raise FileNotFoundError(prebuilt)
    orig = core.gen_trtllm_gen_fused_moe_sm100_module

    def gen(enable_rubin=False):
        global _PDL_MODULE_READY
        spec = orig(enable_rubin=enable_rubin)
        name = f"{spec.name}_{TAG}" if ENABLED else spec.name
        if h82:
            name = f"{name}_h82"
        if h158:
            name = f"{name}_h158"
        if pdl_fc:
            name = f"{name}_pdlfc"
        out_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "vllm_patched" / name
        srcs = list(spec.sources)
        mapping: dict[Path, Path] = {}
        if ENABLED:
            extra: tuple[str, ...] = ()
            if h82:
                extra = (H82_ROUTING_PATCH,)
            if h158:
                extra = (*extra, H158_ROUTING_PATCH)
            mapping = patched_sources(srcs, out_dir, extra)
            srcs = [mapping.get(Path(p), p) for p in srcs]
        lsrc = None
        if pdl_fc:
            stock_l = [p for p in srcs if Path(p).name == LAUNCHER_SOURCE_NAME]
            assert stock_l, f"{LAUNCHER_SOURCE_NAME} not in spec sources"
            lsrc = patched_launcher_source(Path(stock_l[0]), out_dir)
            srcs = [lsrc if Path(p).name == LAUNCHER_SOURCE_NAME else p for p in srcs]
        new = dataclasses.replace(spec, name=name, sources=srcs)
        if prebuilt is not None:
            # JitSpecNvcc loads ``aot_path`` without building when it exists.
            new = dataclasses.replace(
                _prebuilt_spec_cls(type(spec))(
                    **{f.name: getattr(new, f.name) for f in dataclasses.fields(new)}
                ),
                prebuilt=prebuilt,
            )
        else:
            # The persistent cache would load before the prebuilt module and
            # shadow it, so it only applies to JIT builds.
            new = _with_module_cache(new)
        if ENABLED:
            logger.info(
                "exact MoE routing: module %s -> %s (%s), patched sources %s",
                spec.name,
                new.name,
                f"prebuilt {prebuilt}" if prebuilt is not None else "JIT",
                sorted(str(p) for p in mapping.values()),
            )
        if h82:
            logger.info(
                "H82 routing: module %s built with H82_ROUTING_PATCH: early PDL "
                "trigger mask %d (VLLM_H82_ROUTE_EARLY)",
                new.name,
                h82_route_early(),
            )
        if h158:
            logger.info(
                "H158 routing: module %s built with H158_ROUTING_PATCH: lean "
                "single-CTA routing kernel for %s-%s tokens (VLLM_H158_LEAN_ROUTE)",
                new.name,
                os.environ.get("VLLM_H158_LEAN_ROUTE_MIN", "1"),
                os.environ.get("VLLM_H158_LEAN_ROUTE_MAX", "32"),
            )
        if pdl_fc:
            _PDL_MODULE_READY = True
            logger.info(
                "[moe-pdl-fc] MoE PDL subset engaged: module %s -> %s, launcher "
                "source %s (all routing_runner.run sites enable_pdl=false); "
                "FC1/FC2 PDL on, routing PDL off",
                spec.name,
                new.name,
                lsrc,
            )
        return new

    # FlashInfer adapter (see module docstring).
    core.gen_trtllm_gen_fused_moe_sm100_module = gen
    if hasattr(core, "_get_trtllm_moe_sm100_module_impl"):
        core._get_trtllm_moe_sm100_module_impl.cache_clear()
    if pdl_fc:
        # Bypass FlashInfer's sm_107 clamp for the MoE runner only; returns
        # False until gen() has produced the patched module.
        core._device_support_moe_pdl = _moe_pdl_fc_supported
        logger.info(
            "[moe-pdl-fc] FlashInfer _device_support_moe_pdl replaced "
            "(FC1/FC2 PDL once the routing-never-PDL module is built)"
        )
    _installed = ENABLED
    _pdl_fc_installed = pdl_fc
    _h82_installed = h82
    _h158_installed = h158


_PREBUILT_CLS: dict[type, type] = {}


def _prebuilt_spec_cls(base: type) -> type:
    cls = _PREBUILT_CLS.get(base)
    if cls is None:

        @dataclasses.dataclass
        class PrebuiltSpec(base):  # type: ignore[valid-type, misc]
            prebuilt: Path | None = None

            @property
            def aot_path(self) -> Path:
                assert self.prebuilt is not None
                return self.prebuilt

        cls = _PREBUILT_CLS[base] = PrebuiltSpec
    return cls


# Unified diff against FlashInfer 0.6.18.post1
# csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_{common,custom}.cu
# (installed as flashinfer/data/csrc/fused_moe/trtllm_backend/...).
FLASHINFER_PATCH = r"""--- a/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu
+++ b/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu
@@ -14,6 +14,8 @@
  * limitations under the License.
  */
 #include <algorithm>
+#include <cstdio>
+#include <cstdlib>
 
 #include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
 #include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
@@ -38,12 +40,334 @@
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+// Single-CTA post-topK permutation for precomputed top-k results (no cluster, no DSMEM, smem
+// histogram atomics, parallel CTA-table fill). Input is either unpacked ids (mPtrTopKIds, weights
+// given) or the packed (bf16 score, int16 expert) words written by the score kernels
+// (mPtrTopKPacked; the weights are then written from the packed scores, as the stock kernels do).
+// Produces the same per-expert counts, CTA tables, padded sizes and the same (expert,
+// within-expert slot) assignment up to the within-expert order, which is nondeterministic in the
+// stock atomics-based kernels as well.
+namespace gs2 {
+struct Params {
+  void const* topKIn;  // int32 ids (PACKED = false) or uint32 packed (bf16 score | int16 idx << 16)
+  uint16_t* topKWeights;  // bf16 bits; written only for PACKED input
+  int32_t* expandedIdxToPermutedIdx;
+  int32_t* permutedIdxToExpandedIdx;
+  int32_t* permutedIdxToTokenIdx;
+  int32_t* ctaIdxXyToBatchIdx;
+  int32_t* ctaIdxXyToMnLimit;
+  int32_t* numNonExitingCtas;
+  int32_t* permutedIdxSize;
+  int32_t* numTokensPerExpert;
+  int32_t numTokens, numExperts, topK, tileN, paddingLog2;
+  int32_t localStart, localStrideLog2, numLocal;
+  uint32_t topKMagic;  // ceil(2^32 / topK): idx / topK == umulhi(idx, magic) for idx < 2^32 / topK
+  bool isPow2, usePdl, vec;  // vec: 16-byte aligned inputs/outputs (vector loads and stores)
+};
+static constexpr int NT = 1024;
+static constexpr int MaxE = 1024;
+__device__ __forceinline__ int32_t mulT(Params const& p, int32_t a) {
+  return p.isPow2 ? (a << p.paddingLog2) : a * p.tileN;
+}
+__device__ __forceinline__ int32_t divUpT(Params const& p, int32_t a) {
+  return p.isPow2 ? ((a + (1 << p.paddingLog2) - 1) >> p.paddingLog2) : (a + p.tileN - 1) / p.tileN;
+}
+template <bool PACKED>
+__device__ __forceinline__ int32_t loadOne(Params const& p, int idx) {
+  if constexpr (PACKED) {
+    uint32_t const w = static_cast<uint32_t const*>(p.topKIn)[idx];
+    if (p.topKWeights != nullptr) p.topKWeights[idx] = static_cast<uint16_t>(w & 0xffffu);
+    return static_cast<int32_t>(static_cast<int16_t>(w >> 16));
+  } else {
+    return static_cast<int32_t const*>(p.topKIn)[idx];
+  }
+}
+// EPT = entries per thread (multiple of 4); thread t owns entries [(i*NT + t)*4 + j], j<4 (int4 loads/stores).
+template <int EPT, bool AGG, bool PACKED>
+__global__ void __launch_bounds__(NT) gs2RoutingBlockKernel(Params p) {
+  static_assert(EPT % 4 == 0, "EPT multiple of 4");
+  using Scan = cub::BlockScan<int32_t, NT, cub::BLOCK_SCAN_WARP_SCANS>;
+  __shared__ typename Scan::TempStorage tmp;
+  __shared__ int32_t sCnt[MaxE];
+  __shared__ int32_t sCtaOff[MaxE + 1];
+  __shared__ int32_t sOff[MaxE];
+  int const tid = threadIdx.x;
+  int const lane = tid & 31;
+  unsigned const full = 0xffffffffu;
+  unsigned const ltMask = (1u << lane) - 1u;
+  int const n = p.numTokens * p.topK;
+  int const localExtent = p.numLocal << p.localStrideLog2;
+  bool const simpleLocal = p.localStart == 0 && p.localStrideLog2 == 0 && p.numLocal >= p.numExperts;
+  uint32_t const magic = p.topKMagic;
+  // Nothing is read or written in global memory before the wait. The top-K input is written by
+  // the PDL primary (score kernel), so it is read with coherent loads, not ld.global.nc.
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
+  if (p.usePdl) cudaGridDependencySynchronize();
+#endif
+  int32_t eIdx[EPT];
+  int32_t eOff[EPT];
+#pragma unroll
+  for (int g = 0; g < EPT / 4; g++) {
+    int const base = (g * NT + tid) * 4;
+    if (p.vec && base + 3 < n) {
+      uint4 const v = *reinterpret_cast<uint4 const*>(static_cast<uint32_t const*>(p.topKIn) + base);
+      if constexpr (PACKED) {
+        eIdx[4 * g] = static_cast<int16_t>(v.x >> 16);
+        eIdx[4 * g + 1] = static_cast<int16_t>(v.y >> 16);
+        eIdx[4 * g + 2] = static_cast<int16_t>(v.z >> 16);
+        eIdx[4 * g + 3] = static_cast<int16_t>(v.w >> 16);
+        if (p.topKWeights != nullptr) {
+          *reinterpret_cast<uint2*>(p.topKWeights + base) =
+              make_uint2((v.x & 0xffffu) | (v.y << 16), (v.z & 0xffffu) | (v.w << 16));
+        }
+      } else {
+        eIdx[4 * g] = static_cast<int32_t>(v.x);
+        eIdx[4 * g + 1] = static_cast<int32_t>(v.y);
+        eIdx[4 * g + 2] = static_cast<int32_t>(v.z);
+        eIdx[4 * g + 3] = static_cast<int32_t>(v.w);
+      }
+    } else {
+#pragma unroll
+      for (int j = 0; j < 4; j++) eIdx[4 * g + j] = base + j < n ? loadOne<PACKED>(p, base + j) : -1;
+    }
+  }
+  for (int e = tid; e < p.numExperts; e += NT) sCnt[e] = 0;
+  __syncthreads();
+#pragma unroll
+  for (int i = 0; i < EPT; i++) {
+    if ((i / 4) * NT * 4 >= n) break;  // block-uniform
+    int const e = eIdx[i];
+    bool loc;
+    if (simpleLocal) {
+      loc = e >= 0;
+    } else {
+      int const le = e - p.localStart;
+      loc = e >= 0 && le >= 0 && le < localExtent && (le & ((1 << p.localStrideLog2) - 1)) == 0;
+    }
+    if constexpr (AGG) {
+      int const key = loc ? e : -(lane + 1);
+      unsigned const m = __match_any_sync(full, key);
+      int const leader = __ffs(m) - 1;
+      int b0 = 0;
+      if (loc && lane == leader) b0 = atomicAdd(sCnt + e, __popc(m));
+      b0 = __shfl_sync(full, b0, leader);
+      eOff[i] = loc ? b0 + __popc(m & ltMask) : 0;
+    } else {
+      eOff[i] = loc ? atomicAdd(sCnt + e, 1) : 0;
+    }
+  }
+  __syncthreads();
+  int32_t cnt = 0, nCta = 0;
+  if (tid < p.numExperts) {
+    cnt = sCnt[tid];
+    nCta = divUpT(p, cnt);
+  }
+  int32_t ctaOff, total;
+  Scan(tmp).ExclusiveSum(nCta, ctaOff, total);
+  if (tid < p.numExperts) {
+    sCtaOff[tid] = ctaOff;
+    sOff[tid] = mulT(p, ctaOff);
+    if (p.numTokensPerExpert != nullptr) p.numTokensPerExpert[tid] = cnt;
+  }
+  if (tid == 0) {
+    sCtaOff[p.numExperts] = total;
+    p.permutedIdxSize[0] = mulT(p, total);
+    p.numNonExitingCtas[0] = total;
+  }
+  __syncthreads();
+  for (int c = tid; c < total; c += NT) {
+    int lo = 0, hi = p.numExperts;
+    while (lo < hi) {
+      int const mid = (lo + hi) >> 1;
+      if (sCtaOff[mid] <= c) lo = mid + 1; else hi = mid;
+    }
+    int const e = lo - 1;
+    p.ctaIdxXyToBatchIdx[c] = (e - p.localStart) >> p.localStrideLog2;
+    p.ctaIdxXyToMnLimit[c] = min(mulT(p, c + 1), mulT(p, sCtaOff[e]) + sCnt[e]);
+  }
+#pragma unroll
+  for (int g = 0; g < EPT / 4; g++) {
+    int const base = (g * NT + tid) * 4;
+    if (g * NT * 4 >= n) break;
+    int pi[4];
+#pragma unroll
+    for (int j = 0; j < 4; j++) {
+      int const e = eIdx[4 * g + j];
+      bool loc;
+      if (simpleLocal) {
+        loc = e >= 0;
+      } else {
+        int const le = e - p.localStart;
+        loc = e >= 0 && le >= 0 && le < localExtent && (le & ((1 << p.localStrideLog2) - 1)) == 0;
+      }
+      pi[j] = loc ? sOff[e] + eOff[4 * g + j] : -1;
+      int const idx = base + j;
+      if (idx < n && loc) {
+        if (p.permutedIdxToExpandedIdx != nullptr) p.permutedIdxToExpandedIdx[pi[j]] = idx;
+        if (p.permutedIdxToTokenIdx != nullptr)
+          p.permutedIdxToTokenIdx[pi[j]] = static_cast<int32_t>(__umulhi(static_cast<uint32_t>(idx), magic));
+      }
+    }
+    if (p.expandedIdxToPermutedIdx != nullptr) {
+      if (p.vec && base + 3 < n) {
+        *reinterpret_cast<int4*>(p.expandedIdxToPermutedIdx + base) = make_int4(pi[0], pi[1], pi[2], pi[3]);
+      } else {
+#pragma unroll
+        for (int j = 0; j < 4; j++)
+          if (base + j < n) p.expandedIdxToPermutedIdx[base + j] = pi[j];
+      }
+    }
+  }
+  // Trigger after every global write (CTA tables, sizes, permutation maps, weights).
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
+  if (p.usePdl) cudaTriggerProgrammaticLaunchCompletion();
+#endif
+}
+
+struct Cfg {
+  int mode = 1;          // GS2_ROUTE: 0 = stock, 1 = on
+  int maxN = 8 * NT;     // GS2_ROUTE_MAXN: max expanded entries (numTokens*topK); measured on
+                         // VR (E=256, top-8) slower than the stock cooperative kernel from
+                         // 1536 tokens (12288 entries) on
+  int minTok = 17;       // GS2_ROUTE_MINTOK
+  int packed = 1;        // GS2_ROUTE_PACKED: also take packed (bf16 score, idx) top-K input
+  int large = 1;         // GS2_ROUTE_LARGE: also replace the cooperative/multi-kernel permutation
+                         // after the large-batch score kernels (routingCustom::run, > 256 tokens)
+  bool log = false;
+  int agg = 0;           // GS2_ROUTE_AGG: warp-aggregated atomics
+};
+static Cfg const& cfg() {
+  static Cfg const c = [] {
+    Cfg c;
+    if (char const* v = std::getenv("GS2_ROUTE")) c.mode = std::atoi(v);
+    if (char const* v = std::getenv("GS2_ROUTE_MAXN")) c.maxN = std::atoi(v);
+    if (char const* v = std::getenv("GS2_ROUTE_MINTOK")) c.minTok = std::atoi(v);
+    if (char const* v = std::getenv("GS2_ROUTE_PACKED")) c.packed = std::atoi(v);
+    if (char const* v = std::getenv("GS2_ROUTE_LARGE")) c.large = std::atoi(v);
+    if (char const* v = std::getenv("GS2_ROUTE_LOG")) c.log = std::atoi(v) != 0;
+    if (char const* v = std::getenv("GS2_ROUTE_AGG")) c.agg = std::atoi(v);
+    if (c.maxN > 32 * NT) c.maxN = 32 * NT;
+    if (c.log)
+      fprintf(stderr, "[gs2-route] mode=%d maxN=%d minTok=%d packed=%d large=%d agg=%d\n", c.mode,
+              c.maxN, c.minTok, c.packed, c.large, c.agg);
+    return c;
+  }();
+  return c;
+}
+
+static inline bool aligned16(void const* ptr) {
+  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
+}
+
+template <typename DataType>
+bool tryLaunch(DataType const& d, void* stream, bool largePath = false) {
+  Cfg const& c = cfg();
+  char const* mv = std::getenv("GS2_ROUTE");  // re-read per call (host-side only; cheap)
+  if ((mv != nullptr ? std::atoi(mv) : c.mode) == 0) return false;
+  if (largePath && !c.large) return false;
+  if (d.mPtrScores != nullptr) return false;
+  bool const packed = d.mPtrTopKIds == nullptr;
+  if (packed) {
+    // Packed words are {OutputT score; int16 idx}: 4 bytes only for bf16 weights.
+    if (!c.packed || d.mPtrTopKPacked == nullptr || d.mDtypeOutput != tg::Dtype::Bfloat16) return false;
+  }
+  if (d.mNumExperts > MaxE || d.mNumExperts > NT) return false;
+  if (d.mPtrPermutedIdxSize == nullptr || d.mPtrCtaIdxXyToBatchIdx == nullptr ||
+      d.mPtrCtaIdxXyToMnLimit == nullptr || d.mPtrNumNonExitingCtas == nullptr)
+    return false;
+  if (d.mNumTokens < c.minTok) return false;
+  int const n = d.mNumTokens * d.mTopK;
+  if (n > c.maxN) return false;
+  Params p;
+  p.topKIn = packed ? static_cast<void const*>(d.mPtrTopKPacked) : static_cast<void const*>(d.mPtrTopKIds);
+  p.topKWeights = packed ? static_cast<uint16_t*>(d.mPtrTopKWeights) : nullptr;
+  p.expandedIdxToPermutedIdx = d.mPtrExpandedIdxToPermutedIdx;
+  p.permutedIdxToExpandedIdx = d.mPtrPermutedIdxToExpandedIdx;
+  p.permutedIdxToTokenIdx = d.mPtrPermutedIdxToTokenIdx;
+  p.ctaIdxXyToBatchIdx = d.mPtrCtaIdxXyToBatchIdx;
+  p.ctaIdxXyToMnLimit = d.mPtrCtaIdxXyToMnLimit;
+  p.numNonExitingCtas = d.mPtrNumNonExitingCtas;
+  p.permutedIdxSize = d.mPtrPermutedIdxSize;
+  p.numTokensPerExpert = d.mPtrNumTokensPerExpert;
+  p.numTokens = d.mNumTokens;
+  p.numExperts = d.mNumExperts;
+  p.topK = d.mTopK;
+  p.tileN = d.mTileTokensDim;
+  p.paddingLog2 = d.mPaddingLog2;
+  p.isPow2 = d.mPaddingLog2 > 0;
+  p.localStart = d.mLocalExpertsStartIdx;
+  p.localStrideLog2 = d.mLocalExpertsStrideLog2;
+  p.numLocal = d.mNumLocalExperts;
+  p.usePdl = d.mUsePdl;
+  // Vector loads/stores need 16-byte aligned inputs and permutation map (8 for the bf16 weights,
+  // implied by 16 here); otherwise the kernel takes the scalar path, so it always launches once
+  // the conditions above hold.
+  p.vec = aligned16(p.topKIn) && aligned16(d.mPtrExpandedIdxToPermutedIdx) &&
+          (p.topKWeights == nullptr || aligned16(p.topKWeights));
+  p.topKMagic = static_cast<uint32_t>((uint64_t{1} << 32) / d.mTopK + 1);
+  cudaLaunchConfig_t config{};
+  config.gridDim = 1;
+  config.blockDim = NT;
+  config.dynamicSmemBytes = 0;
+  config.stream = reinterpret_cast<cudaStream_t>(stream);
+  cudaLaunchAttribute attrs[1] = {};
+  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
+  attrs[0].val.programmaticStreamSerializationAllowed = int(d.mUsePdl);
+  config.attrs = attrs;
+  config.numAttrs = 1;
+  int const ept = (n + 4 * NT - 1) / (4 * NT) * 4;
+#define GS2_K(E_, P_)                                                                          \
+  CHECK_CUDA_ERROR(c.agg ? cudaLaunchKernelEx(&config, gs2RoutingBlockKernel<E_, true, P_>, p) \
+                         : cudaLaunchKernelEx(&config, gs2RoutingBlockKernel<E_, false, P_>, p))
+#define GS2_L(E_)        \
+  if (packed) {          \
+    GS2_K(E_, true);     \
+  } else {               \
+    GS2_K(E_, false);    \
+  }
+  if (ept <= 4) {
+    GS2_L(4)
+  } else if (ept <= 8) {
+    GS2_L(8)
+  } else if (ept <= 16) {
+    GS2_L(16)
+  } else {
+    GS2_L(32)
+  }
+#undef GS2_L
+#undef GS2_K
+  if (c.log) {
+    static int logged = 0;
+    if (logged < 8) {
+      logged++;
+      fprintf(stderr, "[gs2-route] launch tokens=%d topK=%d packed=%d large=%d vec=%d pdl=%d\n",
+              d.mNumTokens, d.mTopK, int(packed), int(largePath), int(p.vec), int(d.mUsePdl));
+    }
+  }
+  return true;
+}
+}  // namespace gs2
+
+namespace routingCustom {
+// Single-CTA permutation after the large-batch score kernels (called from routingCustom::run,
+// which owns the cooperative / multi-kernel permutation for > 256 tokens).
+bool launchGs2PostTopK(Data const& data, void* stream) {
+  return gs2::tryLaunch(data, stream, /*largePath=*/true);
+}
+}  // namespace routingCustom
+
 // Implementation of shared post-topK pipeline for all routing methods.
 // When topK is already computed (mPtrTopKIds or mPtrTopKPacked), we don't need
 // routing-method-specific logic, so all methods can use the same workflow.
 // This function handles all path selection: single-block, single-cluster, coop, multi-kernel.
 template <typename DataType>
 void runPostTopKPipeline(DataType const& data, void* stream) {
+  // Single-CTA permutation for precomputed ids
+  if (gs2::tryLaunch(data, stream)) {
+    return;
+  }
   // Convert to routingCustom::Data for launching (kernels are shared)
   routingCustom::Data customData;
   // Copy base fields
--- a/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
+++ b/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
@@ -1475,6 +1475,10 @@
 //
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
+// Single-CTA permutation for packed top-K (trtllm_fused_moe_routing_common.cu); returns false
+// when it does not apply (then the stock permutation runs).
+bool launchGs2PostTopK(Data const& data, void* stream);
+
 void run(Data const& data, void* stream) {
   TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                  data.mPtrTopKIds != nullptr)
@@ -1742,6 +1746,15 @@
         " for the active routing policy (large-batch path; see preceding warning). "
         "Add a matching Tier<E, K> to (Histogram)PolicyTraits in RoutingCustomPolicy.cuh.");
 
+    {
+      // Both score kernels wrote mPtrTopKPacked; permute it in one CTA when that applies.
+      Data postTopK = mutableData;
+      postTopK.mPtrScores = nullptr;
+      if (launchGs2PostTopK(postTopK, stream)) {
+        return;
+      }
+    }
+
     bool const canUseCoop =
         (smMajor >= 9) && (data.mNumExperts <= 1024) && (data.mPtrPermutedIdxSize != nullptr);
     bool useCoop = false;
"""


# Unified diff against FlashInfer 0.6.18.post1 csrc/trtllm_fused_moe_kernel_launcher.cu
# (installed as flashinfer/data/csrc/...), applied when VLLM_MOE_PDL_FC=1.
PDL_FC_PATCH = r"""Subject: [PATCH] trtllm fused MoE launcher: routing kernels never PDL-launched

Unified diff against FlashInfer 0.6.18.post1 (tag 8bc3b578)
csrc/trtllm_fused_moe_kernel_launcher.cu
(sha256 7b7adacc9117eb64869bb68b96818af9588a85651afd83cb4a27abaae54152bf;
installed as flashinfer/data/csrc/...).

FlashInfer passes one enable_pdl to Routing::Runner::run and to
MoE::Runner::run (FC1 / FC2 / activation / finalize). On sm_107 its
Python layer clamps enable_pdl off for the whole chain (core.py
_device_support_moe_pdl, upstream #4806: intermittent unspecified launch
failures / hangs reported at the routing kernel launch with PDL on).

This patch passes enable_pdl = false to all four routing_runner.run(...)
call sites, so the routing / permutation kernels (routingCustom block,
dyn-block, cluster, coop, init-expert-counts, and vLLM's 0001 gs2 kernel)
are launched without the programmatic-stream-serialization attribute and,
since every routing wait / trigger is guarded by mUsePdl, never wait on or
trigger a programmatic edge. The MoE runner keeps the caller's enable_pdl,
so FC1 / FC2 can be PDL-launched (FC1 then launches only when the routing
grid has exited; FC2 overlaps FC1's tail). Numerics are unchanged (launch
attributes only).

Applied and built by vllm/model_executor/layers/fused_moe/
flashinfer_exact_routing.py when VLLM_MOE_PDL_FC=1; that module also
bypasses FlashInfer's sm_107 clamp only while this patched module is in use.

--- a/csrc/trtllm_fused_moe_kernel_launcher.cu	2026-10-02 21:59:16
+++ b/csrc/trtllm_fused_moe_kernel_launcher.cu	2026-10-02 21:59:16
@@ -1516,7 +1516,7 @@
         static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
         use_routing_scales_on_input, use_deep_seek_fp8,
         static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
-        norm_topk_prob, replay_ptr, enable_pdl);
+        norm_topk_prob, replay_ptr, /*enable_pdl=*/false);  // [vllm moe-pdl-fc] routing never PDL
 
     check_moe();
     prepare_moe(moe_tactic);
@@ -2612,7 +2612,7 @@
         static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
         use_routing_scales_on_input, use_deep_seek_fp8,
         static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
-        norm_topk_prob, replay_ptr, enable_pdl);
+        norm_topk_prob, replay_ptr, /*enable_pdl=*/false);  // [vllm moe-pdl-fc] routing never PDL
 
     check_moe();
     prepare_moe(moe_tactic);
@@ -3305,7 +3305,7 @@
                        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt,
                        mRoutingBiasDtype, use_routing_scales_on_input, use_deep_seek_fp8,
                        static_cast<RoutingMethodType>(routing_method_type), routing_stream,
-                       mRoutingLogitsDtype, norm_topk_prob, replay_ptr, enable_pdl);
+                       mRoutingLogitsDtype, norm_topk_prob, replay_ptr, /*enable_pdl=*/false);  // [vllm moe-pdl-fc] routing never PDL
 
     check_moe();
     prepare_moe(moe_tactic);
@@ -4403,7 +4403,7 @@
       static_cast<int*>(buffers.num_non_exiting_ctas.data_ptr()), dtype_elt, routing_bias_dtype,
       use_routing_scales_on_input, use_deep_seek_fp8,
       static_cast<RoutingMethodType>(routing_method_type), stream, routing_logits_dtype,
-      norm_topk_prob, static_cast<int16_t*>(buffers.routing_replay_ids.data_ptr()), enable_pdl);
+      norm_topk_prob, static_cast<int16_t*>(buffers.routing_replay_ids.data_ptr()), /*enable_pdl=*/false);  // [vllm moe-pdl-fc] routing never PDL
 }
 
 /// Allocate graph-stable routing metadata storage for each requested tile without launching work.
"""

# Unified diff against the GS2_ROUTE-patched (FLASHINFER_PATCH) FlashInfer 0.6.18.post1
# csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_{common,custom}.cu, applied on top of it
# when VLLM_H82_ROUTE_EARLY is non-zero. The early trigger
# follows FlashInfer PR #6449 (754cbe9e, open; there for the static block kernel only).
H82_ROUTING_PATCH = r"""--- a/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu
+++ b/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu
@@ -68,6 +68,8 @@
 };
 static constexpr int NT = 1024;
 static constexpr int MaxE = 1024;
+// [H82] Early PDL trigger (VLLM_H82_ROUTE_EARLY; see routingCustom::h82SyncEarlyTrigger).
+__device__ int h82EarlyTrigger = 0;
 __device__ __forceinline__ int32_t mulT(Params const& p, int32_t a) {
   return p.isPow2 ? (a << p.paddingLog2) : a * p.tileN;
 }
@@ -103,8 +105,12 @@
   uint32_t const magic = p.topKMagic;
   // Nothing is read or written in global memory before the wait. The top-K input is written by
   // the PDL primary (score kernel), so it is read with coherent loads, not ld.global.nc.
+  int const h82Early = h82EarlyTrigger;  // [H82] read before the wait: set before launch
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
-  if (p.usePdl) cudaGridDependencySynchronize();
+  if (p.usePdl) {
+    cudaGridDependencySynchronize();
+    if (h82Early & 1) cudaTriggerProgrammaticLaunchCompletion();  // [H82] early trigger
+  }
 #endif
   int32_t eIdx[EPT];
   int32_t eOff[EPT];
@@ -278,6 +284,20 @@
       d.mPtrCtaIdxXyToMnLimit == nullptr || d.mPtrNumNonExitingCtas == nullptr)
     return false;
   if (d.mNumTokens < c.minTok) return false;
+  {
+    // [H82] VLLM_H82_ROUTE_EARLY -> h82EarlyTrigger on this device (not while capturing).
+    static int state[64] = {};
+    char const* ev = std::getenv("VLLM_H82_ROUTE_EARLY");
+    int const want = ev != nullptr ? (std::atoi(ev) & 3) : 0;
+    int dev = 0;
+    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
+    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && state[dev] != want + 1 &&
+        cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &status) == cudaSuccess &&
+        status == cudaStreamCaptureStatusNone) {
+      CHECK_CUDA_ERROR(cudaMemcpyToSymbol(h82EarlyTrigger, &want, sizeof(int)));
+      state[dev] = want + 1;
+    }
+  }
   int const n = d.mNumTokens * d.mTopK;
   if (n > c.maxN) return false;
   Params p;
--- a/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
+++ b/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
@@ -26,6 +26,7 @@
 //   6. routingIndicesHistogramKernel  — histogram from packed TopK (defined in RoutingKernel.cuh)
 //   7. routingIndicesOffsetsKernel    — prefix-scan + permutation (defined in RoutingKernel.cuh)
 
+#include <cstdio>
 #include <cstdlib>
 #include <string>
 #include <type_traits>
@@ -36,6 +37,12 @@
 
 namespace moe::dev::routing {
 namespace routingCustom {
+
+// [H82] Early PDL trigger of the small-batch routing kernels (VLLM_H82_ROUTE_EARLY, see
+// h82SyncEarlyTrigger): right after their own grid-dependency wait, as FlashInfer PR #6449 does
+// for the static block kernel. Their dependents (FC1, the single-CTA permutation) wait on this
+// grid before reading what it writes.
+__device__ int h82EarlyTrigger = 0;
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 //
@@ -83,10 +90,14 @@
   }
   __syncthreads();
 
+  int const h82Early = h82EarlyTrigger;  // [H82] read before the wait: set before launch
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
   // then wait on primary grid
   if (params.mUsePdl) {
     cudaGridDependencySynchronize();
+    if (h82Early & 1) {
+      cudaTriggerProgrammaticLaunchCompletion();  // [H82] early trigger
+    }
   }
 #endif
 
@@ -413,9 +424,13 @@
   }
   __syncthreads();
 
+  int const h82Early = h82EarlyTrigger;  // [H82] read before the wait: set before launch
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
   if (params.mUsePdl) {
     cudaGridDependencySynchronize();
+    if (h82Early & 1) {
+      cudaTriggerProgrammaticLaunchCompletion();  // [H82] early trigger
+    }
   }
 #endif
 
@@ -1172,9 +1187,13 @@
     int32_t const warpIdx = threadIdx.x / WarpSize;
     int32_t const tokenIdx = blockIdx.x;
 
+    int const h82Early = h82EarlyTrigger;  // [H82] read before the wait: set before launch
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
     if (params.mUsePdl) {
       cudaGridDependencySynchronize();
+      if (h82Early & 2) {
+        cudaTriggerProgrammaticLaunchCompletion();  // [H82] early trigger
+      }
     }
 #endif
 
@@ -1479,7 +1498,32 @@
 // when it does not apply (then the stock permutation runs).
 bool launchGs2PostTopK(Data const& data, void* stream);
 
+// [H82] Copies VLLM_H82_ROUTE_EARLY (a mask, default 0: 1 = the kernels whose dependent is FC1 --
+// static and dynamic block kernels and the single-CTA permutation; 2 = the block-per-token score
+// kernel, whose dependent is the permutation) to h82EarlyTrigger on the current device when it
+// changed; never while `stream` is capturing (the device keeps its last value, which the captured
+// kernels read at replay).
+static void h82SyncEarlyTrigger(void* stream) {
+  static int state[64] = {};  // 1 + last value written, per device; 0 = never written
+  char const* v = std::getenv("VLLM_H82_ROUTE_EARLY");
+  int const want = v != nullptr ? (std::atoi(v) & 3) : 0;
+  int dev = 0;
+  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || state[dev] == want + 1) {
+    return;
+  }
+  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
+  if (cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status) != cudaSuccess ||
+      status != cudaStreamCaptureStatusNone) {
+    return;
+  }
+  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(h82EarlyTrigger, &want, sizeof(int)));
+  state[dev] = want + 1;
+  fprintf(stderr, "[h82-route] early PDL trigger mask %d (1: FC1 producers, 2: score kernel; device %d)\n",
+          want, dev);
+}
+
 void run(Data const& data, void* stream) {
+  h82SyncEarlyTrigger(stream);
   TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                  data.mPtrTopKIds != nullptr)
       << "Routing kernel requires at least one input parameter";
"""

# Unified diff against the H82-patched (FLASHINFER_PATCH + H82_ROUTING_PATCH) FlashInfer
# 0.6.18.post1 csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu, applied on
# top of it when VLLM_H158_LEAN_ROUTE is non-zero: the lean single-CTA
# routing kernel (routingIndicesH158LeanKernel) and its selection in routingCustom::run.
H158_ROUTING_PATCH = r"""--- a/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
+++ b/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
@@ -662,6 +662,247 @@
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 //
+// [H158] Lean single-CTA routing kernel from raw logits (VLLM_H158_LEAN_ROUTE).
+//
+// One launch replaces the dyn-block kernel (<= 16 tokens) and the block-per-token score kernel +
+// single-CTA permutation pair (17..32 tokens):
+//   Phase 1 is the dyn-block kernel's: one warp per token through the same
+//     ExpertSelectPolicy::apply (same top-K reduction and postprocess code and arguments), so the
+//     selected experts, their order and their weights are bit-identical to the dyn-block kernel's
+//     and to the block-per-token score kernel's (for the Renormalize policy both run
+//     topk::reduceTopK over the same per-lane layout and then calcSoftmax over the K survivors).
+//   Phases 2-4 replace the dyn-block kernel's per-expert serial loops over the tokens: each expert
+//     keeps a 32-bit token mask (atomicOr), its count is popc(mask) and an entry's slot within the
+//     expert is popc(mask & lower tokens). That is the dyn-block kernel's token order, so the
+//     permutation maps and CTA tables equal the dyn-block kernel's (and the single-CTA
+//     permutation's up to its within-expert order, which is atomics-ordered there).
+// It writes the union of what both stock paths write: weights, packed top-K, replay ids,
+// expanded->permuted / permuted->expanded / permuted->token maps, CTA tables, permuted size,
+// non-exiting CTA count and per-expert counts. Nothing in global memory is touched before the
+// grid-dependency wait; dependents are triggered after the last write (as the dyn-block kernel),
+// or right after the wait with the H82 early-trigger mask bit 1.
+//
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+static constexpr int H158LeanMaxNumTokens = 32;  // one warp per token, token masks are 32 bits
+
+template <typename KernelParams>
+__global__ void __launch_bounds__(1024) routingIndicesH158LeanKernel(KernelParams params) {
+  using OutputT = typename KernelParams::OutputT;
+  using InputT = typename KernelParams::InputT;
+  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
+  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
+  static constexpr int MaxNumTopExperts = KernelParams::MaxNumTopExperts;
+  static constexpr int NumThreadsExperts = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
+  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsExperts;
+  static constexpr int NumExpertWarps = NumThreadsExperts / WarpSize;
+  static constexpr int VecSize = MaxNumExperts / WarpSize;
+
+  static_assert(MaxNumExperts % WarpSize == 0);
+  static_assert(MaxNumExperts % NumThreadsExperts == 0);
+
+  __shared__ uint32_t smemTokenMask[MaxNumExperts];
+  __shared__ int32_t smemExpertOffset[MaxNumExperts];
+  __shared__ int32_t smemTopKExpert[H158LeanMaxNumTokens * MaxNumTopExperts];
+  __shared__ int32_t smemWarpTotals[NumExpertWarps];
+  __shared__ int32_t smemWarpTotals2[NumExpertWarps];
+
+  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
+  int32_t const laneIdx = cutlass::arch::LaneId();
+  int32_t const numWarps = blockDim.x / WarpSize;
+
+  auto block = cg::this_thread_block();
+  auto warp = cg::tiled_partition<WarpSize>(block);
+
+  for (int e = threadIdx.x; e < MaxNumExperts; e += blockDim.x) {
+    smemTokenMask[e] = 0u;
+  }
+  __syncthreads();
+
+  int const h82Early = h82EarlyTrigger;  // [H82] read before the wait: set before launch
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
+  if (params.mUsePdl) {
+    cudaGridDependencySynchronize();
+    if (h82Early & 1) {
+      cudaTriggerProgrammaticLaunchCompletion();  // [H82] early trigger
+    }
+  }
+#endif
+
+  int32_t const numTokens = params.mNumTokens;
+  int32_t const topK = params.mTopK;
+
+  // Phase 1: top-K, one warp per token (the host guarantees numTokens <= numWarps).
+  for (int tokenIdx = warpIdx; tokenIdx < numTokens; tokenIdx += numWarps) {
+    BaseType warpTopKScore[MaxNumTopExperts];
+    int32_t warpTopKExpertIdx[MaxNumTopExperts];
+
+    auto scoreOff = tokenIdx * params.mNumExperts;
+    KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize, MaxNumTopExperts>(
+        warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, topK,
+        params.mPtrScores + scoreOff, params);
+
+    if (laneIdx < topK) {
+      int32_t const expertIdx = warpTopKExpertIdx[laneIdx];
+      int32_t const expandedIdx = tokenIdx * topK + laneIdx;
+      smemTopKExpert[expandedIdx] = expertIdx;
+      if (expertIdx >= 0 && expertIdx < MaxNumExperts) {
+        atomicOr(&smemTokenMask[expertIdx], 1u << tokenIdx);
+      }
+      OutputT const weight = OutputT{warpTopKScore[laneIdx]};
+      if (params.mPtrTopKWeights != nullptr) {
+        params.mPtrTopKWeights[expandedIdx] = weight;
+      }
+      if (params.mPtrTopKPacked != nullptr) {
+        params.mPtrTopKPacked[expandedIdx] =
+            PackedScoreIdx<OutputT>{weight, static_cast<int16_t>(expertIdx)};
+      }
+      if (params.mPtrRoutingReplayOut != nullptr) {
+        params.mPtrRoutingReplayOut[expandedIdx] = static_cast<int16_t>(expertIdx);
+      }
+    }
+  }
+  __syncthreads();
+
+  // Phase 2: per-expert counts and CTA counts, then the dyn-block kernel's dual exclusive scan.
+  int32_t accExpertCount[ExpertsPerThread];
+  int32_t numCtaPerExpert[ExpertsPerThread];
+  int32_t tmpCountPerExpert[ExpertsPerThread];
+  bool isLocalExpert[ExpertsPerThread];
+#pragma unroll
+  for (int e = 0; e < ExpertsPerThread; e++) {
+    int const expert = threadIdx.x * ExpertsPerThread + e;
+    bool isLocal = false;
+    if (threadIdx.x < NumThreadsExperts) {
+      auto localExpIdx = expert - params.mLocalExpertsStartIdx;
+      isLocal = localExpIdx >= 0 &&
+                localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
+                (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
+    }
+    isLocalExpert[e] = isLocal;
+    accExpertCount[e] = isLocal ? __popc(smemTokenMask[expert]) : 0;
+    if (params.mIsPow2) {
+      numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
+      tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
+    } else {
+      numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
+      tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
+    }
+  }
+
+  int32_t ctaOffsetPerExpert[ExpertsPerThread];
+  int32_t expertScanCountsPerExpert[ExpertsPerThread];
+  int32_t numNonExitingCtas;
+  {
+    int32_t localPrefix1[ExpertsPerThread], localPrefix2[ExpertsPerThread];
+    int32_t threadTotal1 = 0, threadTotal2 = 0;
+#pragma unroll
+    for (int e = 0; e < ExpertsPerThread; e++) {
+      localPrefix1[e] = threadTotal1;
+      localPrefix2[e] = threadTotal2;
+      threadTotal1 += numCtaPerExpert[e];
+      threadTotal2 += tmpCountPerExpert[e];
+    }
+    int32_t threadPrefix1, threadPrefix2;
+    warpExclusiveScan<NumExpertWarps>(threadTotal1, threadTotal2, laneIdx, warpIdx, smemWarpTotals,
+                                      smemWarpTotals2, threadPrefix1, threadPrefix2,
+                                      numNonExitingCtas);
+#pragma unroll
+    for (int e = 0; e < ExpertsPerThread; e++) {
+      ctaOffsetPerExpert[e] = threadPrefix1 + localPrefix1[e];
+      expertScanCountsPerExpert[e] = threadPrefix2 + localPrefix2[e];
+    }
+  }
+
+  // Phase 3: expert offsets (smem), CTA configs, per-expert counts, sizes.
+  if (threadIdx.x < NumThreadsExperts) {
+#pragma unroll
+    for (int e = 0; e < ExpertsPerThread; e++) {
+      int const expert = threadIdx.x * ExpertsPerThread + e;
+      smemExpertOffset[expert] = expertScanCountsPerExpert[e];
+      if (params.mPtrNumTokensPerExpert != nullptr && expert < params.mNumExperts) {
+        params.mPtrNumTokensPerExpert[expert] = accExpertCount[e];
+      }
+      if (isLocalExpert[e]) {
+        int32_t const mappedLocalIdx =
+            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
+        for (int cta = 0; cta < numCtaPerExpert[e]; ++cta) {
+          params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
+          int32_t mnLimit1, mnLimit2;
+          if (params.mIsPow2) {
+            mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mPaddingLog2);
+            mnLimit2 =
+                mulLog2<int32_t>(ctaOffsetPerExpert[e], params.mPaddingLog2) + accExpertCount[e];
+          } else {
+            mnLimit1 = mulTileN<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mTileTokensDim);
+            mnLimit2 =
+                mulTileN<int32_t>(ctaOffsetPerExpert[e], params.mTileTokensDim) + accExpertCount[e];
+          }
+          params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
+        }
+      }
+    }
+  }
+  if (threadIdx.x == 0) {
+    int32_t permutedIdxSize;
+    if (params.mIsPow2) {
+      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
+    } else {
+      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
+    }
+    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
+    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
+  }
+  __syncthreads();
+
+  // Phase 4: permutation, one thread per (token, k) entry.
+  int32_t const numExpanded = numTokens * topK;
+  for (int expandedIdx = threadIdx.x; expandedIdx < numExpanded; expandedIdx += blockDim.x) {
+    int32_t const tokenIdx = expandedIdx / topK;
+    int32_t const expertIdx = smemTopKExpert[expandedIdx];
+    bool isLocal = false;
+    if (expertIdx >= 0 && expertIdx < MaxNumExperts) {
+      auto localExpIdx = expertIdx - params.mLocalExpertsStartIdx;
+      isLocal = localExpIdx >= 0 &&
+                localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
+                (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
+    }
+    int32_t permutedIdx = int32_t{-1};
+    if (isLocal) {
+      uint32_t const lower = (1u << tokenIdx) - 1u;
+      permutedIdx = smemExpertOffset[expertIdx] + __popc(smemTokenMask[expertIdx] & lower);
+    }
+    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
+      params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
+    }
+    if (isLocal) {
+      if (params.mPtrPermutedIdxToExpandedIdx != nullptr) {
+        params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
+      }
+      if (params.mPtrPermutedIdxToTokenIdx != nullptr) {
+        params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
+      }
+    }
+  }
+
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
+  if (params.mUsePdl) {
+    cudaTriggerProgrammaticLaunchCompletion();
+  }
+#endif
+}
+
+// [H158] Launches the lean kernel; returns whether a compiled tier covered the call.
+bool launchH158LeanKernel(Data const& data, void* stream) {
+  int32_t const maxExperts = queryDispatchedMaxExperts(data);
+  int const threads =
+      std::min(std::max(data.mNumTokens * static_cast<int>(WarpSize), maxExperts), 1024);
+  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesH158LeanKernel, 1, threads, 0, stream);
+  return queryPolicyHasCompiledTier(data);
+}
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+//
 // 2. Cluster kernel — single-cluster fused kernel for ≤256 tokens (SM90+).
 //    Uses distributed shared memory across 8 blocks in a cluster.
 //
@@ -1522,6 +1763,30 @@
           want, dev);
 }
 
+
+// [H158] Whether this logits-routed call runs the lean single-CTA kernel: VLLM_H158_LEAN_ROUTE
+// non-zero (re-read per call, host side only; a captured graph keeps the kernel it captured),
+// VLLM_H158_LEAN_ROUTE_MIN <= tokens <= VLLM_H158_LEAN_ROUTE_MAX (defaults: the kernel's 1..32),
+// at most DynBlockKernelMaxNumExperts experts in the dispatched tier (as the dyn-block kernel),
+// no fused shared experts.
+static bool h158LeanWanted(Data const& data, int32_t dispatchedMaxExperts) {
+  char const* v = std::getenv("VLLM_H158_LEAN_ROUTE");
+  if (v == nullptr || std::atoi(v) == 0) {
+    return false;
+  }
+  static int const lo = [] {
+    char const* s = std::getenv("VLLM_H158_LEAN_ROUTE_MIN");
+    return s != nullptr ? std::max(1, std::atoi(s)) : 1;
+  }();
+  static int const hi = [] {
+    char const* s = std::getenv("VLLM_H158_LEAN_ROUTE_MAX");
+    return std::min(s != nullptr ? std::atoi(s) : H158LeanMaxNumTokens, H158LeanMaxNumTokens);
+  }();
+  return data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr && data.mNumTokens >= lo &&
+         data.mNumTokens <= hi && dispatchedMaxExperts <= DynBlockKernelMaxNumExperts &&
+         data.mNumFusedSharedExperts == 0;
+}
+
 void run(Data const& data, void* stream) {
   h82SyncEarlyTrigger(stream);
   TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
@@ -1569,6 +1834,22 @@
   // the smem budget.
   int32_t const dispatchedMaxExperts = queryDispatchedMaxExperts(data);
 
+  // [H158] Lean single-CTA routing (see routingIndicesH158LeanKernel).
+  if (h158LeanWanted(data, dispatchedMaxExperts)) {
+    bool const launched = launchH158LeanKernel(data, stream);
+    FLASHINFER_CHECK(launched, "routingCustom::run: no compiled tier covers numExperts=",
+                     data.mNumExperts, " topK=", data.mTopK, " (H158 lean routing kernel).");
+    static bool logged = false;
+    if (!logged) {
+      logged = true;
+      fprintf(stderr,
+              "[h158-route] lean single-CTA routing kernel on (VLLM_H158_LEAN_ROUTE): first call "
+              "tokens=%d experts=%d topK=%d pdl=%d\n",
+              data.mNumTokens, data.mNumExperts, data.mTopK, int(data.mUsePdl));
+    }
+    return;
+  }
+
   // Does the currently-active policy pair implement the block-per-token interface?
   bool const policySupportsBlockPerToken = queryPolicySupportsBlockPerToken(data);
 
"""

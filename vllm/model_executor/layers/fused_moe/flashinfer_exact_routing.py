# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-CTA post-top-K routing permutation for FlashInfer's trtllm fused MoE.

For precomputed top-K ids FlashInfer's trtllm fused MoE only runs the post-top-K
permutation (``runPostTopKPipeline`` in
``csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu``). For
moderate token counts that is a cluster kernel (8 CTAs, cluster 8) whose
cluster barriers and DSMEM reads put a fixed latency floor on every MoE layer.
``vllm/third_party/flashinfer_patches/0001-exact-single-cta-moe-routing.patch``
adds a single-CTA, cluster-free permutation for that case; all routing
metadata except the within-expert row order (already nondeterministic in the
stock atomics-based kernels) is identical, so MoE outputs are unchanged.

When enabled, the patch is applied to the installed FlashInfer source of that
file and FlashInfer's trtllm fused-MoE JIT module is built from the patched
source under a renamed module name (``<stock name>_<GS2_ROUTE_TAG>``), so
neither the stock JIT build directory nor a prebuilt AOT module of the stock
name is reused.

Environment:

* ``GS2_ROUTE`` (default ``0``): any other value enables the patched module.
  The patched C++ also reads it on every call (``GS2_ROUTE=0`` at runtime
  selects the stock kernels inside the patched module).
* ``GS2_ROUTE_TAG`` (default ``exact_routing``): suffix of the JIT module name.
* ``GS2_ROUTE_MAXN``, ``GS2_ROUTE_MINTOK``, ``GS2_ROUTE_LOG``,
  ``GS2_ROUTE_AGG``: read by the patched C++ (see the patch description).
* ``VLLM_FLASHINFER_MOE_ROUTING_CSRC``: path of an already patched
  ``trtllm_fused_moe_routing_common.cu`` to build instead of applying the patch.
* ``VLLM_MOE_PDL_FC`` (default ``0``): ``1`` enables the MoE-PDL subset. The
  module is additionally built with
  ``vllm/third_party/flashinfer_patches/0002-moe-routing-never-pdl.patch``
  applied to ``csrc/trtllm_fused_moe_kernel_launcher.cu`` (module name suffix
  ``_pdlfc``): every ``routing_runner.run`` call passes ``enable_pdl=false``,
  so routing / permutation kernels are never PDL-launched and never wait on or
  trigger a programmatic edge, while FC1 / FC2 keep the caller's
  ``enable_pdl``. FlashInfer's sm_107 clamp (``_device_support_moe_pdl``,
  upstream #4806) is bypassed only after that patched module has been
  generated; until then (or if the patch does not apply) MoE PDL stays off.
  Launch attributes only: numerics are unchanged. Works with or without
  ``GS2_ROUTE``.

FlashInfer adapter: FlashInfer builds the module through the module-level
function ``flashinfer.fused_moe.core.gen_trtllm_gen_fused_moe_sm100_module``
(cached by ``_get_trtllm_moe_sm100_module_impl``) and offers no way to pass
different sources, so :func:`maybe_install` replaces that function by one that
returns the renamed spec with the patched source and clears the cache. It must
run before the first trtllm fused-MoE call; vLLM calls it when MoE layers are
constructed. The proper home of the change is the FlashInfer source itself.
"""

import dataclasses
import hashlib
import os
import re
import tempfile
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("GS2_ROUTE", "0") != "0"
TAG = os.environ.get("GS2_ROUTE_TAG", "exact_routing")
_CSRC_OVERRIDE = os.environ.get("VLLM_FLASHINFER_MOE_ROUTING_CSRC")

ROUTING_SOURCE_NAME = "trtllm_fused_moe_routing_common.cu"
PATCH_FILE = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "flashinfer_patches"
    / "0001-exact-single-cta-moe-routing.patch"
)
# sha256 of the FlashInfer 0.6.18.post1 source the patch was generated against.
_BASE_SOURCE_SHA256 = "182f493e45feea4d7fd3cc2c20f76195d6477123934e1dd3d1561ebaaf5342df"

# MoE-PDL subset (VLLM_MOE_PDL_FC=1): FC1 / FC2 PDL-launched, routing never.
PDL_FC = os.environ.get("VLLM_MOE_PDL_FC", "0") == "1"
LAUNCHER_SOURCE_NAME = "trtllm_fused_moe_kernel_launcher.cu"
PDL_PATCH_FILE = PATCH_FILE.parent / "0002-moe-routing-never-pdl.patch"
# sha256 of the FlashInfer 0.6.18.post1 launcher the 0002 patch was generated against.
_LAUNCHER_BASE_SHA256 = "7b7adacc9117eb64869bb68b96818af9588a85651afd83cb4a27abaae54152bf"
_PDL_MARK = b"[vllm moe-pdl-fc] routing never PDL"
# Set once gen() has produced the launcher-patched module spec; the clamp
# bypass returns False until then, so routing can never run PDL-launched from
# an unpatched module.
_PDL_MODULE_READY = False

_HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_installed = False


def _split_lines(data: bytes) -> list[bytes]:
    parts = data.split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


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


def patched_routing_source(stock_source: Path, out_dir: Path) -> Path:
    """Return the path of the patched routing source, generating it in
    ``out_dir`` from ``stock_source`` and :data:`PATCH_FILE` if needed.
    """
    if _CSRC_OVERRIDE:
        src = Path(_CSRC_OVERRIDE)
        if not src.is_file():
            raise FileNotFoundError(src)
        return src
    original = stock_source.read_bytes()
    if hashlib.sha256(original).hexdigest() != _BASE_SOURCE_SHA256:
        logger.warning(
            "exact MoE routing: %s differs from the FlashInfer source the patch "
            "was generated against; applying it with exact context matching",
            stock_source,
        )
    patched = apply_unified_diff(original, PATCH_FILE.read_bytes())
    out = out_dir / ROUTING_SOURCE_NAME
    _write_if_changed(out, patched)
    return out


def patched_launcher_source(stock_source: Path, out_dir: Path) -> Path:
    """Return the path of the 0002-patched launcher (routing never PDL),
    generating it in ``out_dir`` from ``stock_source``. Raises if the patch
    does not apply exactly or does not cover every ``routing_runner.run`` call.
    """
    original = stock_source.read_bytes()
    if hashlib.sha256(original).hexdigest() != _LAUNCHER_BASE_SHA256:
        logger.warning(
            "[moe-pdl-fc] %s differs from the FlashInfer source the 0002 patch "
            "was generated against; applying it with exact context matching",
            stock_source,
        )
    patched = apply_unified_diff(original, PDL_PATCH_FILE.read_bytes())
    n_sites = patched.count(b"routing_runner.run(")
    n_marked = patched.count(_PDL_MARK)
    if n_sites == 0 or n_marked != n_sites:
        raise RuntimeError(
            f"[moe-pdl-fc] 0002 patch covers {n_marked} of {n_sites} "
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


def maybe_install() -> None:
    """Build FlashInfer's trtllm fused-MoE module from the patched routing
    source (``GS2_ROUTE``) and / or the routing-never-PDL launcher
    (``VLLM_MOE_PDL_FC=1``). No-op unless one of them is enabled; idempotent.
    """
    global _installed
    if not (ENABLED or PDL_FC) or _installed:
        return
    from flashinfer.fused_moe import core
    from flashinfer.jit import env as jit_env

    if ENABLED:
        required = Path(_CSRC_OVERRIDE) if _CSRC_OVERRIDE else PATCH_FILE
        if not required.is_file():
            raise FileNotFoundError(required)
    if PDL_FC and not PDL_PATCH_FILE.is_file():
        raise FileNotFoundError(PDL_PATCH_FILE)
    orig = core.gen_trtllm_gen_fused_moe_sm100_module

    def gen(enable_rubin=False):
        global _PDL_MODULE_READY
        spec = orig(enable_rubin=enable_rubin)
        name = f"{spec.name}_{TAG}" if ENABLED else spec.name
        if PDL_FC:
            name = f"{name}_pdlfc"
        out_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "vllm_patched" / name
        srcs = list(spec.sources)
        src = None
        if ENABLED:
            stock = [p for p in srcs if Path(p).name == ROUTING_SOURCE_NAME]
            assert stock, f"{ROUTING_SOURCE_NAME} not in spec sources"
            src = patched_routing_source(Path(stock[0]), out_dir)
            srcs = [src if Path(p).name == ROUTING_SOURCE_NAME else p for p in srcs]
        lsrc = None
        if PDL_FC:
            stock_l = [p for p in srcs if Path(p).name == LAUNCHER_SOURCE_NAME]
            assert stock_l, f"{LAUNCHER_SOURCE_NAME} not in spec sources"
            lsrc = patched_launcher_source(Path(stock_l[0]), out_dir)
            srcs = [lsrc if Path(p).name == LAUNCHER_SOURCE_NAME else p for p in srcs]
        new = dataclasses.replace(spec, name=name, sources=srcs)
        if ENABLED:
            logger.info(
                "exact MoE routing: module %s -> %s, routing source %s",
                spec.name,
                new.name,
                src,
            )
        if PDL_FC:
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
    if PDL_FC:
        # Bypass FlashInfer's sm_107 clamp (upstream #4806) for the MoE runner
        # only; returns False until gen() has produced the patched module.
        core._device_support_moe_pdl = _moe_pdl_fc_supported
        logger.info(
            "[moe-pdl-fc] FlashInfer _device_support_moe_pdl replaced "
            "(FC1/FC2 PDL once the routing-never-PDL module is built)"
        )
    _installed = True

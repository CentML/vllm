# SPDX-License-Identifier: Apache-2.0
"""[l2-x] L2X-A: L2 prefetch of each decoder layer's non-MoE weights, issued right after the previous layer's MoE.

Exact by construction: the only device work added is `cp.async.bulk.prefetch.L2.global` (a cache hint) over static
weight ranges; no tensor is read or written by it. Default OFF (VLLM_L2X unset/0 -> every hook is a Python no-op and no
op is placed in the compiled graph).

Where it acts: only while capturing FULL CUDA graphs (uniform decode / spec-verify batches) with
num_tokens <= L2X_MAX_TOKENS. In PIECEWISE / eager steps the op does nothing (a fork may not span a piecewise split).

Mechanism per site (start of decoder layer i >= 1, input = MoE(i-1) output):
  fork  : side stream waits on the caller stream, launches a 128-CTA x 32-lane prefetch kernel over layer i's ranges
          (in_proj(s) first, then conv / norms, out_proj, router), in list order.
  join  : L2X_JOIN=deferred (default): the caller waits on the side stream at the NEXT site (or after the last layer),
          so the prefetch kernel is never on the critical path. L2X_JOIN=immediate: the optional execute_on_streams provider
          (fork + join in one call; prefetch kernel duration lands on the critical path).
  L2X_MODE=inline: no fork; the prefetch kernel is launched on the caller stream with a PDL attribute.

Env (all read once at import):
  VLLM_L2X=1             enable
  L2X_MODE=fork|inline   default fork
  L2X_JOIN=deferred|immediate
  L2X_MAX_TOKENS=512     enable only for captured batches with <= this many tokens (band gate; low<=128, mid<=1024)
  L2X_SET=attn,router,norm   what to prefetch (attn = the layer's linear_attn / self_attn params; router = mlp.gate +
                         shared_expert_gate; norm = input/post_attention layernorms)
  L2X_BUDGET_MB=64       cap per layer (ranges beyond it are dropped, in list order)
  L2X_CHUNK_KB=4, L2X_CTAS=128, L2X_THREADS=32, L2X_EVICT_LAST=0
No persisting-L2 carve-out is ever set; reset_persisting() runs at exit anyway (defensive, other tenants).
"""


import atexit
import ctypes
import hashlib
import os
import subprocess
from collections.abc import Callable, Sequence

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_L2X", "0") == "1"
MODE = os.environ.get("L2X_MODE", "fork").lower()
JOIN = os.environ.get("L2X_JOIN", "deferred").lower()
MAX_TOKENS = int(os.environ.get("L2X_MAX_TOKENS", "512"))
SET = {s for s in os.environ.get("L2X_SET", "attn,router,norm").lower().split(",") if s}
BUDGET = int(float(os.environ.get("L2X_BUDGET_MB", "64")) * (1 << 20))
CHUNK = int(os.environ.get("L2X_CHUNK_KB", "4")) * 1024
CTAS = int(os.environ.get("L2X_CTAS", "128"))
THREADS = int(os.environ.get("L2X_THREADS", "32"))
EVICT_LAST = os.environ.get("L2X_EVICT_LAST", "0") == "1"
MAX_RANGES = 48
_ON_DEVICE = lambda t: t.is_cuda  # noqa: E731 (overridable in the CPU unit test)
_EXCLUDE = ("mlp.experts", "experts.")  # MoE expert weights are never prefetched


# ---------------------------------------------------------------------------------------------- fork / join helper
try:  # Reuse the optional capture-safe fork/join provider when available.
    from vllm.model_executor.layers.quantization.utils.memsys_locality import execute_on_streams
except Exception:  # pragma: no cover - stand-alone capture-safe fork/join fallback

    def execute_on_streams(streams: Sequence[torch.cuda.Stream], fn: Callable[[int], None]) -> None:
        caller = torch.cuda.current_stream()
        start = torch.cuda.Event()
        start.record(caller)
        entered: list[torch.cuda.Stream] = []
        try:
            for i, s in enumerate(streams):
                s.wait_event(start)
                entered.append(s)
                with torch.cuda.stream(s):
                    fn(i)
        finally:
            for s in entered:
                done = torch.cuda.Event()
                done.record(s)
                caller.wait_event(done)


# ---------------------------------------------------------------------------------------------- extension
_EXT = None


class _CtypesExt:
    """qwen-v2: the kernel is a plain CUDA .so built with nvcc and loaded with ctypes. No torch/ATen headers are needed;
    the qwen-v2 amd64 image lacks cusparse.h, which torch.utils.cpp_extension + ATen/cuda/CUDAContext.h require."""

    def __init__(self, lib):
        u64p = ctypes.POINTER(ctypes.c_ulonglong)
        lib.l2x_prefetch.argtypes = [u64p, u64p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_void_p]
        lib.l2x_prefetch.restype = ctypes.c_int
        lib.l2x_reset_persisting.argtypes = []
        lib.l2x_reset_persisting.restype = ctypes.c_int
        self._lib = lib

    def prefetch(self, ptrs, sizes, chunk, ctas, threads, evict_last, pdl):
        n = int(ptrs.numel())
        if n == 0:
            return
        arr = ctypes.c_ulonglong * n
        p = arr(*[int(v) for v in ptrs.tolist()])
        b = arr(*[int(v) for v in sizes.tolist()])
        stream = torch.cuda.current_stream().cuda_stream
        rc = self._lib.l2x_prefetch(p, b, n, int(chunk), int(ctas), int(threads), int(bool(evict_last)), int(bool(pdl)),
                                    ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"[l2x] prefetch launch failed: cudaError {rc}")

    def reset_persisting(self):
        try:
            self._lib.l2x_reset_persisting()
        except Exception:  # noqa: BLE001  exit-time best effort; the process is going away anyway
            pass


def _build_dir() -> str:
    d = os.environ.get("L2X_BUILD_DIR") or os.path.join(
        os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))),
        "vllm_l2x")
    os.makedirs(d, exist_ok=True)
    return d


def _ext():
    global _EXT
    if _EXT is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l2x_prefetch.cu")
        text = open(src, "rb").read()
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        tag = hashlib.sha256(text + arch.encode()).hexdigest()[:16]
        so = os.path.join(_build_dir(), f"l2x_prefetch_sm{arch}_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            r = subprocess.run([nvcc, "-shared", "-Xcompiler", "-fPIC", "-O3", "-std=c++17",
                                f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", tmp, src],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"[l2x] nvcc build failed (rc {r.returncode}):\n{r.stderr[-4000:]}")
            os.replace(tmp, so)
            logger.info("[l2x] built %s", so)
        _EXT = _CtypesExt(ctypes.CDLL(so))
        atexit.register(_EXT.reset_persisting)
    return _EXT


# ---------------------------------------------------------------------------------------------- per-layer state
_LAYERS: dict[int, dict] = {}  # site id -> {"module": layer, "ranges": (ptrs, bytes) | None}
_LAST_ID = -1
_STATE: dict = {"side": None, "pending": None}


def _collect(layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered, de-duplicated (storage ptr, bytes) list of the layer's non-MoE weights, resolved after loading."""
    groups: list[list[tuple[str, torch.Tensor]]] = []
    attn = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
    if "attn" in SET and attn is not None:
        named = list(attn.named_parameters()) + list(attn.named_buffers())
        # earliest consumer first: input projections, then the rest in registration order (conv, norms, out_proj)
        first = [kv for kv in named if "in_proj" in kv[0] or "qkv_proj" in kv[0]]
        rest = [kv for kv in named if kv not in first]
        groups += [first, rest]
    mlp = getattr(layer, "mlp", None)
    if "router" in SET and mlp is not None:
        for name in ("gate", "shared_expert_gate"):
            m = getattr(mlp, name, None)
            if isinstance(m, torch.nn.Module):
                groups.append(list(m.named_parameters()) + list(m.named_buffers()))
    if "norm" in SET:
        for name in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(layer, name, None)
            if isinstance(m, torch.nn.Module):
                groups.append(list(m.named_parameters()))
    seen: set[int] = set()
    ptrs: list[int] = []
    sizes: list[int] = []
    total = 0
    for group in groups:
        for name, t in group:
            if not isinstance(t, torch.Tensor) or not _ON_DEVICE(t) or any(e in name for e in _EXCLUDE):
                continue
            st = t.untyped_storage()
            p, n = st.data_ptr(), st.nbytes()
            if n < 4096 or p in seen or p % 16:
                continue
            n &= ~15
            if total + n > BUDGET or len(ptrs) >= MAX_RANGES:
                continue
            seen.add(p)
            ptrs.append(p)
            sizes.append(n)
            total += n
    return torch.tensor(ptrs, dtype=torch.int64), torch.tensor(sizes, dtype=torch.int64)


def _ranges(site: int) -> tuple[torch.Tensor, torch.Tensor]:
    ent = _LAYERS[site]
    if ent["ranges"] is None:
        ent["ranges"] = _collect(ent["module"])
        p, b = ent["ranges"]
        logger.info("[l2x] site %d: %d ranges, %.1f MB", site, p.numel(), b.sum().item() / 2**20)
    return ent["ranges"]


def _active(num_tokens: int) -> bool:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available() or not torch.cuda.is_current_stream_capturing():
        return False
    return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL and num_tokens <= MAX_TOKENS


def _join_pending() -> None:
    ev = _STATE["pending"]
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
        _STATE["pending"] = None


def _launch(ptrs: torch.Tensor, sizes: torch.Tensor, pdl: bool) -> None:
    _ext().prefetch(ptrs, sizes, CHUNK, CTAS, THREADS, EVICT_LAST, pdl)


# ---------------------------------------------------------------------------------------------- custom ops
def _site_impl(x: torch.Tensor, site: int) -> None:
    # x is only an ordering anchor (declared mutated so the op stays between MoE(i-1) and layer i's norm).
    _join_pending()  # defensive: never leave a fork open across sites
    if _EXT is None and not torch.cuda.is_current_stream_capturing():
        _ext()  # build/load the extension on the first eager (profile / warmup) call, never inside a capture
    if site < 0 or not _active(x.shape[0]):
        return
    ptrs, sizes = _ranges(site)
    if ptrs.numel() == 0:
        return
    if MODE == "inline":
        _launch(ptrs, sizes, pdl=True)
        return
    if _STATE["side"] is None:
        _STATE["side"] = torch.cuda.Stream()
    side = _STATE["side"]
    if JOIN == "immediate":
        execute_on_streams([side], lambda _i: _launch(ptrs, sizes, pdl=False))
        return
    caller = torch.cuda.current_stream()
    start = torch.cuda.Event()
    start.record(caller)
    side.wait_event(start)
    done = torch.cuda.Event()
    try:
        with torch.cuda.stream(side):
            _launch(ptrs, sizes, pdl=False)
    finally:
        done.record(side)
        _STATE["pending"] = done  # joined at the next site or by l2x_join after the last layer


def _join_impl(x: torch.Tensor) -> None:
    _join_pending()


def _fake(*args, **kwargs) -> None:
    return None


if ENABLED:
    direct_register_custom_op("l2x_site", _site_impl, mutates_args=["x"], fake_impl=_fake)
    direct_register_custom_op("l2x_join", _join_impl, mutates_args=["x"], fake_impl=_fake)
    logger.info(
        "[l2x] enabled mode=%s join=%s max_tokens=%d set=%s budget=%dMB chunk=%dKB ctas=%d evict_last=%s",
        MODE, JOIN, MAX_TOKENS, sorted(SET), BUDGET >> 20, CHUNK >> 10, CTAS, EVICT_LAST,
    )


# ---------------------------------------------------------------------------------------------- model hooks
def configure_decoder_layer(layer: torch.nn.Module, prefix: str = "") -> None:
    """Called at the end of Qwen3_5DecoderLayer.__init__. Layer 0 has no preceding MoE and its input may be a graph
    input (inputs_embeds), so it gets no site. MTP draft layers ("mtp." prefix) are left untouched."""
    global _LAST_ID
    idx = int(getattr(layer, "layer_idx", -1))
    target = "mtp" not in prefix.split(".")
    layer._l2x_site = idx if (ENABLED and target and idx >= 1) else -1
    if ENABLED and target and idx >= 0:
        _LAYERS[idx] = {"module": layer, "ranges": None}
        _LAST_ID = max(_LAST_ID, idx)


def _anchor(hidden_states):
    """The tensor the site/join op is ordered on. Between decoder layers the hidden state is a Tensor, or, on F+1 with
    NQF=1 (MoE finalize deferred into the next layer's fused add-norm), a tuple of two [T, H] tensors (routed, shared).
    Anchor on the first tensor: the op then runs after the MoE that produced it and before the norm that reads it."""
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    if isinstance(hidden_states, (tuple, list)):
        for t in hidden_states:
            if isinstance(t, torch.Tensor):
                return t
    return None


def site(layer: torch.nn.Module, hidden_states) -> None:
    if ENABLED and layer._l2x_site >= 1:
        a = _anchor(hidden_states)
        if a is not None:
            torch.ops.vllm.l2x_site(a, layer._l2x_site)


def after_layer(layer: torch.nn.Module, hidden_states) -> None:
    if ENABLED and layer._l2x_site >= 1 and layer._l2x_site == _LAST_ID:
        a = _anchor(hidden_states)
        if a is not None:
            torch.ops.vllm.l2x_join(a)

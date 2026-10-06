# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute prompt prefix-cache block hashes in the front-end process.

With prefix caching enabled, EngineCore hashes every full block of each new
prompt on its input thread (``Request.__init__`` -> ``request_block_hasher`` ->
``hash_block_tokens``). For long prompts that are re-sent on every turn
(multi-turn / agentic traffic) this pure-Python hashing holds the EngineCore
GIL and delays the busy loop that schedules and launches model steps.

With ``FEH=1`` the front-end process (API server / AsyncLLM process, each with
its own GIL) computes the prompt block hashes with the same function
(``kv_cache_utils.hash_block_tokens``, the configured
``prefix_caching_hash_algo``, the same ``NONE_HASH`` seed resolution and the
same extra keys) and ships them in ``EngineCoreRequest.prompt_block_hashes``.
EngineCore then builds the ``Request`` without hashing the prompt and installs
the shipped hashes. Hashes of generated tokens are still computed by
EngineCore.

Exactness: identical hash values give identical prefix-cache matching and
therefore identical outputs. sha256-based algorithms use a fixed default
``NONE_HASH`` seed unless ``PYTHONHASHSEED`` is set, so all processes agree.
Anything unusual falls back to stock hashing in EngineCore: multimodal, LoRA,
prompt-embeds or mixed token/embeds inputs, prefix caching disabled, block
size mismatch or an unexpected number of hashes.

Environment variables (read once at import):

- ``FEH``: ``1`` enables front-end hashing and the EngineCore fast path
  (default ``0``).
- ``VLLM_FRONTEND_HASH_BLOCK_SIZE``: block size the front end hashes at
  (default ``32``). It must equal EngineCore's resolved hash block size,
  otherwise EngineCore falls back to hashing the prompt itself.
- ``VLLM_FRONTEND_HASH_VERIFY``: ``1`` makes EngineCore also hash the prompt
  itself, compare with the shipped hashes (mismatches are logged) and always
  use its own hashes. Debug aid.
- ``VLLM_FRONTEND_HASH_LOG_EVERY``: log the counters every N requests
  (default ``2000``).

Wire format of ``EngineCoreRequest.prompt_block_hashes``: an 8-byte header
(``b"FEH1"`` + block size as little-endian uint32) followed by the
concatenated 32-byte block hashes of all full prompt blocks.
"""

import os
import threading
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request

logger = init_logger(__name__)

FEH_ENABLED = os.environ.get("FEH", "0") == "1"
FEH_BLOCK = int(os.environ.get("VLLM_FRONTEND_HASH_BLOCK_SIZE", "32"))
FEH_VERIFY = os.environ.get("VLLM_FRONTEND_HASH_VERIFY", "0") == "1"
FEH_LOG_EVERY = int(os.environ.get("VLLM_FRONTEND_HASH_LOG_EVERY", "2000"))

_MAGIC = b"FEH1"
_HDR = 8  # header: _MAGIC + uint32 block size
_HASH_BYTES = 32

STATS = {
    "fe_hashed": 0,
    "fe_skipped": 0,
    "ec_used": 0,
    "ec_fallback": 0,
    "ec_verified": 0,
    "ec_mismatch": 0,
}
_lock = threading.Lock()

# ---------------------------------------------------------------- front end

_FE_STATE: dict[str, Any] = {}


def _frontend_state(cache_config: "CacheConfig") -> tuple[Any, Any]:
    st = _FE_STATE.get("st")
    if st is None:
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as kvu

        algo = cache_config.prefix_caching_hash_algo
        fn = get_hash_fn_by_name(algo)
        # Same seed resolution as EngineCore (sha256: fixed default seed).
        kvu.init_none_hash(fn)
        st = _FE_STATE["st"] = (fn, kvu)
        logger.info(
            "FEH front-end block hashing on: algo=%s block=%d none_hash=%s",
            algo,
            FEH_BLOCK,
            kvu.NONE_HASH.hex()[:16],
        )
    return st


def _compute_prompt_block_hashes(
    cache_config: "CacheConfig", request: "EngineCoreRequest"
) -> bytes | None:
    toks = request.prompt_token_ids
    if (
        toks is None
        or request.mm_features
        or request.lora_request is not None
        or request.prompt_embeds is not None
        or request.prompt_is_token_ids is not None
        or not cache_config.enable_prefix_caching
    ):
        return None
    fn, kvu = _frontend_state(cache_config)
    salt = request.cache_salt
    parent = None
    out = [_MAGIC, FEH_BLOCK.to_bytes(4, "little")]
    n = len(toks) // FEH_BLOCK
    for i in range(n):
        # Same extra keys as generate_block_hash_extra_keys() for a request
        # without multimodal / LoRA / prompt-embeds inputs.
        extra = (salt,) if (i == 0 and salt) else None
        h = kvu.hash_block_tokens(
            fn, parent, toks[i * FEH_BLOCK : (i + 1) * FEH_BLOCK], extra
        )
        out.append(h)
        parent = h
    return b"".join(out)


def log_frontend_enabled() -> None:
    logger.info_once(
        "FEH front-end prompt block hashing enabled (block=%d)", FEH_BLOCK
    )


def attach_prompt_block_hashes(
    cache_config: "CacheConfig", request: "EngineCoreRequest"
) -> None:
    """Front end: set ``request.prompt_block_hashes`` when eligible."""
    try:
        blob = _compute_prompt_block_hashes(cache_config, request)
    except Exception as e:
        # Never break serving: EngineCore falls back to its own hashing.
        logger.warning(
            "FEH front-end hashing failed (%r); request sent without hashes", e
        )
        blob = None
    if blob is None:
        STATS["fe_skipped"] += 1
        return
    STATS["fe_hashed"] += 1
    if STATS["fe_hashed"] % FEH_LOG_EVERY == 1:
        logger.info("FEH frontend stats %s", STATS)
    request.prompt_block_hashes = blob


# -------------------------------------------------------------- engine core


def log_engine_enabled() -> None:
    logger.info_once(
        "FEH EngineCore accepts front-end prompt block hashes (verify=%s)",
        FEH_VERIFY,
    )


def shipped_hashes_usable(
    blob: bytes, request: "EngineCoreRequest", hash_block_size: int | None
) -> bool:
    """Check header, block size and hash count against the prompt."""
    bs = hash_block_size
    return (
        blob[:4] == _MAGIC
        and int.from_bytes(blob[4:8], "little") == bs
        and bs == FEH_BLOCK
        and (len(blob) - _HDR) % _HASH_BYTES == 0
        and (len(blob) - _HDR) // _HASH_BYTES
        == len(request.prompt_token_ids or ()) // bs
    )


def unpack_block_hashes(blob: bytes) -> list["BlockHash"]:
    mv = memoryview(blob)
    return cast(
        "list[BlockHash]",
        [bytes(mv[i : i + _HASH_BYTES]) for i in range(_HDR, len(blob), _HASH_BYTES)],
    )


def count_engine_fallback() -> None:
    with _lock:
        STATS["ec_fallback"] += 1


def count_engine_used() -> None:
    with _lock:
        STATS["ec_used"] += 1
    _maybe_log_engine()


def verify_block_hashes(req: "Request", blob: bytes) -> None:
    """Compare EngineCore's own prompt hashes with the shipped ones."""
    mine = b"".join(req.block_hashes[: (len(blob) - _HDR) // _HASH_BYTES])
    with _lock:
        STATS["ec_verified"] += 1
        if mine != blob[_HDR:]:
            STATS["ec_mismatch"] += 1
            logger.warning(
                "FEH prompt block hash MISMATCH for request %s", req.request_id
            )
    _maybe_log_engine()


def _maybe_log_engine() -> None:
    n = STATS["ec_used"] + STATS["ec_verified"]
    if n % FEH_LOG_EVERY == 1:
        logger.info("FEH engine stats %s", STATS)

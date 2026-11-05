# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send long prompts' token ids from the front end to EngineCore as packed int32.

Without it, an ADD request's ``prompt_token_ids`` crosses the front-end ->
EngineCore ZMQ socket as a msgpack array: the front end encodes one msgpack int
per token, and the EngineCore input thread decodes one Python int per token
(holding the GIL) and then converts that list back into the int32
``array`` that ``VLLM_PROMPT_TOKEN_ARRAY`` stages for the model runner. For
prompts of tens of thousands of tokens this costs input-thread CPU time per
turn, in series with the request's TTFT.

With ``VLLM_PACKED_PROMPT_IDS=1`` the front end sends the ids as one msgpack
``bin`` of little-endian int32 (the wire format of vLLM PR #62702, "Keep prompt
token ids as int32 buffers from vllm-rs to the workers"), built from the int32
array the front end already makes for FEH (``frontend_block_hashing``) when
there is one. EngineCore copies the bytes into an ``array('i')`` (a memcpy),
derives the Python list from it with one ``tolist()`` and hands the same array
to ``Request`` as its ``prompt_token_array``. ``Request`` sees exactly the
same list of ints as before; only the transport changes.

Scope: requests whose ``prompt_token_ids`` is a list of at least
``VLLM_PACKED_PROMPT_IDS_MIN`` ids that all fit int32, with no prompt embeds
or mixed token/embeds mask. Everything else is sent as before. Little-endian
hosts only (the sender falls back to the list elsewhere).

Environment variables (read once at import; set the same in every process):

- ``VLLM_PACKED_PROMPT_IDS``: ``1`` enables it (default ``0``: no behaviour
  change, and ``EngineCoreRequest`` keeps its ``list[int] | None`` field type).
- ``VLLM_PACKED_PROMPT_IDS_MIN``: minimum prompt length to pack (default 1024).
- ``VLLM_PACKED_PROMPT_IDS_VERIFY``: check the first N packed requests on the
  sender (array == list, else the list is sent and a warning is logged).
- ``VLLM_PACKED_PROMPT_IDS_TIME``: ``1`` times the EngineCore input thread's
  decode + preprocess (wall time, GIL waits included) of every ADD request
  and logs the means of packed and of long (>= MIN) list requests every
  ``VLLM_PACKED_PROMPT_IDS_LOG_EVERY`` requests (default 2000).
- ``VLLM_PACKED_PROMPT_IDS_AB``: ``1`` packs only every other eligible request,
  so one run measures both transports' ingest time under the same load.
- ``VLLM_PROMPT_TOKEN_BUFFER``: ``1`` (with ``VLLM_PROMPT_TOKEN_ARRAY=1``) skips
  the ``tolist()``: a packed prompt becomes a ``TokenBuffer``
  (``vllm/v1/token_buffer.py``, int32 storage that reads like a list), and
  ``Request`` keeps its prompt and all-token ids in that form. Consumers see
  the same values; only list objects are built where a consumer slices or
  copies.
"""

import array
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_PACKED_PROMPT_IDS", "0") == "1"
MIN_TOKENS = int(os.environ.get("VLLM_PACKED_PROMPT_IDS_MIN", "1024") or 1024)
_VERIFY_LEFT = [int(os.environ.get("VLLM_PACKED_PROMPT_IDS_VERIFY", "0") or 0)]
TIME = os.environ.get("VLLM_PACKED_PROMPT_IDS_TIME", "0") == "1"
LOG_EVERY = int(os.environ.get("VLLM_PACKED_PROMPT_IDS_LOG_EVERY", "2000") or 2000)
AB = os.environ.get("VLLM_PACKED_PROMPT_IDS_AB", "0") == "1"
# The wire format is little-endian int32; array('i') must be exactly that.
_NATIVE_OK = sys.byteorder == "little" and array.array("i").itemsize == 4
# VLLM_PROMPT_TOKEN_BUFFER: keep a packed prompt as int32 (no per-token ints).
# Only with VLLM_PROMPT_TOKEN_ARRAY=1, whose staging array it relies on.
_TOKEN_BUFFER_ENV = os.environ.get("VLLM_PROMPT_TOKEN_BUFFER", "0") == "1"
TOKEN_BUFFER = (
    _TOKEN_BUFFER_ENV and os.environ.get("VLLM_PROMPT_TOKEN_ARRAY", "0") == "1"
)
if _TOKEN_BUFFER_ENV and not TOKEN_BUFFER:
    logger.warning(
        "VLLM_PROMPT_TOKEN_BUFFER=1 ignored: it needs VLLM_PROMPT_TOKEN_ARRAY=1"
    )

_lock = threading.Lock()
SEND_STATS = {"packed": 0, "handoff_hit": 0, "handoff_miss": 0, "fallback": 0,
              "ab_list": 0}
RECV_STATS = {"packed": 0, "tokens": 0, "unpack_ms": 0.0}
# Per transport: requests, ingest ms, prompt tokens ("list": >= MIN tokens only).
TIME_STATS = {"n": 0, "packed": [0, 0.0, 0], "list": [0, 0.0, 0]}

# Front end: int32 arrays built in process_inputs (the renderer thread, where
# FEH already made one), keyed by the prompt list's id; the list itself is
# held so the id cannot be reused while the entry lives. Taken at send time
# (the event loop thread). Bounded: an entry whose request never reaches the
# send (rejected after processing) is evicted by later requests.
_HANDOFF_MAX = 16
_handoff: "OrderedDict[int, tuple[list, array.array]]" = OrderedDict()


def _int32_array(toks: list) -> array.array | None:
    """The exact int32 array of ``toks``: FEH's cached one if it was built
    for this same list object, else a new one. None if a value does not fit.
    """
    from vllm.v1.engine import frontend_block_hashing

    arr = frontend_block_hashing.peek_int32_array(toks)
    if arr is not None:
        return arr
    try:
        return array.array("i", toks)
    except (OverflowError, TypeError):
        return None


def _eligible(request: "EngineCoreRequest") -> bool:
    toks = request.prompt_token_ids
    return (
        _NATIVE_OK
        and type(toks) is list
        and len(toks) >= MIN_TOKENS
        and request.prompt_embeds is None
        and request.prompt_is_token_ids is None
    )


def stash(request: "EngineCoreRequest") -> None:
    """Front end, end of process_inputs: build (or reuse FEH's) int32 array of
    the prompt off the event loop and keep it for the send.
    """
    if not (ENABLED and _eligible(request)):
        return
    toks = request.prompt_token_ids
    arr = _int32_array(toks)  # type: ignore[arg-type]
    if arr is None:
        return
    with _lock:
        _handoff[id(toks)] = (toks, arr)  # type: ignore[assignment]
        while len(_handoff) > _HANDOFF_MAX:
            _handoff.popitem(last=False)


def pack_request(request: "EngineCoreRequest") -> "EngineCoreRequest":
    """Front end, at send: a shallow copy of an eligible ADD request whose
    prompt_token_ids are little-endian int32 bytes; otherwise the request.
    The caller's request (also held by the output processor) keeps its list.
    """
    if not (ENABLED and _eligible(request)):
        return request
    toks: list = request.prompt_token_ids  # type: ignore[assignment]
    with _lock:
        entry = _handoff.pop(id(toks), None)
    if AB and (SEND_STATS["packed"] + SEND_STATS["ab_list"]) % 2:
        SEND_STATS["ab_list"] += 1
        return request
    if entry is not None and entry[0] is toks and len(entry[1]) == len(toks):
        arr = entry[1]
        SEND_STATS["handoff_hit"] += 1
    else:
        arr = _int32_array(toks)
        SEND_STATS["handoff_miss"] += 1
    if arr is None:
        SEND_STATS["fallback"] += 1
        return request
    if _VERIFY_LEFT[0] > 0:
        _VERIFY_LEFT[0] -= 1
        if arr.tolist() != toks:
            SEND_STATS["fallback"] += 1
            logger.warning(
                "Packed prompt ids VERIFY MISMATCH for request %s; sending the "
                "list", request.request_id,
            )
            return request
    SEND_STATS["packed"] += 1
    n = SEND_STATS["packed"]
    if n == 1:
        logger.info(
            "Packed prompt ids on (VLLM_PACKED_PROMPT_IDS=1): ADD requests with "
            ">= %d prompt tokens carry little-endian int32 bytes (first: %d "
            "tokens)", MIN_TOKENS, len(toks),
        )
    if n % LOG_EVERY == 0:
        logger.info("Packed prompt ids front-end stats %s", SEND_STATS)
    # memoryview: msgspec copies it into the message as one msgpack bin.
    return msgspec.structs.replace(
        request, prompt_token_ids=memoryview(arr).cast("B")
    )


def is_packed(prompt_token_ids: Any) -> bool:
    return type(prompt_token_ids) is bytes


def unpack_request(request: "EngineCoreRequest") -> array.array:
    """EngineCore input thread: replace the packed bytes by the list of ints and
    return the int32 array it was made from (Request's prompt_token_array).
    """
    t0 = time.perf_counter()
    data: bytes = request.prompt_token_ids  # type: ignore[assignment]
    if len(data) % 4:
        raise ValueError(
            f"packed prompt_token_ids of request {request.request_id} has "
            f"{len(data)} bytes, not a multiple of 4"
        )
    arr = array.array("i")
    arr.frombytes(data)
    if sys.byteorder != "little":
        arr.byteswap()
    if TOKEN_BUFFER:
        from vllm.v1.token_buffer import TokenBuffer

        # A memcpy instead of one Python int per token; Request keeps it.
        request.prompt_token_ids = TokenBuffer(arr)  # type: ignore[assignment]
    else:
        request.prompt_token_ids = arr.tolist()
    RECV_STATS["packed"] += 1
    RECV_STATS["tokens"] += len(arr)
    RECV_STATS["unpack_ms"] += (time.perf_counter() - t0) * 1e3
    if RECV_STATS["packed"] == 1:
        logger.info(
            "EngineCore received packed int32 prompt ids "
            "(VLLM_PACKED_PROMPT_IDS=1; first: %d tokens)", len(arr),
        )
        if TOKEN_BUFFER:
            logger.info(
                "Packed prompt ids kept as int32 TokenBuffer, no tolist "
                "(VLLM_PROMPT_TOKEN_BUFFER=1): Request prompt and all-token ids "
                "hold no per-token ints"
            )
    return arr


def record_ingest(seconds: float, num_tokens: int, packed: bool) -> None:
    """VLLM_PACKED_PROMPT_IDS_TIME: one ADD request's decode + preprocess."""
    TIME_STATS["n"] += 1
    if packed or num_tokens >= MIN_TOKENS:
        b = TIME_STATS["packed" if packed else "list"]
        b[0] += 1
        b[1] += seconds * 1e3
        b[2] += num_tokens
    n = TIME_STATS["n"]
    if n == 1 or n % LOG_EVERY == 0:
        p, li = TIME_STATS["packed"], TIME_STATS["list"]
        logger.info(
            "EngineCore ADD ingest (decode + preprocess, input thread): reqs=%d "
            "| packed n=%d mean_ms=%.3f mean_tokens=%.0f unpack_mean_ms=%.3f "
            "| list(>=%d tok) n=%d mean_ms=%.3f mean_tokens=%.0f",
            n, p[0], p[1] / max(1, p[0]), p[2] / max(1, p[0]),
            RECV_STATS["unpack_ms"] / max(1, RECV_STATS["packed"]),
            MIN_TOKENS, li[0], li[1] / max(1, li[0]), li[2] / max(1, li[0]),
        )

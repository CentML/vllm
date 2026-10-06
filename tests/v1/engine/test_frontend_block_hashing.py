# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Front-end prompt block hashes must equal EngineCore's own block hashes."""

import random
from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils as kvu
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import frontend_block_hashing as feh
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

BLOCK = 32


def _make_request(num_tokens: int, cache_salt: str | None) -> EngineCoreRequest:
    rng = random.Random(num_tokens)
    return EngineCoreRequest(
        request_id="r",
        prompt_token_ids=[rng.randrange(248000) for _ in range(num_tokens)],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=cache_salt,
        data_parallel_rank=None,
    )


@pytest.fixture
def hasher(monkeypatch):
    monkeypatch.setattr(feh, "FEH_BLOCK", BLOCK)
    fn = get_hash_fn_by_name("sha256")
    kvu.init_none_hash(fn)
    return kvu.get_request_block_hasher(BLOCK, fn)


CACHE_CONFIG = SimpleNamespace(
    prefix_caching_hash_algo="sha256", enable_prefix_caching=True
)


@pytest.mark.parametrize("num_tokens", [5, 31, 32, 33, 64, 1000, 4097])
@pytest.mark.parametrize("cache_salt", [None, "abc"])
def test_frontend_hashes_match_engine(hasher, num_tokens, cache_salt):
    request = _make_request(num_tokens, cache_salt)
    feh.attach_prompt_block_hashes(CACHE_CONFIG, request)
    blob = request.prompt_block_hashes
    assert blob is not None

    # Survives the front-end -> EngineCore msgpack round trip.
    decoded = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(request)
    )
    assert decoded.prompt_block_hashes == blob
    assert decoded.prompt_token_ids == request.prompt_token_ids

    # Identical to what EngineCore computes itself.
    decoded.prompt_block_hashes = None
    stock = Request.from_engine_core_request(decoded, hasher)
    assert len(stock.block_hashes) == num_tokens // BLOCK
    assert blob[8:] == b"".join(stock.block_hashes)

    # EngineCore accepts it and installs the same hashes.
    assert feh.shipped_hashes_usable(blob, decoded, BLOCK)
    shipped = Request.from_engine_core_request(
        decoded, hasher, block_hashes=feh.unpack_block_hashes(blob)
    )
    assert shipped.block_hashes == stock.block_hashes

    # Generated tokens are still hashed by the attached hasher.
    shipped.append_output_token_ids([1] * BLOCK)
    stock.append_output_token_ids([1] * BLOCK)
    assert shipped.block_hashes == stock.block_hashes


def test_engine_rejects_mismatched_hashes(hasher):
    request = _make_request(100, None)
    feh.attach_prompt_block_hashes(CACHE_CONFIG, request)
    blob = request.prompt_block_hashes
    assert blob is not None
    assert not feh.shipped_hashes_usable(blob, request, 16)
    assert not feh.shipped_hashes_usable(blob[:-1], request, BLOCK)
    assert not feh.shipped_hashes_usable(blob[:-32], request, BLOCK)
    request.prompt_token_ids = request.prompt_token_ids + [0] * BLOCK
    assert not feh.shipped_hashes_usable(blob, request, BLOCK)


def test_frontend_skips_unsupported_requests(hasher):
    request = _make_request(100, None)
    request.lora_request = SimpleNamespace()  # type: ignore[assignment]
    feh.attach_prompt_block_hashes(CACHE_CONFIG, request)
    assert request.prompt_block_hashes is None

    request = _make_request(100, None)
    no_caching = SimpleNamespace(
        prefix_caching_hash_algo="sha256", enable_prefix_caching=False
    )
    feh.attach_prompt_block_hashes(no_caching, request)  # type: ignore[arg-type]
    assert request.prompt_block_hashes is None


def test_request_without_hashes_round_trips():
    request = _make_request(100, None)
    decoded = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(request)
    )
    assert decoded.prompt_block_hashes is None

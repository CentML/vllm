# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dec107: dump real attention inputs for the MXFP4-K error check (opt-in, ``DEC107_KVDUMP=<dir>``).

Saves, for prefill-only forward calls of the first ``DEC107_KVDUMP_LAYERS`` (default 3) attention layers, the call's
query (as given to the backend, plus layer._q_scale_float), key, value (pre-cache, model dtype) and the k/v scales, up to
``DEC107_KVDUMP_MAX`` (default 24) files of >= ``DEC107_KVDUMP_MIN_T`` (default 2048) tokens.
Never during CUDA-graph capture. Dumps contain model inputs and must be handled as sensitive data.
"""
import os
import time

import torch

DIR = os.environ.get("DEC107_KVDUMP", "")
ENABLED = bool(DIR)
_MAX = int(os.environ.get("DEC107_KVDUMP_MAX", "24"))
_NL = int(os.environ.get("DEC107_KVDUMP_LAYERS", "3"))
_MIN_T = int(os.environ.get("DEC107_KVDUMP_MIN_T", "2048"))
_DELAY = float(os.environ.get("DEC107_KVDUMP_DELAY_S", "1500"))   # skip the profile / warm-up dummy runs
_EVERY = int(os.environ.get("DEC107_KVDUMP_EVERY", "7"))           # spread dumps over many prompts
_T0 = time.monotonic()
_n = 0
_seen = 0
_layers: dict[str, int] = {}


def maybe_dump(layer, query, key, value, attn_metadata) -> None:
    global _n
    if _n >= _MAX or torch.cuda.is_current_stream_capturing():
        return
    if getattr(attn_metadata, "num_decode_tokens", 0) or attn_metadata.num_actual_tokens < _MIN_T:
        return
    if time.monotonic() - _T0 < _DELAY:
        return
    name = getattr(layer, "layer_name", str(id(layer)))
    if name not in _layers:
        if len(_layers) >= _NL:
            return
        _layers[name] = len(_layers)
    global _seen
    if _layers.get(name) == 0:
        _seen += 1
    if (_seen - 1) % _EVERY:
        return
    T = attn_metadata.num_actual_tokens
    if not (torch.isfinite(key[:T]).all() and torch.isfinite(value[:T]).all()):
        return
    os.makedirs(DIR, exist_ok=True)
    torch.save({
        "layer": name, "q": query[:T].detach().cpu(), "k": key[:T].detach().cpu(), "v": value[:T].detach().cpu(),
        "q_scale": float(getattr(layer, "_q_scale_float", 1.0)), "k_scale": float(getattr(layer, "_k_scale_float", 1.0)),
        "v_scale": float(getattr(layer, "_v_scale_float", 1.0)),
    }, os.path.join(DIR, f"kv_{os.getpid()}_{_n:03d}.pt"))
    _n += 1

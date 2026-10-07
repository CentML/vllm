# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in census of the GDN prefill shapes of mixed steps (diagnostics only).

VLLM_GDNP_CENSUS=<dir>: for every step with prefill rows, the first GDN
KV-cache group's metadata build appends one JSON line to
<dir>/census-<host>-<pid>.jsonl:
  {"t": wall s, "nr": spec requests, "S": spec tokens, "nd": decodes,
   "np": prefills, "P": prefill tokens, "lens": [prefill chunk lengths],
   "mx": max chunk}
Host-only (reads the CPU copy of the prefill query_start_loc); no GPU work,
no syncs. VLLM_GDNP_CENSUS_MAX (default 200000) caps the records per process.
"""

import atexit
import json
import os
import socket
import time

_DIR = os.environ.get("VLLM_GDNP_CENSUS", "")
ENABLED = bool(_DIR)
_MAX = int(os.environ.get("VLLM_GDNP_CENSUS_MAX", "200000"))
_ST: dict = {"n": 0, "buf": [], "f": None, "first": None, "tf": 0.0}


def _flush() -> None:
    if not _ST["buf"]:
        return
    if _ST["f"] is None:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(
            _DIR, f"census-{socket.gethostname()}-{os.getpid()}.jsonl"
        )
        _ST["f"] = open(path, "a", buffering=1 << 16)  # noqa: SIM115
        atexit.register(_flush)
    _ST["f"].write("".join(_ST["buf"]))
    _ST["f"].flush()
    _ST["buf"] = []


def record(builder, prefill_query_start_loc_cpu, md) -> None:
    """Record one step. ``prefill_query_start_loc_cpu`` is the host
    cumulative-length tensor of the prefill rows (GDNSharedBuild), ``md`` the
    GDNAttentionMetadata just built."""
    try:
        if _ST["n"] >= _MAX:
            return
        np_ = int(md.num_prefills)
        if np_ <= 0 or prefill_query_start_loc_cpu is None:
            return
        key = tuple(builder.layer_names)
        if _ST["first"] is None:
            _ST["first"] = key
        if key != _ST["first"]:
            return  # one record per step (first GDN group only)
        qsl = prefill_query_start_loc_cpu.tolist()
        lens = [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
        rec = {
            "t": round(time.time(), 3),
            "nr": int(md.num_spec_decodes),
            "S": int(md.num_spec_decode_tokens),
            "nd": int(md.num_decodes),
            "np": np_,
            "P": int(md.num_prefill_tokens),
            "lens": lens,
            "mx": max(lens, default=0),
        }
        _ST["buf"].append(json.dumps(rec, separators=(",", ":")) + "\n")
        _ST["n"] += 1
        now = time.monotonic()
        if len(_ST["buf"]) >= 200 or now - _ST["tf"] > 20.0:
            _ST["tf"] = now
            _flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break serving
        _ST["n"] = _MAX

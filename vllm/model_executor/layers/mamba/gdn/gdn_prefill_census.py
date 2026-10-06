# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in census of the GDN prefill shapes of mixed steps (diagnostics only).

VLLM_GDNP_CENSUS=<dir>: for every step with prefill rows, the first GDN
KV-cache group's metadata build appends one JSON line to
<dir>/census-<host>-<pid>.jsonl:
  {"t": wall s, "nr": spec requests, "S": spec tokens, "np": prefills,
   "P": prefill tokens, "lens": [prefill chunk lengths], "mx": max chunk,
   "pk": packed for the GDN layer graphs (0/1), "vsf": V-split variant}
Host-only (reads the CPU copy of query_start_loc); no GPU work, no syncs.
VLLM_GDNP_CENSUS_MAX (default 200000) caps the records per process.
"""

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
    _ST["f"].write("".join(_ST["buf"]))
    _ST["f"].flush()
    _ST["buf"] = []


def record(builder, common_attn_metadata, md) -> None:
    try:
        if _ST["n"] >= _MAX:
            return
        np_ = int(getattr(md, "num_prefills", 0) or 0)
        if np_ <= 0:
            return
        key = tuple(builder.layer_names)
        if _ST["first"] is None:
            _ST["first"] = key
        if key != _ST["first"]:
            return  # one record per step (first GDN group only)
        qsl = common_attn_metadata.query_start_loc_cpu
        nreq = int(common_attn_metadata.num_reqs)
        ql = (qsl[1 : nreq + 1] - qsl[:nreq]).tolist()
        lens = ql[-np_:]
        d = md.__dict__
        rec = {
            "t": round(time.time(), 3),
            "nr": int(getattr(md, "num_spec_decodes", 0) or 0),
            "S": int(getattr(md, "num_spec_decode_tokens", 0) or 0),
            "nd": int(getattr(md, "num_decodes", 0) or 0),
            "np": np_,
            "P": int(getattr(md, "num_prefill_tokens", 0) or 0),
            "lens": lens,
            "mx": int(getattr(md, "prefill_max_seqlen", 0) or 0),
            "pk": int(bool(d.get("_step_plan_graph"))),
            "vsf": d.get("_step_plan_vsf"),
        }
        _ST["buf"].append(json.dumps(rec, separators=(",", ":")) + "\n")
        _ST["n"] += 1
        now = time.monotonic()
        if len(_ST["buf"]) >= 200 or now - _ST["tf"] > 20.0:
            _ST["tf"] = now
            _flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break serving
        _ST["n"] = _MAX

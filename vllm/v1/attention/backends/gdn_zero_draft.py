# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-decode row mask of the GDN metadata builder, with zero-draft decode rows as 1-token spec rows.

Bug fixed (VLLM_GDN_ZERO_DRAFT_AS_SPEC=1, default): with MTP, a request's spec kernel reads its initial SSM state from
block column ``num_accepted - 1`` and the conv window at offset ``num_accepted - 1``; the align pre/post-processing only
moves that state to column 0 when the block column changes. A decode row that runs WITHOUT drafts after an in-block
verify with acc > 1 (budget-clipped drafts, the max_model_len tail, structured-output rejects, an all-zero-draft batch)
was routed to the non-spec decode or 1-token prefill path, which read column 0 (the state after the previous step's
first token) and conv offset 0: the acc - 1 accepted tokens were lost from the recurrent state.
Fix: every decode row (1 scheduled token, prior context) of a spec-decode batch is a spec row (T = 1). The spec kernel
then reads column acc - 1 and the conv update uses offset acc - 1, exactly as for a row with drafts. Rows whose stale
draft count does not match their query length are re-derived from the query length.
Numerics: the affected rows are corrected; zero-draft rows with acc == 1 now take the spec kernel instead of the
non-spec decode / prefill kernel (same math, float-order class).
"""
import os

import torch

ZERO_DRAFT_AS_SPEC = os.environ.get("VLLM_GDN_ZERO_DRAFT_AS_SPEC", "1") == "1"


def spec_row_mask(query_lens_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor | None,
                  num_decode_draft_tokens_cpu: torch.Tensor) -> torch.Tensor:
    """Bool [num_reqs]: rows that run on the spec-decode path. Rows with drafts (num_decode_draft_tokens >= 0 and
    query_len == drafts + 1), plus decode rows without drafts (query_len == 1 and prior context). Without
    seq_lens (unknown context) only the draft rows are returned."""
    ndt = num_decode_draft_tokens_cpu
    draft_rows = (ndt >= 0) & (query_lens_cpu == ndt + 1)
    if seq_lens_cpu is None or seq_lens_cpu.shape[0] != query_lens_cpu.shape[0]:
        return draft_rows
    zero_draft_decode = (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    return draft_rows | zero_draft_decode


# ----------------------------------------------------------------------------------------------------------------------
# VLLM_GDN_R4_COUNT=1 (default; log-only, no numerics change): per worker, count the decode rows that the STOCK rule
# sends to the non-spec path in a spec-decode-enabled model ("exposed": query_len 1 with prior context and no drafts,
# or every row of an all-zero-draft batch), and how many of those had num_accepted > 1 at metadata build time = the
# stale-state rows (preprocess_mamba runs before the build and resets num_accepted to 1 when the block column changed,
# so acc > 1 here means the in-block case). Also total decode rows. The acc > 1 test accumulates on the device (no per-step sync); the
# counters are read and logged every VLLM_GDN_R4_COUNT_EVERY steps and at exit.
# ----------------------------------------------------------------------------------------------------------------------
import atexit  # noqa: E402
import logging  # noqa: E402

R4_COUNT = os.environ.get("VLLM_GDN_R4_COUNT", "1") == "1"
R4_COUNT_EVERY = int(os.environ.get("VLLM_GDN_R4_COUNT_EVERY", "2000"))
_log = logging.getLogger("vllm.gdn_r4_count")
_C = {"steps": 0, "decode_rows": 0, "exposed_rows": 0, "dev": None, "last": None}


def stock_exposed_mask(query_lens_cpu, seq_lens_cpu, num_decode_draft_tokens_cpu):
    """Decode rows the stock classifier (ndt >= 0 = spec; all-zero-draft batch -> all non-spec) runs as non-spec."""
    n = query_lens_cpu.shape[0]
    ndt = num_decode_draft_tokens_cpu[:n]
    decode = query_lens_cpu == 1
    if seq_lens_cpu is not None and seq_lens_cpu.shape[0] == n:
        decode = decode & (seq_lens_cpu > 1)
    spec = ndt >= 0
    if int(spec.sum()) == 0 or int(ndt[spec].sum()) == 0:
        spec = torch.zeros_like(spec)
    return decode & ~spec, decode


def count(m, num_decode_draft_tokens_cpu, num_accepted_tokens) -> None:
    """Called once per step from GDNAttentionMetadataBuilder.build (deduplicated across KV-cache groups)."""
    if not R4_COUNT or num_decode_draft_tokens_cpu is None or torch.cuda.is_current_stream_capturing():
        return  # never records device work into a CUDA graph
    key = (id(m.query_start_loc_cpu), int(m.num_actual_tokens))
    if key == _C["last"]:
        return
    _C["last"] = key
    qsl = m.query_start_loc_cpu
    ql = qsl[1:] - qsl[:-1]
    exposed, decode = stock_exposed_mask(ql, m.seq_lens_cpu_upper_bound, num_decode_draft_tokens_cpu)
    _C["steps"] += 1
    _C["decode_rows"] += int(decode.sum())
    ne = int(exposed.sum())
    if ne and num_accepted_tokens is not None:
        _C["exposed_rows"] += ne
        if _C["dev"] is None:
            _C["dev"] = torch.zeros(1, dtype=torch.int64, device=num_accepted_tokens.device)
        # only on steps with exposed rows (rare): a pinned host index -> async H2D, device gather/compare/add;
        # no device->host read (the counter is read only in report())
        idx = exposed.nonzero().flatten().to(torch.int64).pin_memory()
        _C["dev"] += (num_accepted_tokens.index_select(0, idx.to(num_accepted_tokens.device, non_blocking=True))
                      > 1).sum()
    if R4_COUNT_EVERY > 0 and _C["steps"] % R4_COUNT_EVERY == 0 and not torch.cuda.is_current_stream_capturing():
        report("periodic")


def report(tag) -> None:
    acc_gt1 = int(_C["dev"].item()) if _C["dev"] is not None else 0
    _log.warning("GDN R4 counter (%s): steps %d, decode rows %d, stock-non-spec decode rows %d, of which acc>1 %d "
                 "(%.4f%% of decode rows = stale-state rows under the stock rule)", tag, _C["steps"], _C["decode_rows"],
                 _C["exposed_rows"], acc_gt1, 100.0 * acc_gt1 / max(1, _C["decode_rows"]))


atexit.register(lambda: R4_COUNT and _C["steps"] and report("exit"))

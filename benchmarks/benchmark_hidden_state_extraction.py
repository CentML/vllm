# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark hidden state extraction throughput.

Measures two modes:
  1. Baseline: bulk inference with max_tokens=1, no extraction.
  2. Extract:  async hidden state extraction via ExampleHiddenStatesConnector
               with N concurrent clients, each consuming hidden states as
               soon as their request finishes (overlapping I/O with generation).

Reports tokens/s and prompts/s for each mode.

Usage:
  python benchmarks/benchmark_hidden_state_extraction.py \
      --model Qwen/Qwen3-0.6B \
      --num-prompts 64 \
      --num-clients 8 \
      --prompt-len 8192 \
      --layers 1 2 3 4
"""

import argparse
import asyncio
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open
from transformers import AutoConfig

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


def make_random_prompts(
    num_prompts: int, prompt_len: int, vocab_size: int, seed: int = 42
) -> list[list[int]]:
    """Generate lists of random token IDs."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (prompt_len,)).tolist() for _ in range(num_prompts)
    ]


def consume_hidden_states(path: str) -> float:
    """Load hidden states from disk and compute per-position mean.

    Returns a single float: the grand mean of all hidden state values.
    This forces the benchmark to actually read and reduce the data.
    """
    with safe_open(path, "pt") as f:
        hs = f.get_tensor("hidden_states")
    return hs.mean().item()


def run_baseline(model: str, prompts: list[list[int]], extra_args: dict) -> dict:
    """Baseline: bulk inference, no hidden state extraction."""
    llm = LLM(model=model, enable_prefix_caching=False, **extra_args)
    sampling_params = SamplingParams(max_tokens=1)
    prompt_inputs = [{"prompt_token_ids": p} for p in prompts]

    # Warmup
    llm.generate(prompt_inputs[:4], sampling_params, use_tqdm=False)

    t0 = time.perf_counter()
    outputs = llm.generate(prompt_inputs, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - t0

    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    num_prompts = len(outputs)

    del llm
    torch.cuda.empty_cache()

    return {
        "mode": "baseline",
        "elapsed_s": elapsed,
        "num_prompts": num_prompts,
        "total_prompt_tokens": total_prompt_tokens,
        "tokens_per_s": total_prompt_tokens / elapsed,
        "prompts_per_s": num_prompts / elapsed,
    }


# ---- Async extraction benchmark ----


async def _client_loop(
    engine: AsyncLLM,
    prompt_queue: asyncio.Queue,
    consume_pool: ThreadPoolExecutor,
    results: list[dict],
    client_id: int,
):
    """A single async client: pulls prompts, submits to engine, consumes
    hidden states as soon as each request finishes."""
    loop = asyncio.get_event_loop()
    while True:
        item = await prompt_queue.get()
        if item is None:
            prompt_queue.task_done()
            break
        idx, token_ids = item

        request_id = f"req-{idx}"
        sampling_params = SamplingParams(
            max_tokens=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        final_output = None
        async for output in engine.generate(
            request_id=request_id,
            prompt={"prompt_token_ids": token_ids},
            sampling_params=sampling_params,
        ):
            if output.finished:
                final_output = output

        # Consume hidden states on a thread (disk I/O)
        path = final_output.kv_transfer_params["hidden_states_path"]
        mean_val = await loop.run_in_executor(consume_pool, consume_hidden_states, path)
        num_tokens = len(final_output.prompt_token_ids)

        results.append(
            {
                "request_id": request_id,
                "num_prompt_tokens": num_tokens,
                "mean_hidden_value": mean_val,
            }
        )
        prompt_queue.task_done()


async def _run_extraction_async(
    model: str,
    prompts: list[list[int]],
    num_clients: int,
    layers: list[int],
    tmpdir: str,
    extra_args: dict,
) -> dict:
    engine_args = AsyncEngineArgs(
        model=model,
        enable_prefix_caching=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layers,
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
        **extra_args,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        # Warmup: run a few prompts sequentially
        for i in range(min(4, len(prompts))):
            sp = SamplingParams(max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY)
            async for _ in engine.generate(
                request_id=f"warmup-{i}",
                prompt={"prompt_token_ids": prompts[i]},
                sampling_params=sp,
            ):
                pass

        # Fill prompt queue
        prompt_queue: asyncio.Queue = asyncio.Queue()
        for idx, token_ids in enumerate(prompts):
            prompt_queue.put_nowait((idx, token_ids))
        # Sentinel per client
        for _ in range(num_clients):
            prompt_queue.put_nowait(None)

        results: list[dict] = []
        consume_pool = ThreadPoolExecutor(max_workers=num_clients)

        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _client_loop(engine, prompt_queue, consume_pool, results, i)
            )
            for i in range(num_clients)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        consume_pool.shutdown(wait=True)

        total_prompt_tokens = sum(r["num_prompt_tokens"] for r in results)
        num_prompts = len(results)
        mean_hidden = sum(r["mean_hidden_value"] for r in results) / max(
            len(results), 1
        )

        return {
            "mode": "extract",
            "elapsed_s": elapsed,
            "num_prompts": num_prompts,
            "total_prompt_tokens": total_prompt_tokens,
            "tokens_per_s": total_prompt_tokens / elapsed,
            "prompts_per_s": num_prompts / elapsed,
            "mean_hidden_value": mean_hidden,
        }
    finally:
        engine.shutdown()


def run_extraction(
    model: str,
    prompts: list[list[int]],
    num_clients: int,
    layers: list[int],
    extra_args: dict,
) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        return asyncio.run(
            _run_extraction_async(
                model,
                prompts,
                num_clients,
                layers,
                tmpdir,
                extra_args,
            )
        )


def print_results(results: dict):
    mode = results["mode"]
    print(f"\n{'=' * 60}")
    print(f"  {mode.upper()} RESULTS")
    print(f"{'=' * 60}")
    print(f"  Prompts:             {results['num_prompts']}")
    print(f"  Total prompt tokens: {results['total_prompt_tokens']:,}")
    print(f"  Wall time:           {results['elapsed_s']:.2f}s")
    print(f"  Tokens/s:            {results['tokens_per_s']:,.0f}")
    print(f"  Prompts/s:           {results['prompts_per_s']:.2f}")
    if mode == "extract":
        print(f"  Mean hidden value:   {results['mean_hidden_value']:.6f}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hidden state extraction throughput"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--load-format", type=str, default=None)
    args = parser.parse_args()

    extra_args = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    if args.max_model_len is not None:
        extra_args["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        extra_args["enforce_eager"] = True
    if args.load_format is not None:
        extra_args["load_format"] = args.load_format

    # Get vocab size from HF config without loading the full model
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = hf_config.vocab_size
    prompts = make_random_prompts(args.num_prompts, args.prompt_len, vocab_size)
    print(
        f"Generated {args.num_prompts} prompts, "
        f"{args.prompt_len} tokens each (vocab {vocab_size})"
    )

    if not args.skip_baseline:
        baseline = run_baseline(args.model, prompts, extra_args)
        print_results(baseline)

    if not args.skip_extract:
        extract = run_extraction(
            args.model, prompts, args.num_clients, args.layers, extra_args
        )
        print_results(extract)

    if not args.skip_baseline and not args.skip_extract:
        overhead = (1 - extract["tokens_per_s"] / baseline["tokens_per_s"]) * 100
        print(f"Extraction overhead: {overhead:+.1f}%")


if __name__ == "__main__":
    main()

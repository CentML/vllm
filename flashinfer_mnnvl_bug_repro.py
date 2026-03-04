"""
Minimal reproduction for mnnvl allreduce_fusion deadlock with
kARResidualRMSNorm pattern.

Run with:
  torchrun --nproc_per_node=4 flashinfer_mnnvl_bug_repro.py
  torchrun --nproc_per_node=4 flashinfer_mnnvl_bug_repro.py --backend trtllm
"""
import argparse
import itertools

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)


def run_test(
    rank, world_size, device, flashinfer_comm, workspace, pattern,
    hidden_dim, num_tokens, num_calls, use_oneshot, norm_out_none,
    interleave_matmul, reuse_tensors, desc,
):
    """Run a single test configuration. Returns True if completed."""
    dtype = torch.bfloat16
    eps = 1e-5

    log(rank, f"--- {desc} ---")

    # Pre-allocate tensors if reusing
    if reuse_tensors:
        allreduce_in = torch.randn(num_tokens, hidden_dim, dtype=dtype,
                                   device=device)
        residual = torch.randn(num_tokens, hidden_dim, dtype=dtype,
                               device=device)
    rms_gamma = torch.ones(hidden_dim, dtype=dtype, device=device)

    # For interleaved matmuls
    if interleave_matmul:
        weight = torch.randn(hidden_dim, hidden_dim // world_size,
                             dtype=dtype, device=device)

    for i in range(num_calls):
        if not reuse_tensors:
            allreduce_in = torch.randn(num_tokens, hidden_dim, dtype=dtype,
                                       device=device)
            residual = torch.randn(num_tokens, hidden_dim, dtype=dtype,
                                   device=device)

        # Interleave a matmul between allreduce calls (simulates
        # RowParallelLinear output feeding into allreduce)
        if interleave_matmul:
            x = torch.randn(num_tokens, hidden_dim // world_size,
                             dtype=dtype, device=device)
            allreduce_in = x @ weight.t()

        log(rank, f"  call #{i+1}/{num_calls}: syncing...")
        torch.cuda.synchronize()

        if norm_out_none:
            # vllm's common pattern: norm_out=None means in-place
            # (norm output written to allreduce_in, residual updated in-place)
            norm_out = None
            residual_out = residual
        else:
            norm_out = torch.empty_like(allreduce_in)
            residual_out = allreduce_in

        log(rank, f"  call #{i+1}/{num_calls}: calling allreduce_fusion...")
        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=pattern,
            launch_with_pdl=True,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out if norm_out is not None else allreduce_in,
            residual_in=residual,
            rms_gamma=rms_gamma,
            rms_eps=eps,
            use_oneshot=use_oneshot,
            fp32_acc=True,
        )
        log(rank, f"  call #{i+1}/{num_calls}: returned")

    log(rank, f"  final sync...")
    torch.cuda.synchronize()
    log(rank, f"  PASSED: {desc}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="mnnvl",
                        choices=["mnnvl", "trtllm"])
    parser.add_argument("--hidden-dim", type=int, default=2688)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-calls", type=int, default=6)
    parser.add_argument("--test", type=str, default="all",
                        help="Run a specific test: all, or a number 1-8")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    comm_backend = TorchDistBackend(group=dist.group.WORLD)
    workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend=args.backend,
        world_size=world_size,
        rank=rank,
        max_token_num=1024,
        hidden_dim=args.hidden_dim,
        dtype=torch.bfloat16,
        comm_backend=comm_backend,
    )
    log(rank, f"Workspace: backend={args.backend}, world_size={world_size}")

    pattern = flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm

    common = dict(
        rank=rank, world_size=world_size, device=device,
        flashinfer_comm=flashinfer_comm, workspace=workspace,
        pattern=pattern, hidden_dim=args.hidden_dim,
        num_tokens=args.num_tokens, num_calls=args.num_calls,
    )

    # Test matrix: vary the parameters that differ between standalone and vllm
    tests = [
        # (use_oneshot, norm_out_none, interleave_matmul, reuse_tensors)
        (True,  False, False, False, "oneshot, separate norm_out, fresh tensors"),
        (False, False, False, False, "non-oneshot, separate norm_out, fresh tensors"),
        (True,  True,  False, False, "oneshot, norm_out=None, fresh tensors"),
        (False, True,  False, False, "non-oneshot, norm_out=None, fresh tensors"),
        (True,  True,  True,  False, "oneshot, norm_out=None, matmul, fresh"),
        (False, True,  True,  False, "non-oneshot, norm_out=None, matmul, fresh"),
        (True,  True,  True,  True,  "oneshot, norm_out=None, matmul, reuse"),
        (False, True,  True,  True,  "non-oneshot, norm_out=None, matmul, reuse"),
    ]

    if args.test != "all":
        idx = int(args.test) - 1
        tests = [tests[idx]]

    for i, (oneshot, norm_none, matmul, reuse, desc) in enumerate(tests):
        test_num = i + 1
        try:
            run_test(
                **common,
                use_oneshot=oneshot,
                norm_out_none=norm_none,
                interleave_matmul=matmul,
                reuse_tensors=reuse,
                desc=f"Test {test_num}: {desc}",
            )
        except Exception as e:
            log(rank, f"  FAILED: {desc} - {e}")

    workspace.destroy()
    dist.destroy_process_group()
    log(rank, "All tests done")


if __name__ == "__main__":
    main()

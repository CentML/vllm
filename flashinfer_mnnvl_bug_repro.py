"""
Minimal reproduction for mnnvl allreduce_fusion deadlock.

The fused allreduce+RMSNorm (kARResidualRMSNorm) with mnnvl backend deadlocks
after several calls when interleaved with matmul/MoE-like operations.

Run with:
  torchrun --nproc_per_node=4 flashinfer_mnnvl_bug_repro.py
  torchrun --nproc_per_node=4 flashinfer_mnnvl_bug_repro.py --backend trtllm
"""
import argparse

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)


def fused_allreduce_rmsnorm(
    flashinfer_comm, workspace, allreduce_in, residual, rms_gamma,
    eps=1e-5, use_oneshot=True,
):
    """Simulate vllm's call_trtllm_fused_allreduce_norm with
    AllReduceFusedAddRMSNormPattern (norm_out=None, with residual)."""
    flashinfer_comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace,
        pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        output=None,
        residual_out=residual,   # residual updated in-place
        norm_out=allreduce_in,   # norm output written to allreduce_in
        residual_in=residual,
        rms_gamma=rms_gamma,
        rms_eps=eps,
        use_oneshot=use_oneshot,
        fp32_acc=True,
    )
    return allreduce_in, residual


def moe_like_ops(x, hidden_dim, world_size, device, dtype):
    """Simulate MoE computation: router gemm + expert matmuls + combine.
    This is a surrogate for torch.ops.vllm.moe_forward_shared."""
    num_experts = 128
    # Router
    router_w = torch.randn(num_experts, hidden_dim, dtype=dtype, device=device)
    router_logits = x @ router_w.t()  # [tokens, num_experts]

    # Simulate expert computation (just matmuls)
    intermediate = 2176
    up_w = torch.randn(hidden_dim, intermediate // world_size,
                       dtype=dtype, device=device)
    down_w = torch.randn(intermediate // world_size, hidden_dim,
                         dtype=dtype, device=device)
    h = x @ up_w
    h = torch.nn.functional.relu(h) ** 2  # relu2 activation
    result = h @ down_w

    # Scale and add shared expert
    result *= 2.5
    result += x  # shared expert output
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="mnnvl",
                        choices=["mnnvl", "trtllm"])
    parser.add_argument("--hidden-dim", type=int, default=2688)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-submods", type=int, default=6,
                        help="Number of submod iterations (each has 2 allreduces)")
    parser.add_argument("--skip-moe", action="store_true",
                        help="Skip MoE-like ops between allreduces")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    hidden_dim = args.hidden_dim
    num_tokens = args.num_tokens
    dtype = torch.bfloat16

    # Determine use_oneshot based on tensor size (matching vllm logic)
    tensor_size = num_tokens * hidden_dim * 2  # bf16 = 2 bytes
    max_oneshot_mb = {2: 32, 4: 2, 8: 0.5}  # Hopper defaults
    max_oneshot = max_oneshot_mb.get(world_size, 2) * 1024 * 1024
    use_oneshot = tensor_size <= max_oneshot
    log(rank, f"tensor_size={tensor_size}, max_oneshot={max_oneshot}, "
              f"use_oneshot={use_oneshot}")

    comm_backend = TorchDistBackend(group=dist.group.WORLD)
    workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend=args.backend,
        world_size=world_size,
        rank=rank,
        max_token_num=1024,
        hidden_dim=hidden_dim,
        dtype=dtype,
        comm_backend=comm_backend,
    )
    log(rank, f"Workspace: backend={args.backend}")

    rms_gamma = torch.ones(hidden_dim, dtype=dtype, device=device)
    ar_count = 0

    # Simulate the submod execution pattern from Nemotron:
    # Each submod: mamba_out_proj → allreduce → FusedAddRMSNorm →
    #              MoE → allreduce → FusedAddRMSNorm → mamba_in_proj
    for submod in range(args.num_submods):
        log(rank, f"submod {submod}: start")

        # 1. Mamba out_proj output (simulate RowParallelLinear)
        mamba_w = torch.randn(hidden_dim, hidden_dim // world_size,
                              dtype=dtype, device=device)
        mamba_in = torch.randn(num_tokens, hidden_dim // world_size,
                               dtype=dtype, device=device)
        allreduce_in = mamba_in @ mamba_w.t()
        residual = torch.randn(num_tokens, hidden_dim, dtype=dtype,
                               device=device)

        # 2. First allreduce + FusedAddRMSNorm
        ar_count += 1
        log(rank, f"  AR #{ar_count}: pre-sync...")
        torch.cuda.synchronize()
        log(rank, f"  AR #{ar_count}: calling fused allreduce+rmsnorm")
        hidden, residual = fused_allreduce_rmsnorm(
            flashinfer_comm, workspace, allreduce_in, residual, rms_gamma,
            use_oneshot=use_oneshot,
        )
        log(rank, f"  AR #{ar_count}: returned")

        # 3. MoE computation (between the two allreduces)
        if not args.skip_moe:
            hidden = moe_like_ops(hidden, hidden_dim, world_size, device, dtype)

        # 4. Second allreduce + FusedAddRMSNorm
        ar_count += 1
        allreduce_in = hidden
        log(rank, f"  AR #{ar_count}: pre-sync...")
        torch.cuda.synchronize()
        log(rank, f"  AR #{ar_count}: calling fused allreduce+rmsnorm")
        hidden, residual = fused_allreduce_rmsnorm(
            flashinfer_comm, workspace, allreduce_in, residual, rms_gamma,
            use_oneshot=use_oneshot,
        )
        log(rank, f"  AR #{ar_count}: returned")

        # 5. Mamba in_proj (next layer setup)
        in_proj_w = torch.randn(hidden_dim // world_size + hidden_dim // world_size,
                                hidden_dim, dtype=dtype, device=device)
        projected = hidden @ in_proj_w.t()

        log(rank, f"submod {submod}: done")

    log(rank, "Final sync...")
    torch.cuda.synchronize()
    log(rank, f"All {ar_count} fused allreduces completed!")

    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

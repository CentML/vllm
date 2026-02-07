# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from importlib.util import find_spec

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

fi_ar_available = False
try:
    import flashinfer.comm as flashinfer_comm  # type: ignore[no-redef]
    from flashinfer.comm.mnnvl import TorchDistBackend  # type: ignore[import-not-found, no-redef]

    fi_ar_available = hasattr(flashinfer_comm, "allreduce_fusion")
except ImportError:
    pass

_fi_ar_workspace = None

def get_fi_ar_workspace():
    return _fi_ar_workspace

def initialize_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
) -> None:
    """
    Initialize the workspace if not already initialized.
    """
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        return

    assert fi_ar_available, "FlashInfer All Reduce is not available."

    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    comm_backend = TorchDistBackend(group=group)
    _fi_ar_workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        comm_backend=comm_backend,
    )

    logger.debug(
        "Initialized shared FlashInfer All Reduce workspace: backend=%s, "
        "world_size=%d, rank=%d, max_token_num=%d",
        backend, world_size, rank, max_token_num
    )

    return _fi_ar_workspace


def destroy_fi_ar_workspace():
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        _fi_ar_workspace.destroy()
        _fi_ar_workspace = None


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_token_num: int = 1024,
    ):
        self.disabled = True

        if not fi_ar_available:
            logger.info(
                "FlashInfer All Reduce is disabled because flashinfer is not available"
            )
            return

        if not current_platform.is_cuda():
            logger.info(
                "FlashInfer All Reduce is disabled because it requires CUDA platform"
            )
            return

        if not current_platform.has_device_capability((10, 0)):
            logger.info(
                "FlashInfer All Reduce is disabled because "
                "it requires Blackwell architecture (compute capability 10.0)"
            )
            return

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = device

        if self.world_size == 1:
            return

        self.max_token_num = max_token_num
        self.disabled = False

    def _ensure_workspace(self, hidden_dim: int) -> bool:
        """Ensure the all reduce workspace is initialized."""
        if get_fi_ar_workspace() is not None:
            return True
        try:
            initialize_fi_ar_workspace(
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=self.max_token_num,
                hidden_dim=hidden_dim,
                dtype=torch.bfloat16,
                group=self.group,
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to initialize FlashInfer All Reduce workspace: %s. "
                "FlashInfer All Reduce will be disabled.",
                e,
            )
            self.disabled = True
            return False

    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda:
            return False

        if not input_tensor.is_contiguous():
            return False

        if len(input_tensor.shape) != 2:
            return False

        num_tokens, hidden_dim = input_tensor.shape

        if num_tokens > self.max_token_num:
            return False

        if not self._ensure_workspace():
            return False

        return True

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        workspace = get_fi_ar_workspace()
        assert workspace is not None, "FlashInfer All Reduce workspace not initialized. "
        return flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=False,
        )

    def destroy(self):
        if not self.disabled:
            destroy_fi_ar_workspace()

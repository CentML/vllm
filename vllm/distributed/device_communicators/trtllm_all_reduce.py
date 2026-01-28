# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    from flashinfer.comm import (
        MNNVLAllReduceFusionWorkspace,
        MNNVLAllreduceFusionStrategy,
        trtllm_mnnvl_allreduce,
    )
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import CommBackend

    fi_mnnvl_available = True

    class TorchDistributedCommBackend(CommBackend):
        """
        Use torch distributed instead of MPI to set up
        flashinfer MNNVL workspaces during initialization.
        """

        def __init__(self, group: ProcessGroup):
            self._group = group

        def Get_rank(self) -> int:
            return self._group.rank()

        def Get_size(self) -> int:
            return self._group.size()

        def allgather(self, data: int):
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered

        def bcast(self, data, root: int = 0):
            """
            Broadcast a picklable Python object from `root` to all ranks.
            Uses torch.distributed.broadcast_object_list under the hood.

            Returns the broadcasted object on every rank.
            """
            obj_list = [data]
            # broadcast_object_list mutates obj_list in-place
            dist.broadcast_object_list(obj_list, src=root, group=self._group)
            return obj_list[0]

        def barrier(self):
            """
            Synchronize all ranks in this communicator.
            """
            dist.barrier(group=self._group)

        def Split(self, color: int, key: int) -> "TorchDistributedCommBackend":
            # No need to split, we already use the proper group
            return self

except ImportError:
    fi_mnnvl_available = False


def get_physical_device_ids(
    group: ProcessGroup, device: torch.device
) -> list[int]:
    """Get physical device IDs for all ranks in the group."""
    import vllm.envs as envs
    from vllm.utils.torch_utils import cuda_device_count_stateless

    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices:
        device_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        device_ids = list(range(cuda_device_count_stateless()))

    physical_device_id = device_ids[device.index]
    world_size = dist.get_world_size(group)
    tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
    gather_list = [
        torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
    ]
    dist.all_gather(gather_list, tensor, group=group)
    return [t.item() for t in gather_list]


class TRTLLMAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        hidden_dim: int | None = None,
        max_token_num: int = 1024,
    ):
        self.disabled = True
        self.workspace = None

        if not fi_mnnvl_available:
            logger.info(
                "MNNVL all-reduce is disabled because flashinfer is not available"
            )
            return

        if not current_platform.is_cuda():
            logger.info(
                "MNNVL all-reduce is disabled because it requires CUDA platform"
            )
            return

        device_capability = current_platform.get_device_capability()
        if device_capability is None or not device_capability.to_int() >= 100:
            logger.info(
                "MNNVL all-reduce is disabled because "
                "it requires Blackwell architecture (compute capability 10.0)"
            )
            return

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if self.world_size == 1:
            return

        if not self._is_mnnvl():
            logger.info(
                "MNNVL all-reduce is disabled because "
                "it is only used under Multi-Node NVLINK setup"
            )
            return

        # Store parameters for lazy initialization
        self.hidden_dim = hidden_dim
        self.max_token_num = max_token_num

        # If hidden_dim is provided, initialize now
        if hidden_dim is not None:
            self._initialize_workspace(hidden_dim, max_token_num)
            self.disabled = False
        else:
            # Mark as enabled but not yet initialized
            # Will be initialized on first use
            self.disabled = False
            logger.info(
                "MNNVL all-reduce initialized with lazy workspace creation. "
                "Workspace will be created on first use."
            )

    def _is_mnnvl(self) -> bool:
        """
        Check if current environment is a Multi-Node NVLINK setup.
        """
        all_on_same_node = all(in_the_same_node_as(self.group, source_rank=0))
        # Do not use MNNVL all-reduce for single-node setup.
        if all_on_same_node:
            return False

        # Check if the GPUs are fully connected by NVLINK
        physical_device_ids = get_physical_device_ids(self.group, self.device)
        fully_connected = current_platform.is_fully_connected(physical_device_ids)

        return fully_connected

    def _initialize_workspace(self, hidden_dim: int, max_token_num: int):
        gpus_per_node = sum(in_the_same_node_as(self.group, source_rank=0))

        mapping = Mapping(
            world_size=self.world_size,
            tp_size=self.world_size,
            rank=self.rank,
            gpus_per_node=gpus_per_node,
        )

        # Create custom communicator backend for flashinfer
        comm = TorchDistributedCommBackend(self.group)

        # Create the MNNVL all-reduce workspace
        self.workspace = MNNVLAllReduceFusionWorkspace(
            mapping=mapping,
            max_num_tokens=max_token_num,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            comm_backend=comm,
        )

    def should_use_mnnvl_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda:
            return False

        if input_tensor.dtype != torch.bfloat16:
            return False

        if not input_tensor.is_contiguous():
            return False

        if len(input_tensor.shape) != 2:
            return False

        num_tokens, hidden_dim = input_tensor.shape

        # If workspace not initialized, try to initialize it now
        if self.workspace is None:
            try:
                self._initialize_workspace(hidden_dim, self.max_token_num)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize MNNVL workspace lazily: {e}. "
                    "MNNVL all-reduce will be disabled."
                )
                self.disabled = True
                return False

        # Check if workspace is sufficient for the problem size
        if not self.workspace.is_buffer_size_sufficient(
            self.world_size,
            num_tokens,
            hidden_dim,
            input_tensor.dtype,
            MNNVLAllreduceFusionStrategy.AUTO,
        ):
            return False

        return True

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.workspace is None:
            raise RuntimeError(
                "MNNVL workspace not initialized. "
                "Call should_use_mnnvl_ar first to check availability."
            )

        output_tensor = trtllm_mnnvl_allreduce(
            input=input_tensor,
            workspace=self.workspace,
            launch_with_pdl=False,
            strategy=MNNVLAllreduceFusionStrategy.AUTO,
        )

        return output_tensor

    def destroy(self):
        if not self.disabled:
            try:
                self.workspace.destroy()
                del self.workspace
            except Exception as e:
                logger.warning(f"Error during MNNVL cleanup: {e}")

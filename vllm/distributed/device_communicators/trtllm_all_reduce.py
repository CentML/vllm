# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import in_the_same_node_as

logger = init_logger(__name__)

try:
    import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
    from flashinfer.comm.mapping import Mapping
    fi_trtllm_available = True
except ImportError:
    fi_trtllm_available = False


class TRTLLMAllReduce:

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device]):

        self.disabled = True
        
        if not fi_trtllm_available:
            logger.info("TRTLLM all-reduce is disabled because "
                       "flashinfer is not available")
            return
            
        if not current_platform.is_cuda():
            logger.info("TRTLLM all-reduce is disabled because "
                       "it requires CUDA platform")
            return
            
        if not current_platform.is_device_capability(100):
            logger.info("TRTLLM all-reduce is disabled because "
                       "it requires Blackwell architecture (compute capability 10.0)")
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
            logger.info("TRTLLM all-reduce is disabled because "
                       "it is only used under Multi-Node NVLINK setup")
            return
        logger.info("Using TRTLLM all-reduce because "
                    "it is a Multi-Node NVLINK setup")
        
        self._initialize_workspace()
        self.disabled = False
    
    def _is_mnnvl(self) -> bool:
        """
        Check if current environment is a Multi-Node NVLINK setup.
        """
        all_on_same_node = all(in_the_same_node_as(self.group, source_rank=0))
        # Do not use TRTLLM all-reduce for single-node setup.
        if all_on_same_node:
            return False

        from vllm.distributed.device_communicators.custom_all_reduce import (
            get_physical_device_ids)
        
        # Check if the GPUs are fully connected by NVLINK
        physical_device_ids = get_physical_device_ids(self.group, self.device)
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        
        return fully_connected
    
    
    def _initialize_workspace(self):

        gpus_per_node = sum(in_the_same_node_as(self.group, source_rank=0))
        
        mapping = Mapping(
            world_size=self.world_size,
            tp_size=self.world_size,
            rank=self.rank,
            gpus_per_node=gpus_per_node,
        )
        
        self.mcast_buffer_mnnvl, self.buffer_flags_mnnvl, self.max_num_elements_mnnvl = (
            trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, torch.bfloat16, group=self.group)
        )
        
    
    def should_use_trtllm_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False
            
        if not input_tensor.is_cuda:
            return False
            
        if input_tensor.dtype != torch.bfloat16:
            return False
            
        if not input_tensor.is_contiguous():
            return False
            
        tensor_numel = input_tensor.numel()
        if tensor_numel > self.max_num_elements_mnnvl:
            return False
            
            
        hidden_size = input_tensor.shape[-1]
        if self.max_num_elements_mnnvl % hidden_size != 0:
            return False
            
        return True
    
    def all_reduce(self, 
                   input_tensor: torch.Tensor) -> torch.Tensor:

        output_tensor = torch.empty_like(input_tensor)
        
        multicast_ptr = self.mcast_buffer_mnnvl.get_multicast_ptr()
        buffer_ptrs_dev = self.mcast_buffer_mnnvl.get_buffer_ptrs_dev()
        
        hidden_size = input_tensor.shape[-1]
        buffer_M = self.max_num_elements_mnnvl // hidden_size
        
        trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
            inp=input_tensor,
            multicast_buffer_ptr=multicast_ptr,
            buffer_ptrs_dev=buffer_ptrs_dev,
            buffer_M=buffer_M,
            buffer_flags_mnnvl=self.buffer_flags_mnnvl,
            nranks=self.world_size,
            rank=self.rank,
            wait_for_results=True,
            launch_with_pdl=False,
            out=output_tensor,
        )
        logger.info("**************TRTLLM all-reduce**************")
        
        return output_tensor
    
    def destroy(self):
        if not self.disabled:
            try:
                del self.mcast_buffer_mnnvl
                del self.buffer_flags_mnnvl
                del self.max_num_elements_mnnvl
            except Exception as e:
                logger.warning(f"Error during TRTLLM cleanup: {e}")

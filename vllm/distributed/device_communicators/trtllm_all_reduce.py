# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    logger.info("Importing flashinfer.comm.trtllm_mnnvl_ar")
    import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
    logger.info("Imported flashinfer.comm.trtllm_mnnvl_ar")
    from flashinfer.comm.mapping import Mapping
    logger.info("Imported flashinfer.comm.mapping")
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
        logger.info("Past flashinfer is available check.")
            
        if not current_platform.is_cuda():
            logger.info("TRTLLM all-reduce is disabled because "
                       "it requires CUDA platform")
            return
        logger.info("Past CUDA platform check.")
            
        if not current_platform.is_device_capability(100):
            logger.info("TRTLLM all-reduce is disabled because "
                       "it requires Blackwell architecture (compute capability 10.0)")
            return
        logger.info("Past Blackwell architecture check.")
            
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        logger.info(f"Past world size check. world_size={self.world_size}")
        self.rank = dist.get_rank(self.group)
        logger.info(f"Past rank check. rank={self.rank}")
        
        if self.world_size == 1:
            return
            
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        logger.info(f"Past device check. device={self.device}")
        
        logger.info("initializing TRTLLM all-reduce workspace.")
        # self._initialize_workspace()
        # self.disabled = False
        # logger.info("Using TRTLLM all-reduce.")
    
    def _initialize_workspace(self):
        gpus_per_node = 4
        
        mapping = Mapping(
            world_size=self.world_size,
            tp_size=self.world_size,
            rank=self.rank,
            gpus_per_node=gpus_per_node,
        )
        
        self.mcast_buffer_mnnvl, self.buffer_flags_mnnvl, self.max_num_elements_mnnvl = (
            trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
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
            output=output_tensor,
        )
        
        return output_tensor
    
    def destroy(self):
        if not self.disabled:
            try:
                del self.mcast_buffer_mnnvl
                del self.buffer_flags_mnnvl
                del self.max_num_elements_mnnvl
            except Exception as e:
                logger.warning(f"Error during TRTLLM cleanup: {e}")

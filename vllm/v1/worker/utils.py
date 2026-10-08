# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.mamba_mixer2 import share_replayssm_ring_trackers
from vllm.model_executor.layers.utils import warmup_rocm_skinny_gemm_workspaces
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


def raise_if_nan_logits(num_nans_in_logits: Mapping[str, int]) -> None:
    if not any(num_nans_in_logits.values()):
        return

    corrupted_requests = {
        req_id: num_nans
        for req_id, num_nans in num_nans_in_logits.items()
        if num_nans > 0
    }
    raise RuntimeError(f"NaNs detected in logits: {corrupted_requests}")


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  Layer-compact
    layouts have one segment per layer buffer; dimensions physically outside
    the block dim (separate head groups under LHBNC) and virtual block splits
    each get their own segment.

    Segments may have different block strides and page sizes (e.g. packed
    KV views or models with multiple KV cache groups like MLA + DSA
    indexer). Each segment's block stride determines where a logical block
    begins, while its page size determines how many elements are cleared.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped directly onto a 3-D grid as
    (block_index, seg_index, chunk_index).
    """
    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(
        ptr + block_offset + cols,
        tl.zeros([BLOCK_SIZE], dtype=tl.int32),
        mask=cols < page_size_el,
    )


class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        """Precompute the absolute-address table for the Triton zeroing kernel.

        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Per-layer views are standardized ``[B, H, N, C]`` with blocks at dim 0; dims
        physically outside B (separate head groups under LHBNC) each get their own
        segment. A segment's page spans everything inside its block, so under BHLNC it
        also covers the block's other layers -- safe, since block IDs are global pool
        indices and a newly allocated block owns its whole tile.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each virtual block is represented as an independent segment so its
        physical block stride and zeroed page span remain independent.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        self.device = device
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None
        ) = None

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        # Overlaid layers (packed layouts) share a base address but may have
        # different page sizes; keep the widest span per address so newly
        # allocated blocks are fully zeroed for every overlaying group.
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            assert spec.block_size % kernel_bs == 0
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                if kv.device.type != self.device.type:
                    continue
                dp = kv.data_ptr()

                assert kv.shape[0] % num_blocks == 0, (
                    f"{layer_name}: {kv.shape[0]} kernel blocks is not a "
                    f"multiple of {num_blocks} logical blocks"
                )
                ratio = kv.shape[0] // num_blocks

                el = kv.element_size()
                block_stride_bytes = kv.stride(0) * el
                assert block_stride_bytes % 4 == 0
                assert kv.shape[0] % ratio == 0
                outer_dims = [
                    d
                    for d in range(1, kv.ndim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                inner_dims = [d for d in range(1, kv.ndim) if d not in outer_dims]
                kernel_page_bytes = el + sum(
                    (kv.shape[d] - 1) * kv.stride(d) * el for d in inner_dims
                )
                assert kernel_page_bytes % 4 == 0
                logical_block_stride_bytes = block_stride_bytes * ratio
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    assert (dp + off_bytes) % 4 == 0
                    for virtual_index in range(ratio):
                        addr = dp + off_bytes + virtual_index * block_stride_bytes
                        if (idx := seen_ptrs.get(addr)) is not None:
                            assert (
                                seg_block_strides[idx]
                                == logical_block_stride_bytes // 4
                            )
                            seg_page_sizes[idx] = max(
                                seg_page_sizes[idx], kernel_page_bytes // 4
                            )
                            continue
                        seen_ptrs[addr] = len(seg_addrs)
                        seg_addrs.append(addr)
                        seg_block_strides.append(logical_block_stride_bytes // 4)
                        seg_page_sizes.append(kernel_page_bytes // 4)

        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            max_chunks,
            blk_size,
            n_segs,
        ) = self._meta
        n_blocks = len(block_ids)
        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        grid = (n_blocks, n_segs, max_chunks)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            idx,
            BLOCK_SIZE=blk_size,
        )

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        if kernel_block_size is None:
            kv_cache_spec_builder = self.kv_cache_spec
        elif (
            isinstance(self.kv_cache_spec, MLAAttentionSpec)
            and self.kv_cache_spec.storage_block_size is not None
        ):
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                self.kv_cache_spec.storage_block_size
            )
        else:
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                kernel_block_size
            )
        builder_cls = self.backend.get_builder_cls()
        builder_kwargs = {}
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks, self.kv_cache_spec.block_size, kernel_block_size
            )
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]
        if kernel_block_size is not None:
            for builder in self.metadata_builders:
                builder.set_kernel_block_size(kernel_block_size)

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]

    @property
    def supports_draft_decode_metadata_update(self) -> bool:
        return self.get_metadata_builder().supports_draft_decode_metadata_update

    def update_draft_decode_metadata(
        self,
        attn_metadata: Mapping[str, Any],
    ) -> None:
        metadata = attn_metadata[self.layer_names[0]]
        self.get_metadata_builder().update_draft_decode_metadata(metadata)


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.

    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # MultipleOf constraints also accept the manager size if they accept a divisor.
    # Any remaining candidate must therefore be an explicit size from a backend.
    candidates = {
        size
        for backend in backends
        for size in backend.get_supported_kernel_block_sizes()
        if isinstance(size, int) and kv_manager_block_size % size == 0
    }

    for size in sorted(candidates, reverse=True):
        if block_size_is_supported(backends, size):
            return size
    raise ValueError(f"No common block size for {kv_manager_block_size}.")


def allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate the KV cache and view it as ``[B, H, N, C]`` per layer.

    Every KVCacheTensor places its layers in the same backing allocation: layer ``l`` of
    block ``b`` starts at ``offset + l * layer_stride + b * block_stride``. Cache
    groups overlay each other, so tensors may address the same bytes.
    """
    if not kv_cache_config.kv_cache_tensors:
        return {}

    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
    raw_size = sizes.pop()
    # wvSplitKrc's process-lifetime static workspaces (csrc/rocm/skinny_gemms.cu)
    # are created lazily on the first qualifying GEMM. Force that now, before
    # the giant backing allocation below: if one landed in this segment's
    # rounding tail it would pin the whole segment at engine shutdown.
    if current_platform.is_rocm():
        warmup_rocm_skinny_gemm_workspaces(device)
        # Pad to the page granularity MoRIIO needs to register the shared
        # backing as a single RDMA memory region. Other platforms keep the
        # exact-size allocation: NIXL and SimpleCPUOffload rely on
        # storage.nbytes() matching the logical KV size (see #53974).
        page_size = 4096
        buf_size = ((raw_size + page_size - 1) // page_size) * page_size
    else:
        buf_size = raw_size
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[layer_name]

        if not spec.has_layer_views:
            kv_caches.update((name, buf) for name in tensor.layers)
            continue

        num_blocks = kv_cache_config.num_blocks_of(tensor)
        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]
        if isinstance(spec, MLAAttentionSpec) and spec.storage_block_size is not None:
            kernel_block_size = spec.storage_block_size

        views = create_kv_cache_views(
            buf,
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.

    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if not kv_cache_spec.has_layer_views:
            kernel_block_sizes.append(kv_cache_spec.block_size)
        elif isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
        runner_only_attn_layers: Attention layers handled by the runner only,
            which are excluded from the KV cache groups.

    """
    if not shared_kv_cache_layers:
        return

    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        num_attn_module: Number of attention modules per layer entry.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
        kv_cache_groups: The KV cache groups of the model, used to resolve
            layers that share a KV cache.

    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            current_platform.check_runner_kv_caches_multi_layer()
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    bind_kv_cache_to_layers(
        kv_caches, forward_context, num_attn_module, kv_cache_groups
    )


def bind_kv_cache_to_layers(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Bind layer caches and share ReplaySSM trackers in model-layer order."""
    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)

    ordered_layer_names = sorted(
        kv_caches, key=lambda name: extract_layer_index(name, num_attn_module)
    )
    share_replayssm_ring_trackers(ordered_layer_names, forward_context, kv_cache_groups)


def clear_layer_kv_caches(layers: Iterable[Any]) -> None:
    """Detach the KV/state cache tensors installed by bind_kv_cache().

    The model object can outlive the runner (e.g. LLMEngine's finalizer keeps
    it reachable until engine deletion), so dropping the runner's references
    alone does not release the KV cache memory on teardown paths.
    """
    for layer in layers:
        if not hasattr(layer, "kv_cache"):
            continue
        kv_cache = layer.kv_cache
        layer.kv_cache = torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
        # Clean up quantized KV cache scale views
        # (int8_per_token_head, fp8_per_token_head)
        if hasattr(layer, "impl"):
            if hasattr(layer.impl, "_k_scale_cache"):
                layer.impl._k_scale_cache = None
            if hasattr(layer.impl, "_v_scale_cache"):
                layer.impl._v_scale_cache = None


# Copy-on-write block copies for partial prefix hits: every distinct storage's
# scheduler-block rows src -> dst in ONE launch, 16-byte accesses, no gather
# temp. Table rows are (row start, row length, row stride) in 16-byte units
# relative to an anchor storage; scaling by 4 int32 elements lets Triton prove
# 16-byte alignment, so loads/stores vectorize to 128 bits.
_COW_COPY_BLOCK = 1024
_COW_COPY_ITERS = 4


@triton.jit(do_not_specialize=["num_pairs"])
def _copy_kv_cache_block_rows_kernel(
    anchor_ptr,  # int32 view of one storage; all addresses are relative to it
    table_ptr,  # int64 [num_entries, 3]: start16, len16, stride16
    idx_ptr,  # int64 [2, num_pairs]: src blocks, then dst blocks
    num_pairs,
    BLOCK: tl.constexpr,
    ITERS: tl.constexpr,
):
    pair = tl.program_id(0)
    chunk = tl.program_id(1)
    entry = tl.program_id(2)
    start16 = tl.load(table_ptr + entry * 3)
    row_elems = tl.load(table_ptr + entry * 3 + 1) * 4
    stride16 = tl.load(table_ptr + entry * 3 + 2)
    start = chunk.to(tl.int64) * (BLOCK * ITERS)
    if start < row_elems:
        src = tl.load(idx_ptr + pair)
        dst = tl.load(idx_ptr + num_pairs + pair)
        src_row = anchor_ptr + (start16 + src * stride16) * 4
        dst_row = anchor_ptr + (start16 + dst * stride16) * 4
        for i in tl.static_range(ITERS):
            offs = start + i * BLOCK + tl.arange(0, BLOCK)
            mask = offs < row_elems
            vals = tl.load(src_row + offs, mask=mask)
            tl.store(dst_row + offs, vals, mask=mask)


@triton.jit(do_not_specialize=["num_pairs"])
def _copy_kv_cache_block_rows_per_storage_kernel(
    base_ptr,
    idx_ptr,  # contiguous int64 [2, num_pairs]: src blocks, then dst blocks
    num_pairs,
    row_elems,
    row_stride,
    BLOCK: tl.constexpr,
    ITERS: tl.constexpr,
):
    pair = tl.program_id(0)
    chunk = tl.program_id(1)
    src = tl.load(idx_ptr + pair).to(tl.int64)
    dst = tl.load(idx_ptr + num_pairs + pair).to(tl.int64)
    src_row = base_ptr + src * row_stride
    dst_row = base_ptr + dst * row_stride
    start = chunk.to(tl.int64) * (BLOCK * ITERS)
    for i in tl.static_range(ITERS):
        offs = start + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < row_elems
        vals = tl.load(src_row + offs, mask=mask)
        tl.store(dst_row + offs, vals, mask=mask)


def _fused_copy_kv_cache_block_rows(
    blocks: torch.Tensor, indices: torch.Tensor
) -> bool:
    """Copy dense, disjoint block rows without a gather temporary.

    Unlike the one-launch plan, each row only needs int32 alignment. The caller
    must ensure unique destinations disjoint from the sources; different views
    are copied sequentially, so they need not commute with one another.
    """
    if not blocks.is_cuda:
        return False
    try:
        # A reshape/flatten may silently copy and discard the destination writes.
        rows = blocks.view(blocks.shape[0], -1)
    except RuntimeError:
        return False
    row_bytes = rows.shape[1] * rows.element_size()
    stride_bytes = rows.stride(0) * rows.element_size()
    if (
        rows.stride(1) != 1
        or row_bytes == 0
        or row_bytes > stride_bytes
        or (rows.data_ptr() | row_bytes | stride_bytes) % 4
    ):
        return False
    storage = rows.untyped_storage()
    offset = rows.storage_offset() * rows.element_size() // 4
    base = torch.empty(0, dtype=torch.int32, device=rows.device)
    base.set_(storage, offset, (storage.nbytes() // 4 - offset,))
    row_elems = row_bytes // 4
    num_pairs = indices.numel() // 2
    _copy_kv_cache_block_rows_per_storage_kernel[
        (num_pairs, triton.cdiv(row_elems, _COW_COPY_BLOCK * _COW_COPY_ITERS))
    ](
        base,
        indices,
        num_pairs,
        row_elems,
        stride_bytes // 4,
        BLOCK=_COW_COPY_BLOCK,
        ITERS=_COW_COPY_ITERS,
        num_warps=4,
    )
    return True


def warmup_copy_kv_cache_block_rows(device: torch.device) -> None:
    """Compile the direct copy kernels before serving."""
    if device.type != "cuda" or not (
        envs.VLLM_KV_COW_ONE_LAUNCH or envs.VLLM_FUSED_KV_BLOCK_COPY
    ):
        return
    buf = torch.zeros(64, dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    if envs.VLLM_KV_COW_ONE_LAUNCH:
        table = torch.tensor([[0, 1, 1]], dtype=torch.int64, device=device)
        _copy_kv_cache_block_rows_kernel[(1, 1, 1)](
            buf,
            table,
            indices,
            1,
            BLOCK=_COW_COPY_BLOCK,
            ITERS=_COW_COPY_ITERS,
            num_warps=4,
        )
    if envs.VLLM_FUSED_KV_BLOCK_COPY:
        # Triton specializes each row integer's divisibility by 16, and the
        # base pointer's alignment. Cover all four integer combinations at an
        # aligned base and at a 4-byte offset without allocating real KV pages.
        # Single-word rows additionally specialize the integer value 1.
        warm_rows = ((4, 4), (4, 16), (16, 20), (16, 16), (1, 1), (1, 4), (1, 16))
        for offset in (0, 1):
            for row_elems, row_stride in warm_rows:
                blocks = buf.as_strided(
                    (2, row_elems), (row_stride, 1), storage_offset=offset
                )
                _fused_copy_kv_cache_block_rows(blocks, indices)


# Row layout -> (device table, num entries, max row len16), or None when the
# layout cannot be copied in one launch. Plans hold only addresses, never the
# KV storage, so a released KV cache is freed.
_cow_copy_plans: dict[tuple, tuple[torch.Tensor, int, int] | None] = {}


def _cow_copy_rows(kv_caches: Iterable[torch.Tensor], num_blocks: int) -> tuple | None:
    """Scheduler-block rows exactly as the per-storage path copies them: one
    (storage, row address of block 0, row bytes, block stride bytes) per
    distinct whole storage or per-layer view. None if unsupported.
    """
    device = None
    rows_out: set[tuple[int, int, int, int]] = set()
    seen: set[int] = set()
    anchor = None
    for cache in kv_caches:
        if cache.data_ptr() in seen:
            continue
        seen.add(cache.data_ptr())
        if device is None:
            if not cache.is_cuda:
                return None
            device = cache.device
        elif cache.device != device:
            return None
        kernel_blocks_per_block, remainder = divmod(cache.shape[0], num_blocks)
        if remainder:
            return None
        storage = cache.untyped_storage()
        if anchor is None:
            anchor = storage
        block_stride = cache.stride(0) * cache.element_size() * kernel_blocks_per_block
        if storage.nbytes() == num_blocks * block_stride:
            rows_out.add(
                (storage.data_ptr(), storage.data_ptr(), block_stride, block_stride)
            )
            continue
        # Copying a block element by element equals copying its byte span when
        # the block's elements densely tile that span (any dim order, e.g. a
        # head-minor LBNHC page).
        block = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))[0]
        dims = sorted(
            (stride, size)
            for size, stride in zip(block.shape, block.stride())
            if size != 1
        )
        dense_stride = 1
        for stride, size in dims:
            if stride != dense_stride:
                return None
            dense_stride *= size
        rows_out.add(
            (
                storage.data_ptr(),
                cache.data_ptr(),
                block.numel() * cache.element_size(),
                block_stride,
            )
        )
    if anchor is None:
        return None
    return device, anchor, tuple(sorted(rows_out))


def _cow_copy_rows_commute(
    rows: tuple[tuple[int, int, int, int], ...], num_blocks: int
) -> bool:
    """True if, for every src/dst pair set with unique destinations disjoint
    from the sources, all row copies may run in parallel with the result of
    the sequential per-storage path. Rows of one block stride S that keep each
    block's bytes inside one S-sized slot whose index is (region start in
    num_blocks * S units) + block id cannot read bytes another copy writes
    unless both address the same block (then they write identical bytes).
    Rows of different strides must not overlap at all.
    """
    footprints: dict[tuple[int, int], tuple[int, int]] = {}
    for storage, addr, length, stride in rows:
        if (addr | length | stride) % 16 or length > stride:
            return False
        slot, offset = divmod(addr - storage, stride)
        if offset + length > stride or slot % num_blocks:
            return False
        lo = addr
        hi = addr + (num_blocks - 1) * stride + length
        key = (storage, stride)
        if key in footprints:
            old_lo, old_hi = footprints[key]
            lo, hi = min(lo, old_lo), max(hi, old_hi)
        footprints[key] = (lo, hi)
    spans = sorted(footprints.values())
    return all(prev[1] <= cur[0] for prev, cur in zip(spans, spans[1:]))


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    if not kv_cache_block_copies:
        return
    kv_caches = list(kv_caches)
    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    src_np, dst_np = indices_np[:, 0], indices_np[:, 1]
    plan = None
    # Direct row copies equal gather-then-scatter only when destinations are
    # unique and no block is both a source and a destination.
    direct_copy_safe = (
        len(np.unique(dst_np)) == len(dst_np)
        and not np.intersect1d(src_np, dst_np).size
    )
    if envs.VLLM_KV_COW_ONE_LAUNCH and direct_copy_safe:
        layout = _cow_copy_rows(kv_caches, num_blocks)
        if layout is not None:
            device, anchor_storage, rows = layout
            key = (device, anchor_storage.data_ptr(), num_blocks, rows)
            if key not in _cow_copy_plans:
                _cow_copy_plans[key] = _make_cow_copy_plan(
                    device, anchor_storage, rows, num_blocks
                )
            plan = _cow_copy_plans[key]
    if plan is None:
        _copy_kv_cache_blocks_inplace_per_storage(
            kv_caches, num_blocks, indices_np, direct_copy_safe=direct_copy_safe
        )
        return
    table, num_entries, max_len16 = plan
    anchor = torch.empty(0, dtype=torch.int32, device=device)
    anchor.set_(anchor_storage, 0, (anchor_storage.nbytes() // 4,))
    indices = async_tensor_h2d(np.concatenate([src_np, dst_np]), device=device)
    num_pairs = len(indices_np)
    grid = (
        num_pairs,
        triton.cdiv(max_len16 * 4, _COW_COPY_BLOCK * _COW_COPY_ITERS),
        num_entries,
    )
    _copy_kv_cache_block_rows_kernel[grid](
        anchor,
        table,
        indices,
        num_pairs,
        BLOCK=_COW_COPY_BLOCK,
        ITERS=_COW_COPY_ITERS,
        num_warps=4,
    )


def _make_cow_copy_plan(
    device: torch.device,
    anchor_storage: torch.UntypedStorage,
    rows: tuple[tuple[int, int, int, int], ...],
    num_blocks: int,
) -> tuple[torch.Tensor, int, int] | None:
    if not _cow_copy_rows_commute(rows, num_blocks) or anchor_storage.data_ptr() % 16:
        logger.info_once(
            "KV copy-on-write: %d cache rows are not provably disjoint or "
            "16-byte aligned; using the per-storage copy.",
            len(rows),
        )
        return None
    base = anchor_storage.data_ptr()
    table = np.array(
        [
            ((addr - base) // 16, length // 16, stride // 16)
            for _, addr, length, stride in rows
        ],
        dtype=np.int64,
    )
    max_len16 = max(length for _, _, length, _ in rows) // 16
    logger.info_once(
        "KV copy-on-write: %d cache rows of up to %d bytes in one launch.",
        len(rows),
        max_len16 * 16,
    )
    return torch.from_numpy(table).to(device), len(rows), max_len16


def _copy_kv_cache_blocks_inplace_per_storage(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    indices_np: np.ndarray,
    *,
    direct_copy_safe: bool | None = None,
) -> None:
    if not len(indices_np):
        return
    if direct_copy_safe is None:
        src_np, dst_np = indices_np[:, 0], indices_np[:, 1]
        direct_copy_safe = (
            len(np.unique(dst_np)) == len(dst_np)
            and not np.intersect1d(src_np, dst_np).size
        )
    fused_ok = envs.VLLM_FUSED_KV_BLOCK_COPY and direct_copy_safe
    # Both index rows are contiguous after one shared H2D upload. In particular,
    # the direct path needs no per-storage src/dst .contiguous() launches.
    indices_host = indices_np.T.copy()
    indices: torch.Tensor | None = None
    seen: set[tuple[torch.device, int]] = set()
    copied_storages: set[tuple[torch.device, int]] = set()
    for cache in kv_caches:
        # Layers sharing KV (cross-layer sharing) alias the same view; copy it
        # once. data_ptr distinguishes per-layer views of a shared allocation.
        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)

        if indices is None:
            indices = async_tensor_h2d(indices_host, device=cache.device)
            src, dst = indices.unbind(dim=0)
        assert cache.device == indices.device

        kernel_blocks_per_block, remainder = divmod(cache.shape[0], num_blocks)
        assert remainder == 0, (
            f"{cache.shape[0]} kernel blocks not divisible by "
            f"{num_blocks} scheduler blocks"
        )
        storage = cache.untyped_storage()
        storage_key = (cache.device, storage.data_ptr())
        scheduler_block_stride = (
            cache.stride(0) * cache.element_size() * kernel_blocks_per_block
        )
        if storage.nbytes() == num_blocks * scheduler_block_stride:
            if storage_key in copied_storages:
                continue
            copied_storages.add(storage_key)
            blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
            blocks.set_(storage)
            blocks = blocks.view(num_blocks, -1)
        else:
            # Fold virtual block splitting into the shape so that dim 0 counts
            # scheduler blocks; unflatten of dim 0 is always a view.
            blocks = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))
        if not (fused_ok and _fused_copy_kv_cache_block_rows(blocks, indices)):
            blocks[dst] = blocks[src]


def is_uniform_query_len(num_reqs: int, num_tokens: int, max_query_len: int) -> bool:
    """Whether every request in the batch has the same query length.

    Shape test only; use ``get_uniform_decode_token_count`` to classify a
    scheduled batch, since a prompt chunk can have a decode batch's shape.
    """
    return num_reqs > 0 and num_tokens == max_query_len * num_reqs


def get_uniform_decode_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int, has_prefill: bool
) -> int | None:
    """Per-request token count of a uniform decode batch, or None."""
    if not has_prefill and is_uniform_query_len(num_reqs, num_tokens, max_query_len):
        return max_query_len
    return None


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled. SP is only supported in
    full-graph compilation mode.
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        return False

    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    assert num_input_tokens % tp == 0

    return True


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }

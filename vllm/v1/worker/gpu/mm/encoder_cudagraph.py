# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA Graph Manager for Multimodal Encoders (ViT).

This module provides CUDA graph capture and replay functionality for vision
encoders to eliminate kernel launch overhead and improve GPU utilization.

Two execution modes:
1. Exact match mode: Replay CUDA graph when input grid_thw exactly matches
   a captured configuration. No padding overhead.
2. Padded mode: Pad inputs to fit the smallest captured bucket that can
   accommodate them. Enables higher CUDA graph utilization at the cost of
   padding compute overhead.

Padded mode details:
- Padded with zeros: pixel_values, pos_embeds, rotary_pos_emb_cos/sin
- NOT padded (set to actual values): cu_seqlens, max_seqlen
- This ensures flash attention only processes real tokens (via cu_seqlens)
- Output is trimmed to actual size after graph replay

Key design principles:
1. Capture graphs for specific grid_thw configurations
2. Support both exact match and padded execution
3. Fall back to eager mode when no suitable graph is available
4. Track statistics for monitoring and optimization
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# Grid configurations for CUDA graph capture (T, H, W in patch units)
#
# Strategy: Prioritize small grids where kernel launch overhead dominates.
# For larger grids, computation time dominates and CUDA graph benefit is minimal.
#
# Grids larger than max_grid_size (default 96) should use padded mode or eager.
CUSTOM_GRID_CONFIGS = [
    # === Tier 1: Very small grids (<=32) ===
    (1, 16, 16),  # 256 patches
    (1, 24, 24),  # 576 patches
    (1, 32, 32),  # 1024 patches
    # === Tier 2: Small grids (33-50) ===
    (1, 38, 38),  # 1444 patches
    (1, 40, 40),  # 1600 patches
    (1, 42, 42),  # 1764 patches
    (1, 44, 44),  # 1936 patches
    (1, 46, 46),  # 2116 patches
    (1, 50, 50),  # 2500 patches
    # === Tier 3: Medium-small grids (51-70) ===
    (1, 56, 56),  # 3136 patches
    (1, 62, 62),  # 3844 patches
    (1, 64, 64),  # 4096 patches
    (1, 68, 68),  # 4624 patches
    # === Tier 4: Medium grids (71-96) ===
    (1, 76, 76),  # 5776 patches
    (1, 80, 80),  # 6400 patches
    (1, 94, 94),  # 8836 patches
]

# Default bucket sizes for padded mode (creates square grids)
# These cover medium-large grids that are too big for exact match capture
# but still benefit from CUDA graphs via padding.
DEFAULT_PADDED_BUCKET_SIZES = [100, 128]


class EncoderCudaGraphManager:
    """
    Manages CUDA graphs for multimodal encoders (e.g., ViT in VLMs).

    The manager captures CUDA graphs for specific grid configurations
    (T, H, W in patch units) and replays them during inference when
    input dimensions exactly match.

    Design:
    - Captures graphs for predefined grid configurations
    - Only replays when input exactly matches a captured configuration
    - Falls back to eager mode for non-matching inputs
    - Tracks statistics for monitoring

    Limitations:
    - Requires exact dimension match for graph replay
    - Variable-size images may not benefit from CUDA graphs
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        bucket_sizes: list[int] | None = None,
        grid_configs: list[tuple[int, int, int]] | None = None,
        graph_pool: Any | None = None,
        verbose: bool = False,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        # Get batch sizes from config (for grouped batched mode)
        self.batch_sizes = self._get_batch_sizes_from_config()

        # Get grid configs from config or use defaults (for exact match)
        if grid_configs is None:
            grid_configs = self._get_grid_configs_from_config()

        # Get bucket sizes from config (for padded mode)
        if bucket_sizes is None:
            bucket_sizes = self._get_bucket_sizes_from_config()

        # Merge: grid_configs (exact match) + bucket_sizes (padded mode square grids)
        # Bucket sizes create square grids (1, size, size) for padded mode
        grid_set = set(grid_configs)
        for size in bucket_sizes:
            grid_set.add((1, size, size))

        # Filter out grids that are too large to capture efficiently
        # Large grids (e.g., 256x256+) consume massive memory (~14+ GiB each)
        # and are better served by eager mode or padded execution
        max_grid_size = self._get_max_grid_size_from_config()
        filtered_grids = []
        skipped_grids = []
        for grid in grid_set:
            t, h, w = grid
            if h <= max_grid_size and w <= max_grid_size:
                filtered_grids.append(grid)
            else:
                skipped_grids.append(grid)

        if skipped_grids:
            top_skipped = sorted(
                skipped_grids, key=lambda x: x[1] * x[2], reverse=True
            )[:5]
            logger.info(
                "Skipping %d grids exceeding max_grid_size=%d: %s...",
                len(skipped_grids),
                max_grid_size,
                top_skipped,
            )

        self.grid_configs = filtered_grids

        # CUDA graph storage - keyed by (batch_size, t, h, w) tuple
        # For legacy mode (batch_sizes=None), key is (1, t, h, w)
        self.graphs: dict[tuple[int, int, int, int], torch.cuda.CUDAGraph] = {}
        # Use private pools by default to avoid segfaults with rapid back-to-back
        # graph replays during one-by-one multi-image processing.
        # Set VLLM_ENCODER_SHARED_POOL=1 to use shared pool (saves memory but
        # may cause issues with rapid replays)
        import os

        if os.environ.get("VLLM_ENCODER_SHARED_POOL", "0") == "1":
            self.pool = (
                graph_pool if graph_pool is not None else torch.cuda.graph_pool_handle()
            )
            logger.info("Encoder CUDA graphs: using shared pool")
        else:
            self.pool = None  # Each graph uses private memory (default)

        # Pre-allocated input/output buffers per graph config
        # Key: (batch_size, t, h, w), Value: {"pixel_values": tensor, "grid_thw": list}
        self.input_buffers: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        self.output_buffers: dict[tuple[int, int, int, int], torch.Tensor] = {}

        # Cached pre-computed tensors for CUDA graph replay (exact match mode)
        # Key: (batch_size, t, h, w), Value: dict with pos_embeds, rotary embeddings, etc.
        self.cached_tensors: dict[tuple[int, int, int, int], dict[str, torch.Tensor]] = {}

        # Input buffers for embeddings (padded mode with runtime computation)
        # Key: (batch_size, t, h, w), Value: dict with pos_embeds, rotary_cos/sin, cu_seqlens
        self.embedding_buffers: dict[tuple[int, int, int, int], dict[str, torch.Tensor]] = {}

        # Store metadata about captured graphs
        self.captured_metadata: dict[tuple[int, int, int, int], dict[str, Any]] = {}

        # Vision encoder reference for runtime embedding computation (set at capture)
        self.vision_encoder = None

        # Track if graphs have been captured
        self.captured = False

        # Statistics
        self.cache_hits = 0
        self.eager_fallbacks = 0

        # CUDA event for lightweight synchronization
        # Instead of torch.cuda.synchronize() which waits for ALL GPU work,
        # we use an event to track only the last replay completion.
        # This allows better overlap between encoder and other GPU work.
        self.replay_done_event: torch.cuda.Event | None = None

        # Single-GPU optimization: when TP=1, PP=1, DP=1, we can capture graphs
        # on the current stream instead of a separate stream. This eliminates
        # the need for stream synchronization before replay.
        parallel_config = vllm_config.parallel_config
        self.is_single_gpu = (
            parallel_config.tensor_parallel_size == 1
            and parallel_config.pipeline_parallel_size == 1
            and parallel_config.data_parallel_size == 1
        )
        if self.is_single_gpu:
            logger.info(
                "Encoder CUDA graphs: single-GPU mode enabled "
                "(TP=1, PP=1, DP=1), using optimized sync scheme"
            )

        # Track which grids have had their embedding buffers modified by run_padded()
        # or run_batched_contiguous(). This allows run() to skip restoring cached
        # tensors when not needed. Keys are (batch_size, t, h, w).
        self.modified_grids: set[tuple[int, int, int, int]] = set()

    def _get_grid_configs_from_config(self) -> list[tuple[int, int, int]]:
        """Get encoder grid configurations from config or use defaults."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return CUSTOM_GRID_CONFIGS

        # Check for encoder-specific grid config
        grid_configs = getattr(
            compilation_config, "encoder_cudagraph_grid_configs", None
        )
        if grid_configs is not None:
            # Handle preset name or custom list
            if isinstance(grid_configs, str):
                if grid_configs == "custom":
                    return CUSTOM_GRID_CONFIGS
                else:
                    logger.warning(
                        "Unknown grid config preset '%s', using 'custom'",
                        grid_configs,
                    )
                    return CUSTOM_GRID_CONFIGS
            return [tuple(cfg) for cfg in grid_configs]

        return CUSTOM_GRID_CONFIGS

    def _get_bucket_sizes_from_config(self) -> list[int]:
        """Get encoder CUDA graph bucket sizes from config.

        Bucket sizes enable padded mode for grids that don't have exact matches.
        Default buckets (100, 128) cover medium-large grids efficiently.
        """
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return DEFAULT_PADDED_BUCKET_SIZES

        encoder_sizes = getattr(
            compilation_config, "encoder_cudagraph_bucket_sizes", None
        )
        return (
            encoder_sizes if encoder_sizes is not None else DEFAULT_PADDED_BUCKET_SIZES
        )

    def _get_max_grid_size_from_config(self) -> int:
        """Get maximum grid size for encoder CUDA graph capture.

        Large grids consume massive GPU memory per graph and provide minimal
        benefit since computation time dominates over launch overhead.

        Default is 96 to focus memory on small grids where benefit is highest.
        Grids larger than this will use padded mode (if buckets configured) or eager.
        """
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return 96  # Focus on small grids where benefit is highest

        max_size = getattr(
            compilation_config,
            "encoder_cudagraph_max_grid_size",
            96,  # Default: max 96x96 grids for exact match
        )
        return max_size

    def _get_batch_sizes_from_config(self) -> list[int]:
        """Get batch sizes for grouped batched CUDA graph capture.

        When set (e.g., [4]), captures graphs for processing multiple images
        together with the same grid size. Images are grouped by grid size and
        padded to the largest in each group.

        Default is [1] for legacy one-by-one mode.
        """
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return [1]

        batch_sizes = getattr(
            compilation_config, "encoder_cudagraph_batch_sizes", None
        )
        if batch_sizes is None:
            return [1]  # Legacy mode: batch_size=1 only
        return sorted(batch_sizes)

    def _grid_to_key(self, grid_thw: list[list[int]]) -> tuple[int, int, int] | None:
        """
        Convert a grid_thw list to a hashable key.

        Only supports single-image grids (len(grid_thw) == 1).
        Returns None for multi-image batches.
        """
        if len(grid_thw) != 1:
            return None
        t, h, w = grid_thw[0]
        return (t, h, w)

    def _compute_output_tokens(
        self,
        grid_thw: tuple[int, int, int],
        spatial_merge_size: int,
    ) -> int:
        """Compute number of output tokens for a grid configuration."""
        t, h, w = grid_thw
        # After spatial merge: tokens = T * (H/merge) * (W/merge)
        return t * (h // spatial_merge_size) * (w // spatial_merge_size)

    def _prepare_dummy_inputs_for_grid(
        self,
        grid_config: tuple[int, int, int],
        vision_encoder: nn.Module,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """
        Prepare dummy inputs for CUDA graph capture with a specific grid config.

        Args:
            grid_config: Tuple of (T, H, W) in patch units
            vision_encoder: The vision encoder module
            batch_size: Number of images in the batch (all same grid)

        Returns:
            Dict with pixel_values, grid_thw, and metadata
        """
        t, h, w = grid_config

        # Get vision encoder properties
        patch_size = vision_encoder.patch_size
        temporal_patch_size = vision_encoder.temporal_patch_size
        spatial_merge_size = vision_encoder.spatial_merge_size
        in_channels = 3  # RGB

        # Calculate patch input channels
        patch_input_channels = (
            temporal_patch_size * patch_size * patch_size * in_channels
        )

        # Calculate number of pixel patches per image
        # h, w are in patch units, so num_patches = t * h * w
        num_pixel_patches_per_image = t * h * w
        total_pixel_patches = num_pixel_patches_per_image * batch_size

        # Create dummy pixel values for batch (zeros are fine for warmup/capture)
        pixel_values = torch.zeros(
            total_pixel_patches,
            patch_input_channels,
            dtype=self.dtype,
            device=self.device,
        )

        # Grid (temporal, height, width) for each image in batch
        grid_thw = [[t, h, w]] * batch_size

        # Calculate output tokens per image and total
        output_tokens_per_image = self._compute_output_tokens(
            grid_config, spatial_merge_size
        )
        total_output_tokens = output_tokens_per_image * batch_size

        return {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "num_output_tokens": total_output_tokens,
            "num_output_tokens_per_image": output_tokens_per_image,
            "num_pixel_patches": total_pixel_patches,
            "num_pixel_patches_per_image": num_pixel_patches_per_image,
            "patch_input_channels": patch_input_channels,
            "batch_size": batch_size,
        }

    def capture_graph_for_grid(
        self,
        grid_config: tuple[int, int, int],
        vision_encoder: nn.Module,
        batch_size: int = 1,
    ) -> None:
        """
        Capture a CUDA graph for the given grid configuration and batch size.

        This method pre-computes and caches all grid-dependent tensors
        (position embeddings, rotary embeddings, cu_seqlens) to eliminate
        CPU operations during CUDA graph replay.

        Args:
            grid_config: Tuple of (T, H, W) in patch units
            vision_encoder: The vision encoder module
            batch_size: Number of images with same grid (default 1)
        """
        t, h, w = grid_config
        graph_key = (batch_size, t, h, w)
        logger.debug(
            "Capturing encoder CUDA graph for key %s (batch_size=%d, grid=%s)",
            graph_key, batch_size, grid_config
        )

        # Prepare dummy inputs for batch
        dummy_inputs = self._prepare_dummy_inputs_for_grid(
            grid_config, vision_encoder, batch_size
        )
        pixel_values = dummy_inputs["pixel_values"]
        grid_thw = dummy_inputs["grid_thw"]

        # Store input buffer reference with new key format
        self.input_buffers[graph_key] = {
            "pixel_values": pixel_values.clone(),
            "grid_thw": grid_thw,
        }

        # Store metadata
        self.captured_metadata[graph_key] = {
            "num_output_tokens": dummy_inputs["num_output_tokens"],
            "num_output_tokens_per_image": dummy_inputs["num_output_tokens_per_image"],
            "num_pixel_patches": dummy_inputs["num_pixel_patches"],
            "num_pixel_patches_per_image": dummy_inputs["num_pixel_patches_per_image"],
            "patch_input_channels": dummy_inputs["patch_input_channels"],
            "batch_size": batch_size,
        }

        # Store vision encoder reference for runtime embedding computation
        self.vision_encoder = vision_encoder

        # Check if vision encoder supports optimized CUDA graph forward
        has_cudagraph_forward = hasattr(
            vision_encoder, "forward_cudagraph"
        ) and hasattr(vision_encoder, "precompute_for_cudagraph")

        if has_cudagraph_forward:
            # Pre-compute tensors for the batched grid (used for exact match mode)
            cached = vision_encoder.precompute_for_cudagraph(grid_thw)
            self.cached_tensors[graph_key] = cached
            logger.debug(
                "Pre-computed cached tensors for key %s: "
                "pos_embeds=%s, cu_seqlens=%s",
                graph_key,
                cached["pos_embeds"].shape,
                cached["cu_seqlens"].shape,
            )

            # Create INPUT BUFFERS for embeddings (padded mode runtime computation)
            # These buffers can be updated at runtime before graph replay
            # Note: max_seqlen is a CPU scalar tensor to avoid GPU sync on .item()
            self.embedding_buffers[graph_key] = {
                "pos_embeds": cached["pos_embeds"].clone(),
                "rotary_pos_emb_cos": cached["rotary_pos_emb_cos"].clone(),
                "rotary_pos_emb_sin": cached["rotary_pos_emb_sin"].clone(),
                "cu_seqlens": cached["cu_seqlens"].clone(),
                "max_seqlen": cached["max_seqlen"].clone(),
                "sequence_lengths": cached["sequence_lengths"].clone(),
            }
            embed_buffers = self.embedding_buffers[graph_key]

            # Warmup run with embedding buffers
            # Use set_forward_context to provide vllm_config for torch.compile
            with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
            ):
                warmup_output = vision_encoder.forward_cudagraph(
                    pixel_values,
                    pos_embeds=embed_buffers["pos_embeds"],
                    rotary_pos_emb_cos=embed_buffers["rotary_pos_emb_cos"],
                    rotary_pos_emb_sin=embed_buffers["rotary_pos_emb_sin"],
                    cu_seqlens=embed_buffers["cu_seqlens"],
                    max_seqlen=embed_buffers["max_seqlen"],
                    sequence_lengths=embed_buffers["sequence_lengths"],
                )
                self.output_buffers[graph_key] = torch.empty_like(warmup_output)

            # Capture the graph with embedding BUFFERS (not constants)
            # This allows updating embeddings at runtime for padded mode
            graph = torch.cuda.CUDAGraph()
            input_buffer = self.input_buffers[graph_key]["pixel_values"]

            with (
                set_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                ),
                torch.cuda.graph(graph, self.pool),
            ):
                output = vision_encoder.forward_cudagraph(
                    input_buffer,
                    pos_embeds=embed_buffers["pos_embeds"],
                    rotary_pos_emb_cos=embed_buffers["rotary_pos_emb_cos"],
                    rotary_pos_emb_sin=embed_buffers["rotary_pos_emb_sin"],
                    cu_seqlens=embed_buffers["cu_seqlens"],
                    max_seqlen=embed_buffers["max_seqlen"],
                    sequence_lengths=embed_buffers["sequence_lengths"],
                )
                self.output_buffers[graph_key].copy_(output)
        else:
            # Fallback to original forward (will have CPU gaps)
            logger.warning(
                "Vision encoder does not support forward_cudagraph, "
                "using standard forward (will have CPU gaps)"
            )

            # Warmup run (required before capture)
            with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
            ):
                warmup_output = vision_encoder(pixel_values, grid_thw=grid_thw)
                self.output_buffers[graph_key] = torch.empty_like(warmup_output)

            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            input_buffer = self.input_buffers[graph_key]["pixel_values"]

            with (
                set_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                ),
                torch.cuda.graph(graph, self.pool),
            ):
                output = vision_encoder(input_buffer, grid_thw=grid_thw)
                self.output_buffers[graph_key].copy_(output)

        self.graphs[graph_key] = graph
        cached_suffix = " (with cached tensors)" if has_cudagraph_forward else ""
        logger.debug(
            "Captured encoder CUDA graph for key %s -> %d output tokens%s",
            graph_key,
            dummy_inputs["num_output_tokens"],
            cached_suffix,
        )

    @torch.inference_mode()
    def capture(
        self,
        vision_encoder: nn.Module,
        embed_multimodal_fn: Callable,
    ) -> None:
        """
        Capture CUDA graphs for all configured grid and batch size combinations.

        Args:
            vision_encoder: The vision encoder module (e.g., Qwen3_VisionTransformer)
            embed_multimodal_fn: The model's embed_multimodal method (unused)
        """
        if self.captured:
            logger.warning("Encoder CUDA graphs already captured, skipping")
            return

        # Build list of (batch_size, grid_config) combinations to capture
        capture_combinations = []
        for batch_size in self.batch_sizes:
            for grid_config in self.grid_configs:
                capture_combinations.append((batch_size, grid_config))

        # Log initial memory state
        free_mem_before, total_mem = torch.cuda.mem_get_info(self.device)
        used_mem_before = total_mem - free_mem_before
        logger.info(
            "Capturing encoder CUDA graphs for %d combinations "
            "(batch_sizes=%s, grids=%d) "
            "(GPU memory: %.2f GiB used, %.2f GiB free)",
            len(capture_combinations),
            self.batch_sizes,
            len(self.grid_configs),
            used_mem_before / 1024**3,
            free_mem_before / 1024**3,
        )

        # Capture from smallest to largest so that common smaller grids are
        # captured first. If we run out of memory, only large grids will fail.
        capture_combinations = sorted(
            capture_combinations,
            key=lambda x: x[0] * x[1][0] * x[1][1] * x[1][2],  # batch * t * h * w
            reverse=False,  # Smallest first
        )

        if is_global_first_rank():
            capture_combinations = tqdm(
                capture_combinations, desc="Capturing encoder CUDA graphs"
            )

        # Capture each graph. For single-GPU mode, capture directly on current stream
        # to avoid stream synchronization overhead at replay time.
        # For multi-GPU mode, use graph_capture() context to coordinate with TP/PP.
        for batch_size, grid_config in capture_combinations:
            try:
                if self.is_single_gpu:
                    # Single-GPU: capture on current stream (no separate stream)
                    self.capture_graph_for_grid(
                        grid_config,
                        vision_encoder,
                        batch_size=batch_size,
                    )
                else:
                    # Multi-GPU: use graph_capture() for TP/PP coordination
                    with graph_capture(device=self.device):
                        self.capture_graph_for_grid(
                            grid_config,
                            vision_encoder,
                            batch_size=batch_size,
                        )
            except Exception as e:
                logger.warning(
                    "Failed to capture encoder CUDA graph for "
                    "batch_size=%d, grid=%s: %s. Will use eager mode.",
                    batch_size,
                    grid_config,
                    e,
                )

        self.captured = True

        # Log final memory state
        free_mem_after, _ = torch.cuda.mem_get_info(self.device)
        used_mem_after = total_mem - free_mem_after
        encoder_graph_mem = used_mem_after - used_mem_before
        logger.info(
            "Captured %d encoder CUDA graphs (configs: %s). "
            "Encoder graph memory: %.2f GiB (GPU: %.2f GiB used, %.2f GiB free)",
            len(self.graphs),
            sorted(self.graphs.keys()),
            encoder_graph_mem / 1024**3,
            used_mem_after / 1024**3,
            free_mem_after / 1024**3,
        )

    def get_graph_for_grid(
        self,
        grid_thw: list[list[int]],
        batch_size: int = 1,
    ) -> tuple[int, int, int, int] | None:
        """
        Check if a CUDA graph is available for the given grid and batch size.

        Args:
            grid_thw: List of [T, H, W] for each image (must all be same grid)
            batch_size: Number of images (default 1 for legacy mode)

        Returns:
            The graph key (batch_size, t, h, w) if matching graph exists, None otherwise
        """
        if len(grid_thw) < 1:
            return None
        # All images must have the same grid for batched mode
        t, h, w = grid_thw[0]
        for grid in grid_thw[1:]:
            if grid != [t, h, w]:
                return None  # Mixed grids not supported
        key = (batch_size, t, h, w)
        return key if key in self.graphs else None

    def find_bucket_for_tokens(
        self,
        num_tokens: int,
        spatial_merge_size: int = 2,
        batch_size: int = 1,
    ) -> tuple[int, int, int, int] | None:
        """
        Find the smallest captured grid that can fit the given token count.

        This enables padded execution where inputs smaller than a bucket
        are padded to match the bucket size.

        Args:
            num_tokens: Number of output tokens needed (per image)
            spatial_merge_size: Merge size (default 2)
            batch_size: Required batch size (default 1)

        Returns:
            Graph key (batch_size, T, H, W) of the best bucket, or None if too large
        """
        best_key = None
        best_bucket_tokens = float("inf")

        for graph_key in self.graphs:
            key_batch_size, t, h, w = graph_key
            if key_batch_size != batch_size:
                continue  # Skip graphs with wrong batch size
            grid = (t, h, w)
            bucket_tokens = self._compute_output_tokens(grid, spatial_merge_size)
            if bucket_tokens >= num_tokens and bucket_tokens < best_bucket_tokens:
                best_bucket_tokens = bucket_tokens
                best_key = graph_key

        return best_key

    def run(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
        batch_size: int = 1,
    ) -> torch.Tensor | None:
        """
        Run the vision encoder using a captured CUDA graph if available.

        Args:
            pixel_values: Input pixel values [num_patches, patch_channels]
            grid_thw: List of [T, H, W] for each image (all must be same grid)
            batch_size: Number of images in batch (default 1 for legacy mode)

        Returns:
            Vision encoder output tensor if graph was used, None if no matching graph
        """
        graph_key = self.get_graph_for_grid(grid_thw, batch_size=batch_size)

        if graph_key is None:
            # Don't count miss here - caller may try run_padded() next
            return None

        # Verify input dimensions match
        input_buffer = self.input_buffers[graph_key]["pixel_values"]
        if pixel_values.shape != input_buffer.shape:
            logger.warning(
                "Pixel values shape mismatch: expected %s, got %s. "
                "Falling back to eager mode.",
                input_buffer.shape,
                pixel_values.shape,
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                "Device mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.device,
                pixel_values.device,
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                "Dtype mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.dtype,
                pixel_values.dtype,
            )
            self.eager_fallbacks += 1
            return None

        self.cache_hits += 1

        # Wait for any previous graph replay to complete before modifying buffers.
        # For single-GPU mode, this is not needed because everything is on the same
        # stream and CUDA guarantees ordering. For multi-GPU mode, we need this
        # because the graph runs on a different stream.
        if not self.is_single_gpu and self.replay_done_event is not None:
            self.replay_done_event.synchronize()

        # Ensure contiguous memory layout for safe copy
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        # Copy input to the captured buffer (non-blocking for better overlap)
        input_buffer.copy_(pixel_values, non_blocking=True)

        # For exact match, restore cached embeddings only if modified by run_padded().
        # This avoids 6 unnecessary tensor copies when only using exact-match mode.
        if graph_key in self.modified_grids:
            embed_buffers = self.embedding_buffers[graph_key]
            cached = self.cached_tensors[graph_key]
            embed_buffers["pos_embeds"].copy_(cached["pos_embeds"], non_blocking=True)
            embed_buffers["rotary_pos_emb_cos"].copy_(
                cached["rotary_pos_emb_cos"], non_blocking=True
            )
            embed_buffers["rotary_pos_emb_sin"].copy_(
                cached["rotary_pos_emb_sin"], non_blocking=True
            )
            embed_buffers["cu_seqlens"].copy_(cached["cu_seqlens"], non_blocking=True)
            embed_buffers["max_seqlen"].copy_(cached["max_seqlen"], non_blocking=True)
            embed_buffers["sequence_lengths"].copy_(
                cached["sequence_lengths"], non_blocking=True
            )
            self.modified_grids.discard(graph_key)

        if self.verbose:
            logger.info(
                "run(): graph_key=%s, input_shape=%s, buffer_shape=%s",
                graph_key,
                pixel_values.shape,
                input_buffer.shape,
            )

        if self.is_single_gpu:
            # Single-GPU optimized path: graph was captured on current stream,
            # so buffer copies and replay are on the same stream - no sync needed.
            # Return view directly; caller must use output before next run() call.
            self.graphs[graph_key].replay()
            return self.output_buffers[graph_key]
        else:
            # Multi-GPU path: graph was captured on a separate stream.
            # Sync current stream before replay to ensure buffer copies complete.
            torch.cuda.current_stream().synchronize()

            # Replay the graph
            self.graphs[graph_key].replay()

            # Record event after replay for lightweight sync in next call.
            if self.replay_done_event is None:
                self.replay_done_event = torch.cuda.Event()
            self.replay_done_event.record()

            # Sync to ensure output is ready before clone.
            self.replay_done_event.synchronize()

            # Return a clone of the output to avoid issues with buffer reuse
            return self.output_buffers[graph_key].clone()

    def run_padded(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
        num_output_tokens: int,
        spatial_merge_size: int = 2,
    ) -> tuple[torch.Tensor, int] | None:
        """
        Run the vision encoder with padding to fit a captured bucket.

        This method computes embeddings for the ACTUAL input grid, pads them
        to match the bucket size, then replays the CUDA graph. This ensures
        correct position embeddings while still benefiting from CUDA graphs.

        Args:
            pixel_values: Input pixel values [num_patches, patch_channels]
            grid_thw: List of [T, H, W] for each image (only single image supported)
            num_output_tokens: Expected number of output tokens for the input
            spatial_merge_size: Spatial merge size (default 2)

        Returns:
            Tuple of (output tensor trimmed to actual size, padding_waste_tokens)
            or None if no suitable bucket found
        """
        if len(grid_thw) != 1:
            logger.debug("Padded mode only supports single-image inputs")
            return None

        # Check if vision encoder is available for embedding computation
        if self.vision_encoder is None or not hasattr(
            self.vision_encoder, "precompute_for_cudagraph"
        ):
            logger.debug("Vision encoder not available for padded mode")
            return None

        # Find the smallest bucket that fits (for batch_size=1)
        graph_key = self.find_bucket_for_tokens(
            num_output_tokens, spatial_merge_size, batch_size=1
        )
        if graph_key is None:
            # Don't count miss here - caller will count it when falling back to eager
            max_available = (
                max(
                    self._compute_output_tokens((t, h, w), spatial_merge_size)
                    for (bs, t, h, w) in self.graphs
                    if bs == 1  # Only consider batch_size=1 graphs
                )
                if self.graphs
                else 0
            )
            logger.debug(
                "No bucket found for %d tokens, max available: %d",
                num_output_tokens,
                max_available,
            )
            return None

        # Check if we have embedding buffers for this bucket
        if graph_key not in self.embedding_buffers:
            logger.debug("No embedding buffers for bucket %s", graph_key)
            return None

        # Extract grid from graph_key for _compute_output_tokens
        _, t, h, w = graph_key
        bucket_tokens = self._compute_output_tokens((t, h, w), spatial_merge_size)
        padding_waste = bucket_tokens - num_output_tokens

        # Get the input buffer for this bucket
        input_buffer = self.input_buffers[graph_key]["pixel_values"]
        num_input_patches = pixel_values.shape[0]
        bucket_input_patches = input_buffer.shape[0]

        if num_input_patches > bucket_input_patches:
            logger.warning(
                "Input patches (%d) exceed bucket capacity (%d). "
                "This shouldn't happen.",
                num_input_patches,
                bucket_input_patches,
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                "Device mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.device,
                pixel_values.device,
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                "Dtype mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.dtype,
                pixel_values.dtype,
            )
            self.eager_fallbacks += 1
            return None

        # Ensure contiguous memory layout for safe copy
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        self.cache_hits += 1

        # Wait for any previous graph replay to complete before modifying buffers.
        # For single-GPU mode, this is not needed because everything is on the same
        # stream and CUDA guarantees ordering.
        if not self.is_single_gpu and self.replay_done_event is not None:
            self.replay_done_event.synchronize()

        # Compute embeddings for ACTUAL grid, then pad to bucket size.
        # This ensures correct position embeddings for the actual input size.
        actual_embeds = self.vision_encoder.precompute_for_cudagraph(grid_thw)

        # Get embedding buffers for the bucket
        embed_buffers = self.embedding_buffers[graph_key]

        # Zero the buffers first (for clean padding)
        input_buffer.zero_()
        embed_buffers["pos_embeds"].zero_()
        embed_buffers["rotary_pos_emb_cos"].zero_()
        embed_buffers["rotary_pos_emb_sin"].zero_()

        # Copy actual pixel values to the beginning of the buffer
        input_buffer[:num_input_patches].copy_(pixel_values, non_blocking=True)

        # Copy actual embeddings to the beginning of the buffers (pad with zeros)
        actual_num_patches = actual_embeds["pos_embeds"].shape[0]
        embed_buffers["pos_embeds"][:actual_num_patches].copy_(
            actual_embeds["pos_embeds"], non_blocking=True
        )
        embed_buffers["rotary_pos_emb_cos"][:actual_num_patches].copy_(
            actual_embeds["rotary_pos_emb_cos"], non_blocking=True
        )
        embed_buffers["rotary_pos_emb_sin"][:actual_num_patches].copy_(
            actual_embeds["rotary_pos_emb_sin"], non_blocking=True
        )

        # Update cu_seqlens and max_seqlen to actual values
        # cu_seqlens shape is [num_images + 1], for single image: [0, num_patches]
        # We copy actual values so flash attention processes only the real tokens
        embed_buffers["cu_seqlens"].copy_(
            actual_embeds["cu_seqlens"], non_blocking=True
        )
        embed_buffers["max_seqlen"].copy_(
            actual_embeds["max_seqlen"], non_blocking=True
        )
        embed_buffers["sequence_lengths"].copy_(
            actual_embeds["sequence_lengths"], non_blocking=True
        )

        # Mark this grid as modified so run() knows to restore cached tensors
        self.modified_grids.add(graph_key)

        if self.verbose:
            logger.info(
                "run_padded(): graph_key=%s, actual_grid=%s, "
                "input_patches=%d, bucket_patches=%d",
                graph_key,
                grid_thw[0],
                num_input_patches,
                bucket_input_patches,
            )

        if self.is_single_gpu:
            # Single-GPU optimized path: graph was captured on current stream,
            # so buffer modifications and replay are on same stream - no sync needed.
            # Return view directly; caller must use output before next run() call.
            self.graphs[graph_key].replay()
            full_output = self.output_buffers[graph_key]
            trimmed_output = full_output[:num_output_tokens]
        else:
            # Multi-GPU path: graph was captured on a separate stream.
            # Sync current stream before replay to ensure buffer modifications complete.
            torch.cuda.current_stream().synchronize()

            # Replay the graph with updated embedding buffers
            self.graphs[graph_key].replay()

            # Record event after replay for lightweight sync in next call.
            if self.replay_done_event is None:
                self.replay_done_event = torch.cuda.Event()
            self.replay_done_event.record()

            # Sync to ensure output is ready before clone.
            self.replay_done_event.synchronize()

            # Get output and trim to actual size
            full_output = self.output_buffers[graph_key]
            trimmed_output = full_output[:num_output_tokens].clone()

        if self.verbose:
            logger.debug(
                "Padded execution: %d -> %d tokens (waste: %d, %.1f%%)",
                num_output_tokens,
                bucket_tokens,
                padding_waste,
                padding_waste / bucket_tokens * 100,
            )

        return trimmed_output, padding_waste

    def run_batched(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
        batch_size: int,
    ) -> torch.Tensor | None:
        """
        Run the vision encoder for a batch of images with the same grid size.

        This is used for grouped batching where multiple images are processed
        together with a single CUDA graph replay.

        Args:
            pixel_values: Concatenated pixel values [total_patches, patch_channels]
            grid_thw: List of [T, H, W] for each image (all must be same grid)
            batch_size: Number of images in the batch

        Returns:
            Concatenated output tensor for all images, or None if no matching graph
        """
        if len(grid_thw) != batch_size:
            logger.warning(
                "grid_thw length (%d) doesn't match batch_size (%d)",
                len(grid_thw), batch_size
            )
            return None

        # All images must have the same grid
        if len(grid_thw) < 1:
            return None
        base_grid = grid_thw[0]
        for grid in grid_thw[1:]:
            if grid != base_grid:
                logger.warning(
                    "run_batched requires all images to have same grid, "
                    "got %s and %s", base_grid, grid
                )
                return None

        # Look up the graph for this batch_size and grid
        graph_key = self.get_graph_for_grid(grid_thw, batch_size=batch_size)
        if graph_key is None:
            return None

        # Verify input dimensions match
        input_buffer = self.input_buffers[graph_key]["pixel_values"]
        if pixel_values.shape != input_buffer.shape:
            logger.warning(
                "Pixel values shape mismatch: expected %s, got %s. "
                "Falling back to eager mode.",
                input_buffer.shape,
                pixel_values.shape,
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                "Device mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.device,
                pixel_values.device,
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                "Dtype mismatch: expected %s, got %s. Falling back to eager mode.",
                input_buffer.dtype,
                pixel_values.dtype,
            )
            self.eager_fallbacks += 1
            return None

        self.cache_hits += 1

        # Wait for any previous graph replay to complete before modifying buffers.
        if not self.is_single_gpu and self.replay_done_event is not None:
            self.replay_done_event.synchronize()

        # Ensure contiguous memory layout for safe copy
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        # Copy input to the captured buffer
        input_buffer.copy_(pixel_values, non_blocking=True)

        # For batched exact match, restore cached embeddings if modified
        if graph_key in self.modified_grids:
            embed_buffers = self.embedding_buffers[graph_key]
            cached = self.cached_tensors[graph_key]
            embed_buffers["pos_embeds"].copy_(cached["pos_embeds"], non_blocking=True)
            embed_buffers["rotary_pos_emb_cos"].copy_(
                cached["rotary_pos_emb_cos"], non_blocking=True
            )
            embed_buffers["rotary_pos_emb_sin"].copy_(
                cached["rotary_pos_emb_sin"], non_blocking=True
            )
            embed_buffers["cu_seqlens"].copy_(cached["cu_seqlens"], non_blocking=True)
            embed_buffers["max_seqlen"].copy_(cached["max_seqlen"], non_blocking=True)
            embed_buffers["sequence_lengths"].copy_(
                cached["sequence_lengths"], non_blocking=True
            )
            self.modified_grids.discard(graph_key)

        if self.verbose:
            logger.info(
                "run_batched(): graph_key=%s, batch_size=%d, input_shape=%s",
                graph_key, batch_size, pixel_values.shape,
            )

        if self.is_single_gpu:
            self.graphs[graph_key].replay()
            return self.output_buffers[graph_key]
        else:
            torch.cuda.current_stream().synchronize()
            self.graphs[graph_key].replay()
            if self.replay_done_event is None:
                self.replay_done_event = torch.cuda.Event()
            self.replay_done_event.record()
            self.replay_done_event.synchronize()
            return self.output_buffers[graph_key].clone()

    def run_batched_contiguous(
        self,
        pixel_values: torch.Tensor,
        grid_thw_list: list[list[int]],
        graph_key: tuple[int, int, int, int],
        spatial_merge_size: int = 2,
    ) -> torch.Tensor | None:
        """
        Run batched CUDA graph with contiguous packing and end padding.

        This method packs images contiguously in the buffer (no interleaved padding),
        computes actual cu_seqlens at runtime, and pads only at the end. This ensures
        flash attention reads correct data for each sequence.

        Memory layout:
            Buffer: [img0][img1][img2][img3][PADDING at end]
            cu_seqlens: [0, size0, size0+size1, ..., total_actual]

        Flash attention uses cu_seqlens to process only actual tokens; padding at
        the end is outside all sequence boundaries and is ignored.

        Args:
            pixel_values: Contiguously packed pixel values (no padding between images)
            grid_thw_list: List of [T, H, W] for each image (can be different grids)
            graph_key: The bucket graph key (batch_size, t, h, w) to use
            spatial_merge_size: Spatial merge size (default 2)

        Returns:
            Full output tensor from the bucket, or None if failed.
            Caller should use cu_seqlens to extract per-image outputs.
        """
        if graph_key not in self.graphs:
            logger.debug("No graph for key %s", graph_key)
            return None

        batch_size = graph_key[0]
        if len(grid_thw_list) != batch_size:
            logger.warning(
                "grid_thw_list length (%d) doesn't match graph batch_size (%d)",
                len(grid_thw_list), batch_size
            )
            return None

        # Check if vision encoder is available for embedding computation
        if self.vision_encoder is None or not hasattr(
            self.vision_encoder, "precompute_for_cudagraph"
        ):
            logger.debug("Vision encoder not available for batched contiguous mode")
            return None

        # Check if we have embedding buffers for this bucket
        if graph_key not in self.embedding_buffers:
            logger.debug("No embedding buffers for bucket %s", graph_key)
            return None

        # Get the input buffer for this bucket
        input_buffer = self.input_buffers[graph_key]["pixel_values"]
        actual_input_patches = pixel_values.shape[0]
        bucket_input_patches = input_buffer.shape[0]

        if actual_input_patches > bucket_input_patches:
            logger.warning(
                "Input patches (%d) exceed bucket capacity (%d).",
                actual_input_patches, bucket_input_patches,
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                "Device mismatch: expected %s, got %s.",
                input_buffer.device, pixel_values.device,
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                "Dtype mismatch: expected %s, got %s.",
                input_buffer.dtype, pixel_values.dtype,
            )
            self.eager_fallbacks += 1
            return None

        # Ensure contiguous memory layout
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        self.cache_hits += 1

        # Wait for any previous graph replay to complete
        if not self.is_single_gpu and self.replay_done_event is not None:
            self.replay_done_event.synchronize()

        # Get embedding buffers for the bucket
        embed_buffers = self.embedding_buffers[graph_key]

        # Zero the buffers first (for clean padding at end)
        input_buffer.zero_()
        embed_buffers["pos_embeds"].zero_()
        embed_buffers["rotary_pos_emb_cos"].zero_()
        embed_buffers["rotary_pos_emb_sin"].zero_()

        # Copy actual pixel values to the beginning of the buffer (contiguous)
        input_buffer[:actual_input_patches].copy_(pixel_values, non_blocking=True)

        # Compute embeddings for each actual grid and pack contiguously
        # Also build cu_seqlens from actual cumulative sizes
        pos_embeds_list = []
        rotary_cos_list = []
        rotary_sin_list = []
        sequence_lengths = []

        for grid in grid_thw_list:
            # Compute embeddings for this actual grid
            actual_embeds = self.vision_encoder.precompute_for_cudagraph([grid])
            pos_embeds_list.append(actual_embeds["pos_embeds"])
            rotary_cos_list.append(actual_embeds["rotary_pos_emb_cos"])
            rotary_sin_list.append(actual_embeds["rotary_pos_emb_sin"])
            # Output tokens for this image
            t, h, w = grid
            output_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
            sequence_lengths.append(output_tokens)

        # Concatenate embeddings (contiguous packing)
        packed_pos_embeds = torch.cat(pos_embeds_list, dim=0)
        packed_rotary_cos = torch.cat(rotary_cos_list, dim=0)
        packed_rotary_sin = torch.cat(rotary_sin_list, dim=0)

        # Copy packed embeddings to buffer (padding remains zero at end)
        actual_output_tokens = packed_pos_embeds.shape[0]
        embed_buffers["pos_embeds"][:actual_output_tokens].copy_(
            packed_pos_embeds, non_blocking=True
        )
        embed_buffers["rotary_pos_emb_cos"][:actual_output_tokens].copy_(
            packed_rotary_cos, non_blocking=True
        )
        embed_buffers["rotary_pos_emb_sin"][:actual_output_tokens].copy_(
            packed_rotary_sin, non_blocking=True
        )

        # Build cu_seqlens from actual cumulative sizes
        # cu_seqlens = [0, size0, size0+size1, ..., total]
        cu_seqlens_list = [0]
        for length in sequence_lengths:
            cu_seqlens_list.append(cu_seqlens_list[-1] + length)

        cu_seqlens_tensor = torch.tensor(
            cu_seqlens_list, dtype=torch.int32, device=self.device
        )
        max_seqlen = max(sequence_lengths)
        max_seqlen_tensor = torch.tensor(
            max_seqlen, dtype=torch.int32, device="cpu"
        )
        sequence_lengths_tensor = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=self.device
        )

        # Update cu_seqlens buffer - need to handle size mismatch
        # The captured buffer may be larger, so we update only the actual part
        embed_buffers["cu_seqlens"][:len(cu_seqlens_list)].copy_(
            cu_seqlens_tensor, non_blocking=True
        )
        embed_buffers["max_seqlen"].copy_(max_seqlen_tensor, non_blocking=True)
        embed_buffers["sequence_lengths"][:batch_size].copy_(
            sequence_lengths_tensor, non_blocking=True
        )

        # Mark this grid as modified so run() knows to restore cached tensors
        self.modified_grids.add(graph_key)

        if self.verbose:
            logger.info(
                "run_batched_contiguous(): graph_key=%s, grids=%s, "
                "actual_patches=%d, bucket_patches=%d, cu_seqlens=%s",
                graph_key, grid_thw_list, actual_input_patches,
                bucket_input_patches, cu_seqlens_list,
            )

        if self.is_single_gpu:
            self.graphs[graph_key].replay()
            return self.output_buffers[graph_key]
        else:
            torch.cuda.current_stream().synchronize()
            self.graphs[graph_key].replay()
            if self.replay_done_event is None:
                self.replay_done_event = torch.cuda.Event()
            self.replay_done_event.record()
            self.replay_done_event.synchronize()
            return self.output_buffers[graph_key].clone()

    def count_miss(self) -> None:
        """Count when falling back to eager mode.

        This should be called by the caller when neither run() nor run_padded()
        succeeded and eager execution is used.
        """
        self.eager_fallbacks += 1

    def get_stats(self, verbose: bool = True) -> dict[str, Any]:
        """Get and optionally log cache statistics.

        Args:
            verbose: If True, log stats to INFO level. If False, only return stats dict.
        """
        total = self.cache_hits + self.eager_fallbacks
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        stats = {
            "cache_hits": self.cache_hits,
            "eager_fallbacks": self.eager_fallbacks,
            "hit_rate": hit_rate,
            "num_graphs": len(self.graphs),
            "captured_configs": sorted(self.graphs.keys()),
        }
        if verbose:
            logger.info(
                "Encoder CUDA graph stats: hits=%d, eager=%d, "
                "hit_rate=%.1f%%, num_graphs=%d",
                self.cache_hits,
                self.eager_fallbacks,
                hit_rate * 100,
                len(self.graphs),
            )
        return stats

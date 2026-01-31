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
    from vllm.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)

# Grid configurations for CUDA graph capture (T, H, W in patch units)
#
# Strategy: Prioritize small grids where kernel launch overhead dominates.
# For larger grids, computation time dominates and CUDA graph benefit is minimal.
#
# Grids larger than max_grid_size (default 96) should use padded mode or eager.
CUSTOM_GRID_CONFIGS = [
    # === Tier 1: Very small grids (<=32) ===
    (1, 16, 16),   # 256 patches
    (1, 24, 24),   # 576 patches
    (1, 32, 32),   # 1024 patches

    # === Tier 2: Small grids (33-50) ===
    (1, 38, 38),   # 1444 patches
    (1, 40, 40),   # 1600 patches
    (1, 42, 42),   # 1764 patches
    (1, 44, 44),   # 1936 patches
    (1, 46, 46),   # 2116 patches
    (1, 50, 50),   # 2500 patches

    # === Tier 3: Medium-small grids (51-70) ===
    (1, 56, 56),   # 3136 patches
    (1, 62, 62),   # 3844 patches
    (1, 64, 64),   # 4096 patches
    (1, 68, 68),   # 4624 patches

    # === Tier 4: Medium grids (71-96) ===
    (1, 76, 76),   # 5776 patches
    (1, 80, 80),   # 6400 patches
    (1, 94, 94),   # 8836 patches
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
            logger.info(
                f"Skipping {len(skipped_grids)} grids exceeding max_grid_size={max_grid_size}: "
                f"{sorted(skipped_grids, key=lambda x: x[1]*x[2], reverse=True)[:5]}..."
            )

        self.grid_configs = filtered_grids

        # CUDA graph storage - keyed by (t, h, w) tuple
        self.graphs: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        # Use private pools by default to avoid segfaults with rapid back-to-back
        # graph replays during one-by-one multi-image processing.
        # Set VLLM_ENCODER_SHARED_POOL=1 to use shared pool (saves memory but
        # may cause issues with rapid replays)
        import os
        if os.environ.get("VLLM_ENCODER_SHARED_POOL", "0") == "1":
            self.pool = graph_pool if graph_pool is not None else torch.cuda.graph_pool_handle()
            logger.info("Encoder CUDA graphs: using shared pool")
        else:
            self.pool = None  # Each graph uses private memory (default)

        # Pre-allocated input/output buffers per grid config
        # Key: (t, h, w), Value: {"pixel_values": tensor, "grid_thw": list}
        self.input_buffers: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.output_buffers: dict[tuple[int, int, int], torch.Tensor] = {}

        # Cached pre-computed tensors for CUDA graph replay (used for exact match mode)
        # Key: (t, h, w), Value: dict with pos_embeds, rotary embeddings, cu_seqlens, etc.
        self.cached_tensors: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}

        # Input buffers for embeddings (used for padded mode with runtime computation)
        # Key: (t, h, w), Value: dict with pos_embeds, rotary_cos, rotary_sin, cu_seqlens buffers
        self.embedding_buffers: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}

        # Store metadata about captured graphs
        self.captured_metadata: dict[tuple[int, int, int], dict[str, Any]] = {}

        # Reference to vision encoder for runtime embedding computation (set during capture)
        self.vision_encoder = None

        # Track if graphs have been captured
        self.captured = False

        # Statistics
        self.cache_hits = 0
        self.eager_fallbacks = 0

    def _get_grid_configs_from_config(self) -> list[tuple[int, int, int]]:
        """Get encoder grid configurations from config or use defaults."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return CUSTOM_GRID_CONFIGS

        # Check for encoder-specific grid config
        grid_configs = getattr(
            compilation_config,
            'encoder_cudagraph_grid_configs',
            None
        )
        if grid_configs is not None:
            # Handle preset name or custom list
            if isinstance(grid_configs, str):
                if grid_configs == "custom":
                    return CUSTOM_GRID_CONFIGS
                else:
                    logger.warning(
                        f"Unknown grid config preset '{grid_configs}', "
                        "using 'custom'"
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
            compilation_config,
            'encoder_cudagraph_bucket_sizes',
            None
        )
        return encoder_sizes if encoder_sizes is not None else DEFAULT_PADDED_BUCKET_SIZES

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
            'encoder_cudagraph_max_grid_size',
            96  # Default: max 96x96 grids for exact match
        )
        return max_size

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
    ) -> dict[str, Any]:
        """
        Prepare dummy inputs for CUDA graph capture with a specific grid config.

        Args:
            grid_config: Tuple of (T, H, W) in patch units
            vision_encoder: The vision encoder module

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

        # Calculate number of pixel patches (before patch embedding)
        # h, w are in patch units, so num_patches = t * h * w
        num_pixel_patches = t * h * w

        # Create dummy pixel values (zeros are fine for warmup/capture)
        pixel_values = torch.zeros(
            num_pixel_patches,
            patch_input_channels,
            dtype=self.dtype,
            device=self.device,
        )

        # Grid THW for this configuration
        grid_thw = [[t, h, w]]

        # Calculate output tokens
        output_tokens = self._compute_output_tokens(
            grid_config, spatial_merge_size
        )

        return {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "num_output_tokens": output_tokens,
            "num_pixel_patches": num_pixel_patches,
            "patch_input_channels": patch_input_channels,
        }

    def capture_graph_for_grid(
        self,
        grid_config: tuple[int, int, int],
        vision_encoder: nn.Module,
    ) -> None:
        """
        Capture a CUDA graph for the given grid configuration.

        This method pre-computes and caches all grid-dependent tensors
        (position embeddings, rotary embeddings, cu_seqlens) to eliminate
        CPU operations during CUDA graph replay.

        Args:
            grid_config: Tuple of (T, H, W) in patch units
            vision_encoder: The vision encoder module
        """
        logger.debug(f"Capturing encoder CUDA graph for grid config {grid_config}")

        # Prepare dummy inputs
        dummy_inputs = self._prepare_dummy_inputs_for_grid(grid_config, vision_encoder)
        pixel_values = dummy_inputs["pixel_values"]
        grid_thw = dummy_inputs["grid_thw"]

        # Store input buffer reference
        self.input_buffers[grid_config] = {
            "pixel_values": pixel_values.clone(),
            "grid_thw": grid_thw,
        }

        # Store metadata
        self.captured_metadata[grid_config] = {
            "num_output_tokens": dummy_inputs["num_output_tokens"],
            "num_pixel_patches": dummy_inputs["num_pixel_patches"],
            "patch_input_channels": dummy_inputs["patch_input_channels"],
        }

        # Store vision encoder reference for runtime embedding computation
        self.vision_encoder = vision_encoder

        # Check if vision encoder supports optimized CUDA graph forward
        has_cudagraph_forward = hasattr(vision_encoder, 'forward_cudagraph') and \
                                hasattr(vision_encoder, 'precompute_for_cudagraph')

        if has_cudagraph_forward:
            # Pre-compute tensors for the bucket grid (used for exact match mode)
            cached = vision_encoder.precompute_for_cudagraph(grid_thw)
            self.cached_tensors[grid_config] = cached
            logger.debug(
                f"Pre-computed cached tensors for grid config {grid_config}: "
                f"pos_embeds={cached['pos_embeds'].shape}, "
                f"cu_seqlens={cached['cu_seqlens'].shape}"
            )

            # Create INPUT BUFFERS for embeddings (for padded mode with runtime computation)
            # These buffers can be updated at runtime before graph replay
            # Note: max_seqlen is a CPU scalar tensor to avoid GPU sync on .item()
            self.embedding_buffers[grid_config] = {
                "pos_embeds": cached["pos_embeds"].clone(),
                "rotary_pos_emb_cos": cached["rotary_pos_emb_cos"].clone(),
                "rotary_pos_emb_sin": cached["rotary_pos_emb_sin"].clone(),
                "cu_seqlens": cached["cu_seqlens"].clone(),
                "max_seqlen": cached["max_seqlen"].clone(),
            }
            embed_buffers = self.embedding_buffers[grid_config]

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
                )
                self.output_buffers[grid_config] = torch.empty_like(warmup_output)

            # Capture the graph with embedding BUFFERS (not constants)
            # This allows updating embeddings at runtime for padded mode
            graph = torch.cuda.CUDAGraph()
            input_buffer = self.input_buffers[grid_config]["pixel_values"]

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
                )
                self.output_buffers[grid_config].copy_(output)
        else:
            # Fallback to original forward (will have CPU gaps)
            logger.warning(
                f"Vision encoder does not support forward_cudagraph, "
                f"using standard forward (will have CPU gaps)"
            )

            # Warmup run (required before capture)
            with set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
            ):
                warmup_output = vision_encoder(pixel_values, grid_thw=grid_thw)
                self.output_buffers[grid_config] = torch.empty_like(warmup_output)

            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            input_buffer = self.input_buffers[grid_config]["pixel_values"]

            with (
                set_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                ),
                torch.cuda.graph(graph, self.pool),
            ):
                output = vision_encoder(input_buffer, grid_thw=grid_thw)
                self.output_buffers[grid_config].copy_(output)

        self.graphs[grid_config] = graph
        logger.debug(
            f"Captured encoder CUDA graph for grid config {grid_config} "
            f"-> {dummy_inputs['num_output_tokens']} output tokens"
            f"{' (with cached tensors)' if has_cudagraph_forward else ''}"
        )

    @torch.inference_mode()
    def capture(
        self,
        vision_encoder: nn.Module,
        embed_multimodal_fn: Callable,
    ) -> None:
        """
        Capture CUDA graphs for all configured grid configurations.

        Args:
            vision_encoder: The vision encoder module (e.g., Qwen3_VisionTransformer)
            embed_multimodal_fn: The model's embed_multimodal method (unused but kept for API)
        """
        if self.captured:
            logger.warning("Encoder CUDA graphs already captured, skipping")
            return

        # Log initial memory state
        free_mem_before, total_mem = torch.cuda.mem_get_info(self.device)
        used_mem_before = total_mem - free_mem_before
        logger.info(
            f"Capturing encoder CUDA graphs for {len(self.grid_configs)} "
            f"grid configurations (GPU memory: {used_mem_before / 1024**3:.2f} GiB used, "
            f"{free_mem_before / 1024**3:.2f} GiB free)"
        )

        # Capture from smallest to largest so that common smaller grids are
        # captured first. If we run out of memory, only large grids will fail.
        configs_to_capture = sorted(
            self.grid_configs,
            key=lambda x: x[0] * x[1] * x[2],
            reverse=False  # Smallest first
        )

        if is_global_first_rank():
            configs_to_capture = tqdm(
                configs_to_capture,
                desc="Capturing encoder CUDA graphs"
            )

        # Capture each graph in its own graph_capture context to isolate failures.
        # If one capture fails, the pool state won't affect subsequent captures.
        for grid_config in configs_to_capture:
            try:
                with graph_capture(device=self.device):
                    self.capture_graph_for_grid(
                        grid_config,
                        vision_encoder,
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to capture encoder CUDA graph for grid config "
                    f"{grid_config}: {e}. Will use eager mode."
                )

        self.captured = True

        # Log final memory state
        free_mem_after, _ = torch.cuda.mem_get_info(self.device)
        used_mem_after = total_mem - free_mem_after
        encoder_graph_mem = used_mem_after - used_mem_before
        logger.info(
            f"Captured {len(self.graphs)} encoder CUDA graphs "
            f"(configs: {sorted(self.graphs.keys())}). "
            f"Encoder graph memory: {encoder_graph_mem / 1024**3:.2f} GiB "
            f"(GPU: {used_mem_after / 1024**3:.2f} GiB used, "
            f"{free_mem_after / 1024**3:.2f} GiB free)"
        )

    def get_graph_for_grid(
        self,
        grid_thw: list[list[int]],
    ) -> tuple[int, int, int] | None:
        """
        Check if a CUDA graph is available for the given grid configuration.

        Args:
            grid_thw: List of [T, H, W] for each image

        Returns:
            The grid config key if a matching graph exists, None otherwise
        """
        key = self._grid_to_key(grid_thw)
        if key is None:
            return None
        return key if key in self.graphs else None

    def find_bucket_for_tokens(
        self,
        num_tokens: int,
        spatial_merge_size: int = 2,
    ) -> tuple[int, int, int] | None:
        """
        Find the smallest captured grid that can fit the given token count.

        This enables padded execution where inputs smaller than a bucket
        are padded to match the bucket size.

        Args:
            num_tokens: Number of output tokens needed
            spatial_merge_size: Merge size (default 2)

        Returns:
            Grid config (T, H, W) of the best bucket, or None if too large
        """
        best_grid = None
        best_bucket_tokens = float('inf')

        for grid_key in self.graphs.keys():
            bucket_tokens = self._compute_output_tokens(grid_key, spatial_merge_size)
            if bucket_tokens >= num_tokens and bucket_tokens < best_bucket_tokens:
                best_bucket_tokens = bucket_tokens
                best_grid = grid_key

        return best_grid

    def run(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor | None:
        """
        Run the vision encoder using a captured CUDA graph if available.

        Args:
            pixel_values: Input pixel values [num_patches, patch_channels]
            grid_thw: List of [T, H, W] for each image

        Returns:
            Vision encoder output tensor if graph was used, None if no matching graph
        """
        grid_key = self.get_graph_for_grid(grid_thw)

        if grid_key is None:
            # Don't count miss here - caller may try run_padded() next
            return None

        # Verify input dimensions match
        input_buffer = self.input_buffers[grid_key]["pixel_values"]
        if pixel_values.shape != input_buffer.shape:
            logger.warning(
                f"Pixel values shape mismatch: expected {input_buffer.shape}, "
                f"got {pixel_values.shape}. Falling back to eager mode."
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                f"Device mismatch: expected {input_buffer.device}, "
                f"got {pixel_values.device}. Falling back to eager mode."
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                f"Dtype mismatch: expected {input_buffer.dtype}, "
                f"got {pixel_values.dtype}. Falling back to eager mode."
            )
            self.eager_fallbacks += 1
            return None

        self.cache_hits += 1

        # Sync before modifying buffers: ensure any previous graph replay
        # (from a prior call) has completed. Without this, we could modify
        # buffers while a previous replay is still reading them.
        torch.cuda.synchronize()

        # Ensure contiguous memory layout for safe copy
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        # Copy input to the captured buffer (non-blocking for better overlap)
        input_buffer.copy_(pixel_values, non_blocking=True)

        # For exact match, restore cached embeddings (may have been modified by run_padded)
        if grid_key in self.embedding_buffers and grid_key in self.cached_tensors:
            embed_buffers = self.embedding_buffers[grid_key]
            cached = self.cached_tensors[grid_key]
            embed_buffers["pos_embeds"].copy_(cached["pos_embeds"], non_blocking=True)
            embed_buffers["rotary_pos_emb_cos"].copy_(
                cached["rotary_pos_emb_cos"], non_blocking=True)
            embed_buffers["rotary_pos_emb_sin"].copy_(
                cached["rotary_pos_emb_sin"], non_blocking=True)
            embed_buffers["cu_seqlens"].copy_(cached["cu_seqlens"], non_blocking=True)
            embed_buffers["max_seqlen"].copy_(cached["max_seqlen"], non_blocking=True)

        if self.verbose:
            logger.info(
                f"run(): grid_key={grid_key}, "
                f"input_shape={pixel_values.shape}, buffer_shape={input_buffer.shape}"
            )

        # Sync before replay: graph was captured on a separate stream, but buffer
        # modifications (copy) happen on the default stream. Without sync,
        # replay may start before copies complete.
        torch.cuda.synchronize()

        # Replay the graph
        self.graphs[grid_key].replay()

        # Return a clone of the output to avoid issues with buffer reuse
        return self.output_buffers[grid_key].clone()

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
        if self.vision_encoder is None or not hasattr(self.vision_encoder, 'precompute_for_cudagraph'):
            logger.debug("Vision encoder not available for padded mode")
            return None

        # Find the smallest bucket that fits
        bucket_grid = self.find_bucket_for_tokens(num_output_tokens, spatial_merge_size)
        if bucket_grid is None:
            # Don't count miss here - caller will count it when falling back to eager
            logger.debug(
                f"No bucket found for {num_output_tokens} tokens, "
                f"max available: {max(self._compute_output_tokens(g, spatial_merge_size) for g in self.graphs.keys()) if self.graphs else 0}"
            )
            return None

        # Check if we have embedding buffers for this bucket
        if bucket_grid not in self.embedding_buffers:
            logger.debug(f"No embedding buffers for bucket {bucket_grid}")
            return None

        bucket_tokens = self._compute_output_tokens(bucket_grid, spatial_merge_size)
        padding_waste = bucket_tokens - num_output_tokens

        # Get the input buffer for this bucket
        input_buffer = self.input_buffers[bucket_grid]["pixel_values"]
        num_input_patches = pixel_values.shape[0]
        bucket_input_patches = input_buffer.shape[0]

        if num_input_patches > bucket_input_patches:
            logger.warning(
                f"Input patches ({num_input_patches}) exceed bucket capacity "
                f"({bucket_input_patches}). This shouldn't happen."
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                f"Device mismatch: expected {input_buffer.device}, "
                f"got {pixel_values.device}. Falling back to eager mode."
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                f"Dtype mismatch: expected {input_buffer.dtype}, "
                f"got {pixel_values.dtype}. Falling back to eager mode."
            )
            self.eager_fallbacks += 1
            return None

        # Ensure contiguous memory layout for safe copy
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        self.cache_hits += 1

        # Sync before modifying buffers: ensure any previous graph replay
        # (from a prior call) has completed. Without this, we could zero/modify
        # buffers while a previous replay is still reading them.
        torch.cuda.synchronize()

        # === KEY FIX: Compute embeddings for ACTUAL grid, then pad ===
        # This ensures correct position embeddings for the actual input size
        actual_embeds = self.vision_encoder.precompute_for_cudagraph(grid_thw)

        # Get embedding buffers for the bucket
        embed_buffers = self.embedding_buffers[bucket_grid]

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
            actual_embeds["pos_embeds"], non_blocking=True)
        embed_buffers["rotary_pos_emb_cos"][:actual_num_patches].copy_(
            actual_embeds["rotary_pos_emb_cos"], non_blocking=True)
        embed_buffers["rotary_pos_emb_sin"][:actual_num_patches].copy_(
            actual_embeds["rotary_pos_emb_sin"], non_blocking=True)

        # Update cu_seqlens and max_seqlen to actual values
        # cu_seqlens shape is [num_images + 1], for single image it's [2]: [0, num_patches]
        # We copy the actual values so flash attention processes only the real tokens
        embed_buffers["cu_seqlens"].copy_(actual_embeds["cu_seqlens"], non_blocking=True)
        embed_buffers["max_seqlen"].copy_(actual_embeds["max_seqlen"], non_blocking=True)

        if self.verbose:
            logger.info(
                f"run_padded(): bucket_grid={bucket_grid}, "
                f"actual_grid={grid_thw[0]}, input_patches={num_input_patches}, "
                f"bucket_patches={bucket_input_patches}"
            )

        # Sync before replay: graph was captured on a separate stream, but buffer
        # modifications (zero, copy) happen on the default stream. Without sync,
        # replay may start before copies complete, reading zeros/partial data.
        torch.cuda.synchronize()

        # Replay the graph with updated embedding buffers
        self.graphs[bucket_grid].replay()

        # Get output and trim to actual size
        full_output = self.output_buffers[bucket_grid]
        trimmed_output = full_output[:num_output_tokens].clone()

        logger.debug(
            f"Padded execution: {num_output_tokens} -> {bucket_tokens} tokens "
            f"(waste: {padding_waste}, {padding_waste/bucket_tokens*100:.1f}%)"
        )

        return trimmed_output, padding_waste

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
                f"Encoder CUDA graph stats: "
                f"hits={self.cache_hits}, eager={self.eager_fallbacks}, "
                f"hit_rate={hit_rate:.1%}, num_graphs={len(self.graphs)}"
            )
        return stats



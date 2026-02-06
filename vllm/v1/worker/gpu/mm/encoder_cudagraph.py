# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA Graph Manager for Multimodal Encoders (ViT).

This module provides CUDA graph capture and replay functionality for vision
encoders to eliminate kernel launch overhead and improve GPU utilization.

Primary execution mode - Budget Batching:
- Captures CUDA graphs for multiple token budget levels (e.g., [2048, 4096,
  8192, 13824]), each with a fixed max_images_per_batch.
- At runtime, images are sorted smallest-first and greedily packed into
  budget-sized batches. The smallest fitting budget graph is selected.
- cu_seqlens is padded to max_images_per_batch + 1 by repeating the last
  value, creating zero-length sequences for empty slots (no-op in FA2/FA4).
- Works with any number of images (1 or many) and any grid sizes.

Key design principles:
1. Capture graphs based on token budgets, not grid sizes
2. Reuse one graph for any batch where total tokens fit the budget
3. Fall back to eager mode when no suitable graph is available
4. Track statistics for monitoring and optimization
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    graph_capture,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

# Grid configurations for CUDA graph capture (T, H, W in patch units)
#
# Top 100 most common grids for embedding cache pre-warming.
# Pre-warming these grids at startup avoids cold-start embedding computation
# at runtime, eliminating ~20 small kernel launches per grid on first encounter.
# Based on MLPerf VLM dataset analysis (~71% coverage with top 100 grids).
EMBEDDING_WARMUP_GRIDS: list[tuple[int, int, int]] = [
    # Top 50 grids (sorted by frequency)
    (1, 62, 62),
    (1, 32, 32),
    (1, 50, 50),
    (1, 38, 38),
    (1, 76, 76),
    (1, 94, 94),
    (1, 64, 64),
    (1, 124, 124),
    (1, 68, 68),
    (1, 100, 100),
    (1, 16, 16),
    (1, 24, 24),
    (1, 46, 46),
    (1, 44, 44),
    (1, 42, 42),
    (1, 40, 40),
    (1, 56, 56),
    (1, 128, 128),
    (1, 18, 18),
    (1, 28, 28),
    (1, 34, 34),
    (1, 80, 80),
    (1, 30, 30),
    (1, 38, 50),
    (1, 22, 22),
    (1, 112, 112),
    (1, 36, 36),
    (1, 34, 50),
    (1, 188, 188),
    (1, 14, 20),
    (1, 90, 90),
    (1, 44, 42),
    (1, 16, 18),
    (1, 54, 54),
    (1, 48, 48),
    (1, 40, 42),
    (1, 60, 60),
    (1, 88, 88),
    (1, 26, 26),
    (1, 156, 156),
    (1, 94, 62),
    (1, 30, 38),
    (1, 24, 38),
    (1, 20, 20),
    (1, 24, 16),
    (1, 18, 16),
    (1, 120, 120),
    (1, 60, 80),
    (1, 52, 52),
    (1, 66, 66),
    # Next 50 grids
    (1, 20, 14),
    (1, 24, 32),
    (1, 160, 160),
    (1, 28, 38),
    (1, 30, 40),
    (1, 38, 42),
    (1, 58, 58),
    (1, 20, 32),
    (1, 50, 38),
    (1, 48, 64),
    (1, 78, 78),
    (1, 24, 20),
    (1, 42, 62),
    (1, 62, 94),
    (1, 36, 42),
    (1, 32, 20),
    (1, 150, 150),
    (1, 50, 42),
    (1, 50, 76),
    (1, 72, 72),
    (1, 32, 24),
    (1, 46, 42),
    (1, 92, 94),
    (1, 82, 82),
    (1, 32, 38),
    (1, 90, 94),
    (1, 14, 22),
    (1, 76, 100),
    (1, 94, 92),
    (1, 24, 18),
    (1, 54, 42),
    (1, 38, 32),
    (1, 18, 24),
    (1, 28, 32),
    (1, 30, 42),
    (1, 56, 76),
    (1, 62, 42),
    (1, 28, 50),
    (1, 32, 42),
    (1, 36, 50),
    (1, 38, 24),
    (1, 108, 82),
    (1, 16, 20),
    (1, 26, 38),
    (1, 38, 36),
    (1, 34, 42),
    (1, 76, 50),
    (1, 38, 56),
    (1, 48, 42),
    (1, 30, 32),
]


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
        graph_pool: Any | None = None,
        verbose: bool = False,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        # CUDA graph storage - keyed by (batch_size, t, h, w) tuple
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

        # Input buffers for embeddings (padded mode with runtime computation)
        # Key: (batch_size, t, h, w), Value: dict with pos_embeds, rotary, cu_seqlens
        self.embedding_buffers: dict[
            tuple[int, int, int, int], dict[str, torch.Tensor]
        ] = {}

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

        # Per-grid embedding cache for batched contiguous mode
        # Key: (t, h, w), Value: dict with pos_embeds, rotary_cos, rotary_sin
        # This avoids recomputing embeddings at runtime - just look up and concat
        self.grid_embedding_cache: dict[
            tuple[int, int, int], dict[str, torch.Tensor]
        ] = {}

        # Budget batching config
        # Maps token_budget -> graph_key for budget batch CUDA graphs
        self.budget_graph_keys: dict[int, tuple[int, int, int, int]] = {}
        self.token_budgets: list[int] = []
        self.max_images_per_batch: int = 0
        self._read_budget_config()

    def _read_budget_config(self) -> None:
        """Read budget batching configuration from compilation config."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return

        token_budgets = getattr(
            compilation_config, "encoder_cudagraph_token_budgets", None
        )
        max_images = getattr(
            compilation_config, "encoder_cudagraph_max_images_per_batch", None
        )

        if token_budgets is None and max_images is None:
            return

        if (token_budgets is None) != (max_images is None):
            logger.warning(
                "encoder_cudagraph_token_budgets and "
                "encoder_cudagraph_max_images_per_batch must both be set. "
                "Budget batching disabled."
            )
            return

        if token_budgets is None or max_images is None:
            return

        if max_images <= 0:
            logger.warning(
                "encoder_cudagraph_max_images_per_batch must be positive. "
                "Budget batching disabled."
            )
            return

        bad_budgets = [b for b in token_budgets if b % max_images != 0]
        if bad_budgets:
            logger.warning(
                "encoder_cudagraph_token_budgets values %s are not divisible "
                "by max_images_per_batch=%d. Budget batching disabled.",
                bad_budgets,
                max_images,
            )
            return

        self.token_budgets = sorted(token_budgets)
        self.max_images_per_batch = max_images

        logger.info(
            "Budget batching configured: token_budgets=%s, max_images_per_batch=%d",
            self.token_budgets,
            self.max_images_per_batch,
        )

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
            graph_key,
            batch_size,
            grid_config,
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

        # Store vision encoder reference for runtime embedding computation
        self.vision_encoder = vision_encoder

        # Check if vision encoder supports optimized CUDA graph forward
        has_cudagraph_forward = hasattr(
            vision_encoder, "forward_cudagraph"
        ) and hasattr(vision_encoder, "precompute_for_cudagraph")

        if has_cudagraph_forward:
            cached = vision_encoder.precompute_for_cudagraph(grid_thw)

            # Cache per-grid embeddings for batched contiguous mode
            # This avoids recomputing embeddings at runtime - just lookup and concat
            grid_key = (t, h, w)
            if grid_key not in self.grid_embedding_cache:
                # Compute embeddings for a single image of this grid size
                single_cached = vision_encoder.precompute_for_cudagraph([[t, h, w]])
                self.grid_embedding_cache[grid_key] = {
                    "pos_embeds": single_cached["pos_embeds"],
                    "rotary_pos_emb_cos": single_cached["rotary_pos_emb_cos"],
                    "rotary_pos_emb_sin": single_cached["rotary_pos_emb_sin"],
                }
                logger.debug(
                    "Cached per-grid embeddings for grid %s: pos_embeds=%s",
                    grid_key,
                    single_cached["pos_embeds"].shape,
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

    def capture_budget_graphs(self, vision_encoder: nn.Module) -> None:
        """
        Capture CUDA graphs for budget batching mode.

        For each configured token_budget, captures a graph with
        max_images_per_batch image slots. The graph uses a synthetic grid
        that produces the right tensor shapes. At runtime, embedding buffers
        are overwritten with actual per-image values from grid_embedding_cache.

        Args:
            vision_encoder: The vision encoder module
        """
        if not self.token_budgets or self.max_images_per_batch <= 0:
            return

        merge = getattr(vision_encoder, "spatial_merge_size", 2)

        for token_budget in self.token_budgets:
            per_image_output = token_budget // self.max_images_per_batch
            if per_image_output <= 0:
                logger.warning(
                    "token_budget=%d too small for max_images=%d, skipping",
                    token_budget,
                    self.max_images_per_batch,
                )
                continue

            # Synthetic grid: (1, merge, per_image_output * merge)
            # Output tokens per image:
            # 1 * (merge/merge) * (per_image_output*merge/merge)
            # = per_image_output
            # Total output = max_images * per_image_output = token_budget
            grid_config = (1, merge, per_image_output * merge)

            try:
                if self.is_single_gpu:
                    self.capture_graph_for_grid(
                        grid_config,
                        vision_encoder,
                        batch_size=self.max_images_per_batch,
                    )
                else:
                    with graph_capture(device=self.device):
                        self.capture_graph_for_grid(
                            grid_config,
                            vision_encoder,
                            batch_size=self.max_images_per_batch,
                        )

                graph_key = (
                    self.max_images_per_batch,
                    1,
                    merge,
                    per_image_output * merge,
                )
                self.budget_graph_keys[token_budget] = graph_key
                logger.info(
                    "Captured budget graph: token_budget=%d, "
                    "max_images=%d, graph_key=%s",
                    token_budget,
                    self.max_images_per_batch,
                    graph_key,
                )
            except Exception as e:
                logger.warning(
                    "Failed to capture budget graph for token_budget=%d: %s",
                    token_budget,
                    e,
                )

    def find_budget_graph(
        self,
        total_output_tokens: int,
    ) -> tuple[int, int, int, int] | None:
        """
        Find the smallest budget graph that fits the given total output tokens.

        Args:
            total_output_tokens: Total output tokens for the packed batch

        Returns:
            Graph key (batch_size, t, h, w) or None if no budget fits
        """
        best_key = None
        best_budget = float("inf")

        for budget, graph_key in self.budget_graph_keys.items():
            if budget >= total_output_tokens and budget < best_budget:
                best_budget = budget
                best_key = graph_key

        return best_key

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

        # Pre-warm embedding cache for common grids
        self._prewarm_embedding_cache(vision_encoder)

        # Capture budget batch graphs
        if self.token_budgets and self.max_images_per_batch > 0:
            self.capture_budget_graphs(vision_encoder)

        self.captured = True

    def _prewarm_embedding_cache(self, vision_encoder: nn.Module) -> None:
        """
        Pre-warm the embedding cache for common grid configurations.

        This avoids cold-start embedding computation at runtime by pre-computing
        embeddings for the top 100 most common grids. Each grid that would
        otherwise trigger ~20 small kernel launches on first encounter will
        instead hit the cache.

        Args:
            vision_encoder: The vision encoder module with precompute_for_cudagraph
        """
        if not hasattr(vision_encoder, "precompute_for_cudagraph"):
            logger.debug(
                "Vision encoder lacks precompute_for_cudagraph, skipping warmup"
            )
            return

        # Filter out grids that are already cached (from graph capture)
        grids_to_warm = [
            g for g in EMBEDDING_WARMUP_GRIDS if g not in self.grid_embedding_cache
        ]

        if not grids_to_warm:
            logger.debug("All warmup grids already cached")
            return

        if self.verbose:
            logger.info(
                "Pre-warming embedding cache for %d grids (%d already cached)",
                len(grids_to_warm),
                len(EMBEDDING_WARMUP_GRIDS) - len(grids_to_warm),
            )

        for grid in grids_to_warm:
            t, h, w = grid
            try:
                cached = vision_encoder.precompute_for_cudagraph([[t, h, w]])
                self.grid_embedding_cache[grid] = {
                    "pos_embeds": cached["pos_embeds"],
                    "rotary_pos_emb_cos": cached["rotary_pos_emb_cos"],
                    "rotary_pos_emb_sin": cached["rotary_pos_emb_sin"],
                }
            except Exception as e:
                logger.debug("Failed to pre-warm grid %s: %s", grid, e)

        # Calculate and log embedding cache memory consumption
        if self.verbose:
            cache_memory_bytes = self._compute_embedding_cache_memory()
            logger.info(
                "Embedding cache warmed: %d grids total, memory: %.2f MiB",
                len(self.grid_embedding_cache),
                cache_memory_bytes / (1024 * 1024),
            )

    def _compute_embedding_cache_memory(self) -> int:
        """
        Compute the total GPU memory consumption of the embedding cache.

        Returns:
            Total memory in bytes used by all cached embeddings.
        """
        total_bytes = 0
        for grid, cached in self.grid_embedding_cache.items():
            for key, tensor in cached.items():
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

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
        num_actual_images = len(grid_thw_list)
        is_budget_graph = graph_key in self.budget_graph_keys.values()

        if num_actual_images > batch_size:
            logger.warning(
                "grid_thw_list length (%d) exceeds graph batch_size (%d)",
                num_actual_images,
                batch_size,
            )
            return None

        if num_actual_images != batch_size and not is_budget_graph:
            logger.warning(
                "grid_thw_list length (%d) doesn't match graph batch_size (%d)"
                " and not a budget graph",
                num_actual_images,
                batch_size,
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
                actual_input_patches,
                bucket_input_patches,
            )
            self.eager_fallbacks += 1
            return None

        # Verify device and dtype match
        if pixel_values.device != input_buffer.device:
            logger.warning(
                "Device mismatch: expected %s, got %s.",
                input_buffer.device,
                pixel_values.device,
            )
            self.eager_fallbacks += 1
            return None

        if pixel_values.dtype != input_buffer.dtype:
            logger.warning(
                "Dtype mismatch: expected %s, got %s.",
                input_buffer.dtype,
                pixel_values.dtype,
            )
            self.eager_fallbacks += 1
            return None

        # Ensure contiguous memory layout
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        # Count actual images processed (for accurate hit rate)
        self.cache_hits += num_actual_images

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

        # Look up cached embeddings for each grid and pack contiguously
        # This avoids expensive per-image precompute_for_cudagraph calls
        pos_embeds_list = []
        rotary_cos_list = []
        rotary_sin_list = []
        sequence_lengths = []
        cache_miss_grids: list[tuple[int, int, int]] = []

        for grid in grid_thw_list:
            t, h, w = grid
            grid_key = (t, h, w)
            # Each temporal frame is a separate attention sequence in patch space.
            # This matches the eager path: np.repeat(h*w, t) per image.
            for _ in range(t):
                sequence_lengths.append(h * w)

            # Try to use cached embeddings (populated during graph capture)
            if grid_key in self.grid_embedding_cache:
                cached = self.grid_embedding_cache[grid_key]
                pos_embeds_list.append(cached["pos_embeds"])
                rotary_cos_list.append(cached["rotary_pos_emb_cos"])
                rotary_sin_list.append(cached["rotary_pos_emb_sin"])
            else:
                # Cache miss - compute on-the-fly but don't cache
                # (avoids unbounded GPU memory growth at runtime)
                cache_miss_grids.append(grid_key)
                if self.vision_encoder is not None:
                    actual_embeds = self.vision_encoder.precompute_for_cudagraph([grid])
                    pos_embeds_list.append(actual_embeds["pos_embeds"])
                    rotary_cos_list.append(actual_embeds["rotary_pos_emb_cos"])
                    rotary_sin_list.append(actual_embeds["rotary_pos_emb_sin"])
                else:
                    logger.warning("Grid %s not cached and no vision encoder", grid_key)
                    return None

        if cache_miss_grids and self.verbose:
            logger.info(
                "Embedding cache miss for grids: %s (computed on-the-fly)",
                cache_miss_grids,
            )

        # Concatenate cached embeddings (just tensor concat, no computation)
        packed_pos_embeds = torch.cat(pos_embeds_list, dim=0)
        packed_rotary_cos = torch.cat(rotary_cos_list, dim=0)
        packed_rotary_sin = torch.cat(rotary_sin_list, dim=0)

        # Copy packed embeddings to buffer (padding remains zero at end)
        actual_embed_len = packed_pos_embeds.shape[0]
        embed_buffers["pos_embeds"][:actual_embed_len].copy_(
            packed_pos_embeds, non_blocking=True
        )
        embed_buffers["rotary_pos_emb_cos"][:actual_embed_len].copy_(
            packed_rotary_cos, non_blocking=True
        )
        embed_buffers["rotary_pos_emb_sin"][:actual_embed_len].copy_(
            packed_rotary_sin, non_blocking=True
        )

        # Build cu_seqlens from actual cumulative sizes
        # cu_seqlens = [0, size0, size0+size1, ..., total_actual]
        cu_seqlens_list = [0]
        for length in sequence_lengths:
            cu_seqlens_list.append(cu_seqlens_list[-1] + length)

        # For budget graphs: pad cu_seqlens to batch_size + 1 by repeating
        # the last value. This creates zero-length sequences for empty slots
        # that flash attention skips (no-op).
        # Note: num_sequences = sum(t_i) for all images. For images (t=1),
        # this equals num_images <= batch_size. For videos (t>1), it could
        # exceed batch_size — fall back to eager in that case.
        if is_budget_graph and len(sequence_lengths) > batch_size:
            logger.debug(
                "Too many sequences (%d) for budget graph batch_size (%d), "
                "falling back to eager",
                len(sequence_lengths),
                batch_size,
            )
            return None

        is_flashinfer = (
            self.vllm_config.model_config.multimodal_config.mm_encoder_attn_backend
            == AttentionBackendEnum.FLASHINFER
        )
        if is_budget_graph and len(cu_seqlens_list) < batch_size + 1:
            last_val = cu_seqlens_list[-1]
            while len(cu_seqlens_list) < batch_size + 1:
                cu_seqlens_list.append(last_val)
            if is_flashinfer:
                hidden_size = (
                    self.vllm_config.model_config.hf_config.vision_config.hidden_size
                )
                use_data_parallel = (
                    self.vllm_config.model_config.multimodal_config.mm_encoder_tp_mode
                    == "data"
                    if self.vllm_config.model_config.multimodal_config
                    else False
                )
                tp_size = (
                    1 if use_data_parallel else get_tensor_model_parallel_world_size()
                )
                scale = hidden_size // tp_size
                cu_seqlens_qk = [
                    cu_seqlens_list[i] * scale * 2 for i in range(len(cu_seqlens_list))
                ]
                cu_seqlens_v = [
                    cu_seqlens_list[i] * scale * 3 for i in range(len(cu_seqlens_list))
                ]
                cu_seqlens_o = [
                    cu_seqlens_list[i] * scale for i in range(len(cu_seqlens_list))
                ]
                cu_seqlens_list = cu_seqlens_qk + cu_seqlens_v + cu_seqlens_o

        # For budget graphs: pad sequence_lengths with zeros for empty slots
        if is_budget_graph and len(sequence_lengths) < batch_size:
            sequence_lengths = list(sequence_lengths) + [0] * (
                batch_size - len(sequence_lengths)
            )

        cu_seqlens_tensor = torch.tensor(
            cu_seqlens_list, dtype=torch.int32, device=self.device
        )
        max_seqlen = (
            max(s for s in sequence_lengths if s > 0) if sequence_lengths else 0
        )
        max_seqlen_tensor = torch.tensor(max_seqlen, dtype=torch.int32, device="cpu")
        sequence_lengths_tensor = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=self.device
        )

        # Copy full cu_seqlens and sequence_lengths to buffers
        # For budget graphs, sizes match exactly (padded to batch_size + 1).
        # For non-budget graphs, copy only the actual part.
        cu_seqlens_buf = embed_buffers["cu_seqlens"]
        seq_len_buf = embed_buffers["sequence_lengths"]
        if is_budget_graph:
            cu_seqlens_buf.copy_(cu_seqlens_tensor, non_blocking=True)
            seq_len_buf.copy_(sequence_lengths_tensor, non_blocking=True)
        else:
            cu_seqlens_buf[: len(cu_seqlens_list)].copy_(
                cu_seqlens_tensor, non_blocking=True
            )
            seq_len_buf[:batch_size].copy_(sequence_lengths_tensor, non_blocking=True)
        embed_buffers["max_seqlen"].copy_(max_seqlen_tensor, non_blocking=True)

        if self.verbose:
            logger.info(
                "run_batched_contiguous(): graph_key=%s, grids=%s, "
                "actual_patches=%d, bucket_patches=%d, cu_seqlens=%s",
                graph_key,
                grid_thw_list,
                actual_input_patches,
                bucket_input_patches,
                cu_seqlens_list,
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

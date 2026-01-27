# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA Graph Manager for Multimodal Encoders (ViT).

This module provides CUDA graph capture and replay functionality for vision
encoders to eliminate kernel launch overhead and improve GPU utilization.

Key design principles:
1. Capture graphs for specific grid_thw configurations (not just token counts)
2. Only replay when input dimensions exactly match captured configuration
3. Fall back to eager mode for non-matching inputs
4. Track statistics for monitoring and optimization

Limitations:
- CUDA graphs are only used when input dimensions exactly match captured graphs
- Variable-size images that don't match any captured configuration use eager mode
- Multiple images in a batch are processed sequentially through graph replay
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)

# Default grid configurations to capture (T, H, W in patch units)
# These are common configurations for Qwen-VL models after smart_resize
# Format: (temporal, height_patches, width_patches)
DEFAULT_ENCODER_GRID_CONFIGS = [
    # Common single-frame image configurations (T=1)
    # After smart_resize with factor=28 (patch=14, merge=2), common sizes:
    (1, 16, 16),   # ~224x224 -> 64 output tokens
    (1, 24, 24),   # ~336x336 -> 144 output tokens
    (1, 32, 32),   # ~448x448 -> 256 output tokens
    (1, 48, 48),   # ~672x672 -> 576 output tokens
    (1, 64, 64),   # ~896x896 -> 1024 output tokens
    (1, 80, 80),   # ~1120x1120 -> 1600 output tokens
    (1, 96, 96),   # ~1344x1344 -> 2304 output tokens
]

# Optimized grid configurations for MLPerf Shopify dataset
# Based on analysis: 96% of images have 4000-8200 output tokens
# Using square grids that cover the common token ranges with padding
SHOPIFY_OPTIMIZED_GRID_CONFIGS = [
    # Small images (rare, <5% of dataset)
    (1, 64, 64),    # 1024 tokens - covers up to ~1024 tokens
    (1, 80, 80),    # 1600 tokens
    (1, 96, 96),    # 2304 tokens
    (1, 112, 112),  # 3136 tokens
    # Main distribution (95% of dataset: 4000-8200 tokens)
    (1, 128, 128),  # 4096 tokens - covers P10 (4646)
    (1, 144, 144),  # 5184 tokens - covers ~P25 (5351)
    (1, 160, 160),  # 6400 tokens - covers ~P50-P75 (6072-6904)
    (1, 176, 176),  # 7744 tokens - covers ~P90 (7948)
    (1, 184, 184),  # 8464 tokens - covers max (8161)
]

# Alternative: Rectangular grids for better aspect ratio coverage
SHOPIFY_RECTANGULAR_GRID_CONFIGS = [
    # 4:3 and 3:4 aspect ratios for product images
    (1, 128, 128),  # 4096 tokens (square)
    (1, 112, 144),  # 4032 tokens (3:4)
    (1, 144, 112),  # 4032 tokens (4:3)
    (1, 144, 144),  # 5184 tokens (square)
    (1, 128, 160),  # 5120 tokens (4:5)
    (1, 160, 128),  # 5120 tokens (5:4)
    (1, 160, 160),  # 6400 tokens (square)
    (1, 144, 176),  # 6336 tokens (9:11)
    (1, 176, 144),  # 6336 tokens (11:9)
    (1, 176, 176),  # 7744 tokens (square)
    (1, 160, 192),  # 7680 tokens (5:6)
    (1, 192, 160),  # 7680 tokens (6:5)
    (1, 184, 184),  # 8464 tokens (square, max)
]

# Legacy bucket sizes for backward compatibility
DEFAULT_ENCODER_CUDAGRAPH_BUCKET_SIZES = [
    64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192
]

# =============================================================================
# TOKEN BUCKET PRESETS FOR PADDED CUDA GRAPHS
# =============================================================================
# These define output token buckets. Inputs are padded to the smallest bucket
# that fits, trading padding overhead for CUDA graph utilization.
#
# For Shopify dataset analysis:
#   - P10=4646, P25=5351, P50=6072, P75=6904, P90=7948, Max=8161 tokens

# Fine-grained buckets: More buckets = less padding waste, more GPU memory
SHOPIFY_TOKEN_BUCKETS_FINE = [
    1024,   # Small images (<5% of dataset)
    2048,
    3072,
    4096,   # ~P10
    4608,
    5120,   # ~P25
    5632,
    6144,   # ~P50
    6656,
    7168,   # ~P75
    7680,
    8192,   # ~P90-P99
    8464,   # Max coverage
]

# Medium granularity: Balanced tradeoff
SHOPIFY_TOKEN_BUCKETS_MEDIUM = [
    1024,   # Small images
    2048,
    3072,
    4096,   # Covers up to P10
    5120,   # Covers P10-P25
    6144,   # Covers P25-P50
    7168,   # Covers P50-P75
    8192,   # Covers P75-P99
    8464,   # Max coverage
]

# Coarse buckets: Fewer graphs, more padding waste
SHOPIFY_TOKEN_BUCKETS_COARSE = [
    2048,   # Small images
    4096,   # Up to ~P10
    6144,   # P10-P50
    8192,   # P50-P99
    8464,   # Max
]

# Single bucket: Maximum CUDA graph utilization, maximum padding
SHOPIFY_TOKEN_BUCKETS_SINGLE = [
    8464,   # All images padded to max
]


def token_bucket_to_grid(token_bucket: int, merge_size: int = 2) -> tuple[int, int, int]:
    """
    Convert a token bucket size to a square grid configuration.

    Args:
        token_bucket: Number of output tokens (after spatial merge)
        merge_size: Spatial merge size (default 2 for Qwen-VL)

    Returns:
        Grid config (T, H_patches, W_patches)
    """
    # For square grid: tokens = (H/merge)^2, so H = merge * sqrt(tokens)
    side = int(math.ceil(math.sqrt(token_bucket))) * merge_size
    return (1, side, side)


def get_grid_configs_from_token_buckets(
    token_buckets: list[int],
    merge_size: int = 2,
) -> list[tuple[int, int, int]]:
    """Convert token bucket list to grid configurations."""
    return [token_bucket_to_grid(t, merge_size) for t in token_buckets]


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
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = dtype

        # Get grid configs from config or use defaults
        if grid_configs is None:
            grid_configs = self._get_grid_configs_from_config()
        self.grid_configs = grid_configs

        # Legacy bucket sizes (for backward compatibility with bucket-based API)
        if bucket_sizes is None:
            bucket_sizes = self._get_bucket_sizes_from_config()
        self.bucket_sizes = sorted(bucket_sizes)

        # CUDA graph storage - keyed by (t, h, w) tuple
        self.graphs: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

        # Pre-allocated input/output buffers per grid config
        # Key: (t, h, w), Value: {"pixel_values": tensor, "grid_thw": list}
        self.input_buffers: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.output_buffers: dict[tuple[int, int, int], torch.Tensor] = {}

        # Store metadata about captured graphs
        self.captured_metadata: dict[tuple[int, int, int], dict[str, Any]] = {}

        # Track if graphs have been captured
        self.captured = False

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.eager_fallbacks = 0

    def _get_grid_configs_from_config(self) -> list[tuple[int, int, int]]:
        """Get encoder grid configurations from config or use defaults."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return DEFAULT_ENCODER_GRID_CONFIGS

        # Check for token bucket config first (new preferred way)
        token_buckets = getattr(
            compilation_config,
            'encoder_cudagraph_token_buckets',
            None
        )
        if token_buckets is not None:
            # Handle preset names for token buckets
            if isinstance(token_buckets, str):
                bucket_presets = {
                    "shopify_fine": SHOPIFY_TOKEN_BUCKETS_FINE,
                    "shopify_medium": SHOPIFY_TOKEN_BUCKETS_MEDIUM,
                    "shopify_coarse": SHOPIFY_TOKEN_BUCKETS_COARSE,
                    "shopify_single": SHOPIFY_TOKEN_BUCKETS_SINGLE,
                }
                if token_buckets in bucket_presets:
                    buckets = bucket_presets[token_buckets]
                    logger.info(
                        f"Using token bucket preset '{token_buckets}': {buckets}"
                    )
                    return get_grid_configs_from_token_buckets(buckets)
                else:
                    logger.warning(
                        f"Unknown token bucket preset '{token_buckets}', "
                        f"available: {list(bucket_presets.keys())}"
                    )
            elif isinstance(token_buckets, list):
                logger.info(f"Using custom token buckets: {token_buckets}")
                return get_grid_configs_from_token_buckets(token_buckets)

        # Check for encoder-specific grid config
        grid_configs = getattr(
            compilation_config,
            'encoder_cudagraph_grid_configs',
            None
        )
        if grid_configs is not None:
            # Handle preset names
            if isinstance(grid_configs, str):
                if grid_configs == "shopify":
                    return SHOPIFY_OPTIMIZED_GRID_CONFIGS
                elif grid_configs == "shopify_rectangular":
                    return SHOPIFY_RECTANGULAR_GRID_CONFIGS
                elif grid_configs == "default":
                    return DEFAULT_ENCODER_GRID_CONFIGS
                else:
                    logger.warning(
                        f"Unknown grid config preset '{grid_configs}', "
                        "using default"
                    )
                    return DEFAULT_ENCODER_GRID_CONFIGS
            return [tuple(cfg) for cfg in grid_configs]

        return DEFAULT_ENCODER_GRID_CONFIGS

    def _get_bucket_sizes_from_config(self) -> list[int]:
        """Get encoder CUDA graph bucket sizes from config or use defaults."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return DEFAULT_ENCODER_CUDAGRAPH_BUCKET_SIZES

        encoder_sizes = getattr(
            compilation_config,
            'encoder_cudagraph_bucket_sizes',
            None
        )
        if encoder_sizes is not None:
            return encoder_sizes

        return DEFAULT_ENCODER_CUDAGRAPH_BUCKET_SIZES

    def get_padded_size(self, num_visual_tokens: int) -> int | None:
        """
        Find the smallest bucket size >= num_visual_tokens.

        Returns None if the input is larger than all buckets.
        Note: This is for backward compatibility. For actual graph lookup,
        use get_graph_for_grid() instead.
        """
        for bucket_size in self.bucket_sizes:
            if num_visual_tokens <= bucket_size:
                return bucket_size
        return None

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

    def find_best_grid_for_padding(
        self,
        grid_thw: list[list[int]],
        spatial_merge_size: int = 2,
    ) -> tuple[int, int, int] | None:
        """
        Find the smallest captured grid that can accommodate the input with padding.

        For CUDA graph compatibility with variable-size inputs, this finds the
        smallest captured configuration where:
        - T_captured >= T_input
        - H_captured >= H_input
        - W_captured >= W_input

        Args:
            grid_thw: Input grid configuration [[T, H, W]]
            spatial_merge_size: Merge size for spatial dimensions (default 2)

        Returns:
            The best matching captured grid config, or None if no match found
        """
        key = self._grid_to_key(grid_thw)
        if key is None:
            return None

        t_in, h_in, w_in = key

        # First check for exact match
        if key in self.graphs:
            return key

        # Find smallest captured grid that can accommodate input
        best_match = None
        best_waste = float('inf')

        for captured_key in self.graphs.keys():
            t_cap, h_cap, w_cap = captured_key

            # Check if captured grid can accommodate input
            if t_cap >= t_in and h_cap >= h_in and w_cap >= w_in:
                # Calculate waste (padding overhead)
                input_tokens = self._compute_output_tokens(key, spatial_merge_size)
                captured_tokens = self._compute_output_tokens(
                    captured_key, spatial_merge_size
                )
                waste = captured_tokens - input_tokens

                if waste < best_waste:
                    best_waste = waste
                    best_match = captured_key

        if best_match is not None:
            logger.debug(
                f"Found padding-compatible grid: input={key} -> captured={best_match} "
                f"(waste={best_waste} tokens)"
            )

        return best_match

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

        # Create dummy pixel values
        pixel_values = torch.randn(
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

        # Warmup run (required before capture)
        with torch.cuda.stream(torch.cuda.current_stream()):
            warmup_output = vision_encoder(pixel_values, grid_thw=grid_thw)

            # Allocate output buffer based on actual output shape
            self.output_buffers[grid_config] = torch.empty_like(warmup_output)

        torch.cuda.synchronize()

        # Capture the graph
        graph = torch.cuda.CUDAGraph()

        # Get a fresh reference to the input buffer for capture
        input_buffer = self.input_buffers[grid_config]["pixel_values"]

        with torch.cuda.graph(graph, pool=self.pool):
            output = vision_encoder(input_buffer, grid_thw=grid_thw)
            self.output_buffers[grid_config].copy_(output)

        self.graphs[grid_config] = graph
        logger.debug(
            f"Captured encoder CUDA graph for grid config {grid_config} "
            f"-> {dummy_inputs['num_output_tokens']} output tokens"
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

        logger.info(
            f"Capturing encoder CUDA graphs for {len(self.grid_configs)} "
            f"grid configurations"
        )

        # Capture from largest to smallest (more memory efficient)
        configs_to_capture = sorted(
            self.grid_configs,
            key=lambda x: x[0] * x[1] * x[2],
            reverse=True
        )

        if is_global_first_rank():
            configs_to_capture = tqdm(
                configs_to_capture,
                desc="Capturing encoder CUDA graphs"
            )

        with graph_capture(device=self.device):
            for grid_config in configs_to_capture:
                try:
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
        logger.info(
            f"Captured {len(self.graphs)} encoder CUDA graphs "
            f"(configs: {sorted(self.graphs.keys())})"
        )

    def can_use_graph(self, num_visual_tokens: int) -> bool:
        """
        Check if a CUDA graph might be available for the given token count.

        Note: This is a heuristic check. Actual graph usage depends on
        exact grid_thw match via get_graph_for_grid().
        """
        if not self.captured:
            return False
        padded_size = self.get_padded_size(num_visual_tokens)
        return padded_size is not None

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
            self.cache_misses += 1
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

        self.cache_hits += 1

        # Copy input to the captured buffer
        input_buffer.copy_(pixel_values)

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

        This method pads the input to match a captured CUDA graph bucket,
        executes the graph, and returns the trimmed output.

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

        # Find the smallest bucket that fits
        bucket_grid = self.find_bucket_for_tokens(num_output_tokens, spatial_merge_size)
        if bucket_grid is None:
            self.cache_misses += 1
            logger.debug(
                f"No bucket found for {num_output_tokens} tokens, "
                f"max available: {max(self._compute_output_tokens(g, spatial_merge_size) for g in self.graphs.keys()) if self.graphs else 0}"
            )
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

        self.cache_hits += 1

        # Zero the buffer first (for clean padding)
        input_buffer.zero_()

        # Copy actual input to the beginning of the buffer
        input_buffer[:num_input_patches].copy_(pixel_values)

        # Replay the graph (uses the bucket's grid_thw for position embeddings)
        self.graphs[bucket_grid].replay()

        # Get output and trim to actual size
        full_output = self.output_buffers[bucket_grid]
        trimmed_output = full_output[:num_output_tokens].clone()

        logger.debug(
            f"Padded execution: {num_output_tokens} -> {bucket_tokens} tokens "
            f"(waste: {padding_waste}, {padding_waste/bucket_tokens*100:.1f}%)"
        )

        return trimmed_output, padding_waste

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses + self.eager_fallbacks
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "eager_fallbacks": self.eager_fallbacks,
            "hit_rate": hit_rate,
            "num_graphs": len(self.graphs),
            "captured_configs": sorted(self.graphs.keys()),
        }


def get_encoder_cudagraph_bucket_sizes(
    max_visual_tokens: int,
    min_bucket: int = 64,
    growth_factor: float = 1.5,
) -> list[int]:
    """
    Generate bucket sizes for encoder CUDA graphs.

    Uses exponential growth to cover the range [min_bucket, max_visual_tokens]
    with reasonable granularity.

    Args:
        max_visual_tokens: Maximum number of visual tokens to support
        min_bucket: Minimum bucket size
        growth_factor: Multiplier for each successive bucket

    Returns:
        List of bucket sizes
    """
    buckets = []
    current = min_bucket

    while current <= max_visual_tokens:
        buckets.append(int(current))
        current = int(current * growth_factor)

    # Ensure max is included
    if buckets[-1] < max_visual_tokens:
        buckets.append(max_visual_tokens)

    return buckets


def generate_grid_configs_for_resolution_range(
    min_size: int = 448,
    max_size: int = 1344,
    step: int = 224,
    patch_size: int = 14,
    temporal_values: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Generate grid configurations for a range of image resolutions.

    Args:
        min_size: Minimum image dimension in pixels
        max_size: Maximum image dimension in pixels
        step: Step size in pixels
        patch_size: Patch size of the vision encoder
        temporal_values: List of temporal dimensions to include (default [1])

    Returns:
        List of (T, H, W) tuples in patch units
    """
    if temporal_values is None:
        temporal_values = [1]

    configs = []
    for h_pixels in range(min_size, max_size + 1, step):
        for w_pixels in range(min_size, max_size + 1, step):
            h_patches = h_pixels // patch_size
            w_patches = w_pixels // patch_size
            for t in temporal_values:
                configs.append((t, h_patches, w_patches))

    return configs

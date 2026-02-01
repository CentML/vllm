# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.mm.encoder_cudagraph import EncoderCudaGraphManager

logger = init_logger(__name__)


class EncoderRunner:
    def __init__(
        self,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
    ):
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.vllm_config = vllm_config

        self.inputs_embeds = torch.zeros(
            max_num_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self.req_id_to_mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.tmp_is_mm_embed = UvaBufferPool(max_num_tokens, torch.bool)

        # Encoder CUDA graph manager (optional)
        self.encoder_cudagraph_manager: EncoderCudaGraphManager | None = None
        self.encoder_cudagraph_padded_mode: bool = True
        self._encoder_call_count: int = 0
        self._init_encoder_cudagraph_manager()

    def _init_encoder_cudagraph_manager(self) -> None:
        """Initialize encoder CUDA graph manager if enabled in config."""
        if self.vllm_config is None:
            return

        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return

        if not getattr(compilation_config, "cudagraph_mm_encoder", False):
            return

        # Import here to avoid circular imports
        from vllm.v1.worker.gpu.mm.encoder_cudagraph import EncoderCudaGraphManager

        bucket_sizes = getattr(
            compilation_config, "encoder_cudagraph_bucket_sizes", None
        )

        # Check if padded mode is enabled
        self.encoder_cudagraph_padded_mode = getattr(
            compilation_config,
            "encoder_cudagraph_padded_mode",
            True,  # Default to padded mode for better CUDA graph utilization
        )

        self.encoder_cudagraph_manager = EncoderCudaGraphManager(
            vllm_config=self.vllm_config,
            device=self.device,
            dtype=self.dtype,
            bucket_sizes=bucket_sizes,
        )

        # Log configuration
        grid_configs = self.encoder_cudagraph_manager.grid_configs
        logger.info(
            "Encoder CUDA graph manager initialized: "
            f"padded_mode={self.encoder_cudagraph_padded_mode}, "
            f"num_grids={len(grid_configs)}, "
            f"grids={grid_configs}"
        )

    def capture_encoder_cudagraphs(
        self,
        model: SupportsMultiModal,
    ) -> None:
        """
        Capture CUDA graphs for the encoder.

        Should be called during model warmup after the model is loaded.
        """
        if self.encoder_cudagraph_manager is None:
            return

        if not hasattr(model, "visual") or model.visual is None:
            logger.warning(
                "Model does not have a visual encoder, "
                "skipping encoder CUDA graph capture"
            )
            return

        self.encoder_cudagraph_manager.capture(
            vision_encoder=model.visual,
            embed_multimodal_fn=model.embed_multimodal,
        )

    def add_request(self, req_id: str, mm_features: list[MultiModalFeatureSpec]):
        self.req_id_to_mm_features[req_id] = mm_features

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_cache.pop(mm_hash, None)

    def remove_request(self, req_id: str) -> None:
        self.req_id_to_mm_features.pop(req_id, None)

    def prepare_mm_inputs(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> tuple[list[str], list[MultiModalKwargsItem]]:
        mm_hashes: list[str] = []
        mm_kwargs: list[MultiModalKwargsItem] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.req_id_to_mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append(mm_feature.data)
        return mm_hashes, mm_kwargs

    def _get_grid_thw_from_kwargs(
        self,
        mm_kwargs_group: dict,
        modality: str,
    ) -> list[list[int]] | None:
        """
        Extract grid_thw from mm_kwargs_group.

        Returns None if grid_thw is not available.
        """
        if modality not in ("image", "video"):
            return None

        # Try to get grid_thw from the kwargs
        grid_thw = mm_kwargs_group.get("image_grid_thw")
        if grid_thw is None:
            grid_thw = mm_kwargs_group.get("video_grid_thw")
        if grid_thw is None:
            return None

        # Convert to list if tensor
        if hasattr(grid_thw, "tolist"):
            grid_thw = grid_thw.tolist()

        return grid_thw

    def _estimate_visual_tokens(
        self,
        mm_kwargs_group: dict,
        modality: str,
    ) -> int | None:
        """
        Estimate the number of visual tokens for CUDA graph bucket selection.

        Returns None if estimation is not possible.
        """
        grid_thw = self._get_grid_thw_from_kwargs(mm_kwargs_group, modality)
        if grid_thw is None:
            return None

        # Calculate total visual tokens (after spatial merge, assuming 2x2)
        # Formula: sum of (T * H/merge * W/merge) for each item
        # Note: grid_thw contains [T, H, W] where H and W are already in patch units
        spatial_merge_size = 2  # Default for Qwen-VL models
        total_tokens = 0
        for t, h, w in grid_thw:
            tokens_per_image = t * (h // spatial_merge_size) * (w // spatial_merge_size)
            total_tokens += tokens_per_image

        return total_tokens

    @torch.inference_mode()
    def execute_mm_encoder(
        self,
        model: SupportsMultiModal,
        mm_hashes: list[str],
        mm_kwargs: list[MultiModalKwargsItem],
    ) -> list[torch.Tensor]:
        if not mm_hashes:
            return []

        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=False,
        ):
            # Try to use CUDA graph if available
            cudagraph_result = None
            if self.encoder_cudagraph_manager is not None:
                cudagraph_result = self._execute_with_cudagraph(
                    model, mm_kwargs_group, modality, num_items
                )

            if cudagraph_result is not None:
                # CUDA graph was used successfully
                curr_group_outputs = cudagraph_result
            else:
                # Fall back to eager mode
                curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output

        # Log encoder CUDA graph stats
        self._encoder_call_count += 1
        if self.encoder_cudagraph_manager is not None:
            self.encoder_cudagraph_manager.get_stats()

        return encoder_outputs

    def _execute_with_cudagraph(
        self,
        model: SupportsMultiModal,
        mm_kwargs_group: dict,
        modality: str,
        num_items: int,
    ) -> list[torch.Tensor] | None:
        """
        Execute the encoder using CUDA graphs if a matching graph is available.

        Supports two modes:
        1. Exact match: Only use CUDA graph if grid_thw exactly matches
        2. Padded mode: Pad inputs to fit the smallest available bucket

        Args:
            model: The multimodal model
            mm_kwargs_group: Batched multimodal kwargs
            modality: The modality type ("image" or "video")
            num_items: Number of items in the batch

        Returns:
            List of encoder outputs if CUDA graph was used, None otherwise
        """
        if self.encoder_cudagraph_manager is None:
            return None

        # Extract grid_thw from kwargs
        grid_thw = self._get_grid_thw_from_kwargs(mm_kwargs_group, modality)
        if grid_thw is None:
            return None

        # Currently only supports single-image batches for CUDA graph
        if len(grid_thw) != 1:
            logger.debug(
                "CUDA graph only supports single-image batches, "
                f"got {len(grid_thw)} images. Using eager mode."
            )
            return None

        # Extract pixel_values
        if modality == "image":
            pixel_values = mm_kwargs_group.get("pixel_values")
        else:  # video
            pixel_values = mm_kwargs_group.get("pixel_values_videos")

        if pixel_values is None:
            logger.debug("No pixel_values found in kwargs. Using eager mode.")
            return None

        # Ensure pixel_values is on the correct device
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        # Get spatial merge size for token calculations
        spatial_merge_size = getattr(model.visual, "spatial_merge_size", 2)
        t, h, w = grid_thw[0]
        num_output_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)

        # Try exact match first
        grid_key = self.encoder_cudagraph_manager.get_graph_for_grid(grid_thw)
        if grid_key is not None:
            # Exact match found - try to run
            output = self.encoder_cudagraph_manager.run(pixel_values, grid_thw)
            if output is not None:
                logger.info(
                    f"ViT CUDA graph EXACT: grid=({t}, {h}, {w}), "
                    f"tokens={num_output_tokens}"
                )
                return [output[:num_output_tokens]]

        # Try padded execution if enabled
        if self.encoder_cudagraph_padded_mode:
            result = self.encoder_cudagraph_manager.run_padded(
                pixel_values,
                grid_thw,
                num_output_tokens,
                spatial_merge_size,
            )
            if result is not None:
                output, padding_waste = result
                logger.info(
                    f"ViT CUDA graph PADDED: grid=({t}, {h}, {w}), "
                    f"tokens={num_output_tokens}, waste={padding_waste}"
                )
                return [output]

        # No CUDA graph available
        logger.info(f"ViT EAGER: grid=({t}, {h}, {w}), tokens={num_output_tokens}")
        return None

    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        computed_prefill_lens: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        is_prefilling = (computed_prefill_lens < prefill_lens).tolist()
        all_decode = not any(is_prefilling)
        if all_decode:
            # All decode requests, so no need to gather any embeddings.
            return [], torch.zeros(
                total_num_scheduled_tokens,
                dtype=torch.bool,
                device=self.device,
            )

        query_start = computed_prefill_lens.tolist()
        query_end = (computed_prefill_lens + num_scheduled_tokens).tolist()

        mm_embeds: list[torch.Tensor] = []
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device="cpu",
            pin_memory=False,
        )
        for i, req_id in enumerate(req_ids):
            if not is_prefilling[i]:
                # OPTIMIZATION: Skip decode requests.
                continue

            mm_features = self.req_id_to_mm_features[req_id]
            for mm_feature in mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= query_end[i]:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= query_start[i]:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(query_start[i] - start_pos, 0)
                end_idx = min(query_end[i] - start_pos, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = query_start_loc[i] + start_pos - query_start[i]
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )
                mm_embeds.append(mm_embeds_item)

        # Copy the is_mm_embed tensor to the GPU.
        is_mm_embed = self.tmp_is_mm_embed.copy_to_gpu(is_mm_embed)
        return mm_embeds, is_mm_embed

    @torch.inference_mode()
    def get_inputs_embeds(
        self,
        model: SupportsMultiModal,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = model.embed_input_ids(
            input_ids,
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        # Copy to the pre-allocated buffer for CUDA graphs.
        self.inputs_embeds[: x.shape[0]] = x
        return self.inputs_embeds

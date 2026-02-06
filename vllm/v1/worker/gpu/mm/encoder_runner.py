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
        self.encoder_cudagraph_budget_mode: bool = False
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

        self.encoder_cudagraph_manager = EncoderCudaGraphManager(
            vllm_config=self.vllm_config,
            device=self.device,
            dtype=self.dtype,
        )

        # Check if budget batching is configured
        self.encoder_cudagraph_budget_mode = bool(
            self.encoder_cudagraph_manager.token_budgets
            and self.encoder_cudagraph_manager.max_images_per_batch > 0
        )

        logger.info(
            "Encoder CUDA graph manager initialized: budget_mode=%s",
            self.encoder_cudagraph_budget_mode,
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
                curr_group_outputs = list(model.embed_multimodal(**mm_kwargs_group))

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
        Execute the encoder using budget batch CUDA graphs.

        Packs images (sorted smallest-first) into budget-sized batches
        and replays the smallest fitting CUDA graph. Falls back to eager
        if no budget graph fits.

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

        if not self.encoder_cudagraph_budget_mode:
            return None

        # Extract grid_thw from kwargs
        grid_thw = self._get_grid_thw_from_kwargs(mm_kwargs_group, modality)
        if grid_thw is None:
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
        visual = getattr(model, "visual", None)
        spatial_merge_size = getattr(visual, "spatial_merge_size", 2)

        return self._execute_budget_batch(
            pixel_values, grid_thw, spatial_merge_size
        )

    def _execute_budget_batch(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
        spatial_merge_size: int,
    ) -> list[torch.Tensor] | None:
        """
        Execute images using budget batch CUDA graphs.

        Sorts images by output token count (smallest first), greedily packs
        them into budget-sized batches, and replays the appropriate CUDA graph.

        Args:
            pixel_values: Concatenated pixel values for all images
            grid_thw: List of [T, H, W] for each image
            spatial_merge_size: Spatial merge size (e.g., 2)

        Returns:
            List of per-image output tensors in original order, or None
        """
        manager = self.encoder_cudagraph_manager
        if manager is None or not manager.budget_graph_keys:
            return None

        max_budget = max(manager.budget_graph_keys.keys())
        max_images = manager.max_images_per_batch

        # Compute per-image info: (output_tokens, input_patches, original_idx)
        image_info: list[tuple[int, int, int]] = []
        for i, (t, h, w) in enumerate(grid_thw):
            out_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
            in_patches = t * h * w
            image_info.append((out_tokens, in_patches, i))

        # Sort by output tokens ascending (small first)
        sorted_images = sorted(image_info, key=lambda x: x[0])

        # Compute pixel_values offsets for each original image
        patch_offsets = [0]
        for t, h, w in grid_thw:
            patch_offsets.append(patch_offsets[-1] + t * h * w)

        # Greedy packing into budget batches
        batches: list[list[tuple[int, int, int]]] = []
        current_batch: list[tuple[int, int, int]] = []
        current_tokens = 0

        for out_tokens, in_patches, orig_idx in sorted_images:
            if (
                current_tokens + out_tokens <= max_budget
                and len(current_batch) < max_images
            ):
                current_batch.append((out_tokens, in_patches, orig_idx))
                current_tokens += out_tokens
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(out_tokens, in_patches, orig_idx)]
                current_tokens = out_tokens

        if current_batch:
            batches.append(current_batch)

        # Execute each packed batch
        outputs: list[torch.Tensor | None] = [None] * len(grid_thw)

        for batch in batches:
            total_out_tokens = sum(out_tok for out_tok, _, _ in batch)

            # Find smallest budget graph that fits
            graph_key = manager.find_budget_graph(total_out_tokens)
            if graph_key is None:
                # No budget fits - fall back entirely
                logger.debug(
                    "No budget graph for %d tokens, falling back to eager",
                    total_out_tokens,
                )
                return None

            # Concatenate pixel values in sorted order
            pv_slices = []
            batch_grids = []
            for _, _, orig_idx in batch:
                start = patch_offsets[orig_idx]
                end = patch_offsets[orig_idx + 1]
                pv_slices.append(pixel_values[start:end])
                batch_grids.append(grid_thw[orig_idx])

            packed_pv = torch.cat(pv_slices, dim=0)

            # Run the budget graph
            output = manager.run_batched_contiguous(
                packed_pv, batch_grids, graph_key, spatial_merge_size
            )
            if output is None:
                logger.debug(
                    "Budget graph replay failed for key %s, falling back to eager",
                    graph_key,
                )
                return None

            # Split output by per-image output token counts
            offset = 0
            for out_tokens, _, orig_idx in batch:
                outputs[orig_idx] = output[offset : offset + out_tokens].clone()
                offset += out_tokens

            if manager.verbose:
                logger.info(
                    "ViT BUDGET BATCH: %d images, %d tokens, graph_key=%s",
                    len(batch),
                    total_out_tokens,
                    graph_key,
                )

        # Check all images were processed
        if any(o is None for o in outputs):
            return None

        return outputs  # type: ignore[return-value]

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

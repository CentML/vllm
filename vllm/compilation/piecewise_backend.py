# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from typing import Any

import torch.fx as fx

from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: VllmBackend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1
        self.is_encoder_compilation = vllm_backend.is_encoder

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        if self.is_encoder_compilation:
            # For encoder compilation we use the max int32 value
            # to set the upper bound of the compile ranges
            max_int32 = 2**31 - 1
            last_compile_range = self.compile_ranges[-1]
            assert (
                last_compile_range.end
                == vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.compile_ranges[-1] = Range(
                start=last_compile_range.start, end=max_int32
            )

        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        # Use encoder-specific capture sizes for encoder compilation
        if self.is_encoder_compilation:
            encoder_capture_sizes = self.compilation_config.encoder_cudagraph_capture_sizes
            if encoder_capture_sizes is not None:
                # Convert from output tokens to input patches
                # encoder_cudagraph_capture_sizes is specified in output tokens
                # but runtime_shape (from sym_shape_indices) is in input patches
                merge_size_sq = self.compilation_config.encoder_spatial_merge_size ** 2
                self.compile_sizes = [size * merge_size_sq for size in encoder_capture_sizes]
                logger.debug_once(
                    "PiecewiseBackend: converted encoder capture sizes from "
                    "output tokens %s to input patches %s (merge_size²=%d)",
                    tuple(encoder_capture_sizes), tuple(self.compile_sizes), merge_size_sq
                )
            else:
                self.compile_sizes = None
        else:
            self.compile_sizes = self.compilation_config.compile_sizes
        log_string = (
            f"PiecewiseBackend: compile_sizes: {self.compile_sizes} "
            f"(is_encoder={self.is_encoder_compilation})"
        )
        logger.debug_once(log_string)

        self.sym_shape_indices = sym_shape_indices

        # the entries for ranges that we need to either
        self.range_entries: dict[Range, RangeEntry] = {}

        # to_be_compiled_ranges tracks the remaining ranges to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_ranges: set[Range] = set(self.compile_ranges)

        # We only keep compilation management inside this class directly.
        if self.compile_sizes is not None:
            for size in self.compile_sizes:
                if isinstance(size, str):
                    assert size == "cudagraph_capture_sizes"
                    raise NotImplementedError(
                        "cudagraph_capture_sizes not supported in compile_sizes."
                        "This should be handled in `post_init_cudagraph_sizes`."
                    )
                else:
                    assert isinstance(size, int)
                    range = Range(start=size, end=size)
                    if range not in self.compile_ranges:
                        self.range_entries[range] = RangeEntry(
                            compile_range=range,
                        )
                        self.to_be_compiled_ranges.add(range)

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

    def check_for_ending_compilation(self) -> None:
        if self.is_last_graph and not self.to_be_compiled_ranges:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def _fakify_args(self, args: tuple[Any, ...]) -> list[Any]:
        # We need to pass fake example_inputs, otherwise torch.compile
        # will fakify the example_inputs potentially causing some non dynamic
        # dimension to be be duck shaped to other existing shapes that have hints
        # matching their values.
        # This is problem because it can lead to unintended specializations!
        # if the new wrongly dynamic dim is specialized
        # it will force specializing the whole shape
        # torch.compile probably should not accept
        # non fake tensors as example inputs!
        # See issue https://github.com/vllm-project/vllm/issues/27899
        fake_example_inputs = []
        for node in self.graph.graph.nodes:
            # All place holders come first
            if node.op == "placeholder":
                fake_example_inputs.append(node.meta["example_value"])
            else:
                break
        assert len(fake_example_inputs) == len(args)
        return fake_example_inputs

    def _maybe_compile_for_range_entry(
        self, range_entry: RangeEntry, args: tuple[Any, ...]
    ) -> Any:
        if not range_entry.compiled:
            range_entry.compiled = True
            self.to_be_compiled_ranges.remove(range_entry.compile_range)

            is_exact_size = range_entry.compile_range.is_single_size()

            # args are real arguments
            # fakify for range, real args for concrete size.
            # For concrete size, we clear the shape env in
            # compiler_manager.compile() so no need to fakify.
            args_list = (
                self._fakify_args(args)
                if not is_exact_size
                else list(args)
            )
            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args_list,
                self.vllm_backend.inductor_config,
                self.compilation_config,
                compile_range=range_entry.compile_range,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
            )

            self.check_for_ending_compilation()

    def _find_range_for_shape(self, runtime_shape: int) -> RangeEntry | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if self.compile_sizes is None:
            logger.debug(
                "PIECEWISE: compile_sizes is None, shape=%d, is_encoder=%s",
                runtime_shape, self.is_encoder_compilation
            )
            return None

        if runtime_shape in self.compile_sizes:
            # Exact match with capture size - will use cudagraph
            logger.debug(
                "PIECEWISE: exact match shape=%d in compile_sizes, is_encoder=%s",
                runtime_shape, self.is_encoder_compilation
            )
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            # No exact match - fall back to compile_ranges (no cudagraph)
            for range in self.compile_ranges:
                if runtime_shape in range:
                    logger.debug(
                        "PIECEWISE: shape=%d not in compile_sizes, "
                        "using compile_range=%s (NO CUDAGRAPH), is_encoder=%s",
                        runtime_shape, range, self.is_encoder_compilation
                    )
                    return self.range_entries[range]
            # Shape not in any range - will cause assertion error
            logger.warning(
                "PIECEWISE: shape=%d not in compile_sizes=%s or "
                "compile_ranges=%s, is_encoder=%s",
                runtime_shape, self.compile_sizes, self.compile_ranges,
                self.is_encoder_compilation
            )
        return None

    def __call__(self, *args: Any) -> Any:
        runtime_shape = args[self.sym_shape_indices[0]]

        range_entry = self._find_range_for_shape(runtime_shape)

        assert range_entry is not None, (
            f"Shape: {runtime_shape} out of considered ranges: {self.compile_ranges}"
        )

        self._maybe_compile_for_range_entry(range_entry, args)  # type: ignore[arg-type]

        return range_entry.runnable(*args)  # type: ignore[union-attr]

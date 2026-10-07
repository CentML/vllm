# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``kernel[grid](*args, **kwargs)`` with less host work per launch.

``JITFunction.run`` binds and specializes the arguments, looks the compiled
kernel up by that key and launches it. Around that it reads two env-backed
knobs (debug, instrumentation mode), runs the pre-run hooks, builds the launch
metadata for the launch hooks and passes the (empty by default) enter/exit
hook chains to the C launcher, which calls them. On an eager path issuing a
few hundred small launches per step (Qwen GDN layers in mixed prefill/decode
batches) that bookkeeping costs a few microseconds per launch.

``FastLaunch(fn)[grid](*args, **kwargs)`` uses the same binder, cache key and
kernel cache as ``fn[grid](...)`` and launches the same compiled kernel on the
same (current) stream with the same arguments. It only drops the bookkeeping
while that is a no-op: no pre-run hooks, no stage-inspection hook, empty
launch hook chains. Otherwise, on a cache miss (the JIT path compiles and
caches the kernel) and when a used global value changed (the JIT path
raises), it calls ``fn[grid](...)``. The debug and instrumentation knobs are
read once, at the first launch.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)

knobs = triton.knobs
HookChain = triton.knobs.HookChain
driver = triton.runtime.driver
JITFunction = triton.runtime.jit.JITFunction
compute_cache_key = triton.runtime.jit.compute_cache_key

_MISSING = object()


def _empty_hook(h: Any) -> bool:
    return h is None or (isinstance(h, HookChain) and not h.calls)


class FastLaunch:
    __slots__ = ("fn", "_options", "_off")

    def __init__(self, fn: Any) -> None:
        assert isinstance(fn, JITFunction), type(fn)
        self.fn = fn
        self._options: dict[str, Any] | None = None
        # Set if the JIT path cached the kernel under another key than the
        # one computed here (a Triton version whose run() keys differently):
        # every launch then takes the JIT path.
        self._off = False

    def __getitem__(self, grid: tuple[int, ...]):
        return lambda *args, **kwargs: self.run(grid, *args, **kwargs)

    def _slow(self) -> bool:
        fn = self.fn
        if (
            fn.pre_run_hooks
            or knobs.runtime.add_stages_inspection_hook is not None
            or not _empty_hook(knobs.runtime.launch_enter_hook)
            or not _empty_hook(knobs.runtime.launch_exit_hook)
        ):
            return True
        for (name, _), (val, globals_dict) in fn.used_global_vals.items():
            if globals_dict.get(name, _MISSING) != val:
                return True
        return False

    def run(self, grid: tuple[int, ...], *args, **kwargs) -> None:
        fn = self.fn
        if self._options is None:
            self._options = {
                "debug": fn.debug or knobs.runtime.debug,
                "instrumentation_mode": knobs.compilation.instrumentation_mode,
            }
        if self._off or "debug" in kwargs or self._slow():
            fn[grid](*args, **kwargs)
            return
        device = driver.active.get_current_device()
        kernel_cache, kernel_key_cache, _, _, binder = fn.device_caches[device]
        # JITFunction.run's keyword order: the options are keyed in order.
        bound_args, specialization, options = binder(*args, **kwargs, **self._options)
        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key)
        if kernel is None:
            fn[grid](*args, **kwargs)
            if key not in kernel_cache:
                self._off = True
                logger.warning_once(
                    "FastLaunch(%s): Triton cached the kernel under another "
                    "key; using the JIT launch path.",
                    fn.__name__,
                )
            return
        n = len(grid)
        kernel.run(
            grid[0],
            grid[1] if n > 1 else 1,
            grid[2] if n > 2 else 1,
            driver.active.get_current_stream(device),
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            None,
            *bound_args.values(),
        )

    def bind(self, *args, **kwargs) -> tuple[Any, list[str], list[Any]] | None:
        """(compiled kernel, parameter names, bound values) that ``run`` would
        launch for these arguments, or None when ``run`` would take the JIT
        path (hooks, debug, changed globals, kernel not compiled yet). The
        caller may relaunch the kernel with ``launch_bound`` and values whose
        specialization (dtypes, pointer alignment, specialized ints,
        constexprs) equals these.
        """
        fn = self.fn
        if self._options is None:
            self._options = {
                "debug": fn.debug or knobs.runtime.debug,
                "instrumentation_mode": knobs.compilation.instrumentation_mode,
            }
        if self._off or "debug" in kwargs or self._slow():
            return None
        device = driver.active.get_current_device()
        kernel_cache, kernel_key_cache, _, _, binder = fn.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs, **self._options)
        kernel = kernel_cache.get(
            compute_cache_key(kernel_key_cache, specialization, options)
        )
        if kernel is None:
            return None
        return kernel, list(bound_args.keys()), list(bound_args.values())


def launch_bound(kernel: Any, grid: tuple[int, ...], values: list[Any]) -> None:
    """Launch a kernel returned by ``FastLaunch.bind`` on the current stream,
    exactly as ``FastLaunch.run`` does.
    """
    device = driver.active.get_current_device()
    n = len(grid)
    kernel.run(
        grid[0],
        grid[1] if n > 1 else 1,
        grid[2] if n > 2 else 1,
        driver.active.get_current_stream(device),
        kernel.function,
        kernel.packed_metadata,
        None,
        None,
        None,
        *values,
    )

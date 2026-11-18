# SPDX-License-Identifier: Apache-2.0
"""In-process compatibility for Quack 0.6.5 on CuTe DSL 4.8."""

from functools import cache
from inspect import Parameter, signature


@cache
def ensure_quack_compatibility() -> None:
    """Preserve Quack's tuple conversion while forwarding CuTe DSL keywords.

    No installed files are changed. Import Quack before installing the adapter
    so a later model import cannot replace it with the incompatible converter.
    Newer converters that already accept the keyword are left alone.
    """
    import quack.cute_dsl_utils as quack
    from cutlass.cute import _tvm_ffi_args_spec_converter as converter

    current = converter._convert_single_arg
    params = signature(current).parameters
    if "is_constexpr" in params or any(
        p.kind == Parameter.VAR_KEYWORD for p in params.values()
    ):
        return
    if current is not quack._patched_convert_single_arg:
        raise RuntimeError("Unrecognized CuTe DSL argument converter")
    original = quack._original_convert_single_arg

    def compatible(arg, arg_name, arg_type, ctx, **kwargs):
        if kwargs.get("is_constexpr", False):
            return original(arg, arg_name, arg_type, ctx, **kwargs)
        if (
            isinstance(arg, tuple)
            and hasattr(type(arg), "_fields")
            and (arg_type is None or not hasattr(arg_type, "_fields"))
        ):
            return original(arg, arg_name, type(arg), ctx, **kwargs)
        # Keep Quack's Constexpr -> ConstNone handling for its own tuple ABI.
        from typing import get_origin

        import cutlass

        if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
            return quack.spec.ConstNone(arg_name)
        return original(arg, arg_name, arg_type, ctx, **kwargs)

    converter._convert_single_arg = compatible

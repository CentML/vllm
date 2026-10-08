# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts: execute production AST without importing GPU dependencies."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
ROUTING = ROOT / "vllm/model_executor/layers/fused_moe/flashinfer_exact_routing.py"
FINALIZE = ROOT / "vllm/model_executor/layers/fusion/moe_finalize.py"


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    for function in functions:
        function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class Symbol:
    dtype = SimpleNamespace(element_ty="bf16")

    def to(self, *args):
        return self

    def __add__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __lt__(self, other):
        return self


class PdlContractTests(unittest.TestCase):
    def test_routing_fail_closed(self):
        envs = SimpleNamespace(VLLM_MOE_ROUTING_EARLY_PDL=True,
                               VLLM_FI_SM107_MOE_PDL_MAX_TOKENS=208)
        platform = ModuleType("vllm.platforms")
        platform.current_platform = SimpleNamespace(is_device_capability=lambda cc: cc == 107)
        policy_name = "vllm.model_executor.layers.fusion.mxfp8_pdl"
        policy = ModuleType(policy_name)
        policy.mxfp8_producer_early_trigger = lambda: True
        ns = extract(ROUTING, ["_routing_early_pdl_enabled"],
                     dict(envs=envs, ENABLED=True, PREBUILT=None))
        with patch.dict(sys.modules, {"vllm.platforms": platform, policy_name: policy}):
            run = ns["_routing_early_pdl_enabled"]
            self.assertTrue(run())
            for key, value in (("ENABLED", False), ("PREBUILT", "unverified.so")):
                with self.subTest(key=key):
                    old = ns[key]
                    ns[key] = value
                    with self.assertRaises(ValueError):
                        run()
                    ns[key] = old
            for bound in (0, -1):
                envs.VLLM_FI_SM107_MOE_PDL_MAX_TOKENS = bound
                with self.assertRaises(ValueError):
                    run()
            envs.VLLM_FI_SM107_MOE_PDL_MAX_TOKENS = 208
            platform.current_platform.is_device_capability = lambda cc: False
            with self.assertRaises(ValueError):
                run()
            platform.current_platform.is_device_capability = lambda cc: True
            policy.mxfp8_producer_early_trigger = lambda: False
            with self.assertRaises(ValueError):
                run()
            envs.VLLM_MOE_ROUTING_EARLY_PDL = False
            self.assertFalse(run())

    def test_finalize_dependency_order(self):
        events = []
        value = Symbol()
        tl = SimpleNamespace(int64="int64", program_id=lambda axis: value,
                             arange=lambda a, b: value,
                             store=lambda *a, **kw: events.append("store"),
                             extra=SimpleNamespace(cuda=SimpleNamespace(
                                 gdc_wait=lambda: events.append("wait"),
                                 gdc_launch_dependents=lambda: events.append("release"))))
        def reduce(*args):
            events.append("read_reduce")
            return value
        ns = extract(FINALIZE, ["_moe_finalize_kernel"], dict(tl=tl, moe_finalize_row=reduce))
        for launch, early, expected in (
            (False, False, ["read_reduce", "store"]),
            (True, False, ["wait", "read_reduce", "store", "release"]),
            (True, True, ["wait", "release", "read_reduce", "store"]),
        ):
            with self.subTest(launch=launch, early=early):
                events.clear()
                ns["_moe_finalize_kernel"](value, value, value, value, 2048, 2048, 2048, 8, launch, early)
                self.assertEqual(events, expected)

    def test_finalize_launch_gate(self):
        calls = []
        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append((grid, kwargs))
        envs = SimpleNamespace(VLLM_MOE_FINALIZE_PDL=False)
        platform = SimpleNamespace(is_arch_support_pdl=lambda: True)
        tensor = SimpleNamespace(dtype="bf16", device="cuda")
        tensor.stride = lambda axis: 2048
        output = object()
        ns = extract(FINALIZE, ["moe_finalize"], dict(
            envs=envs, current_platform=platform,
            torch=SimpleNamespace(empty=lambda *a, **kw: output),
            triton=SimpleNamespace(next_power_of_2=lambda n: n),
            check_unfinalized=lambda *args: (2, 2048, 8),
            mxfp8_producer_early_trigger=lambda: True,
            _moe_finalize_kernel=Kernel()))
        for enabled, supported, expected in ((False, True, False), (True, False, False), (True, True, True)):
            envs.VLLM_MOE_FINALIZE_PDL = enabled
            platform.is_arch_support_pdl = lambda: supported
            self.assertIs(ns["moe_finalize"](tensor, tensor, tensor), output)
            self.assertEqual(calls[-1][1]["LAUNCH_PDL"], expected)
            self.assertEqual(calls[-1][1]["EARLY_TRIGGER"], expected)
            self.assertEqual(calls[-1][1]["launch_pdl"], expected)
        ns["check_unfinalized"] = lambda *args: (0, 2048, 8)
        count = len(calls)
        self.assertIs(ns["moe_finalize"](tensor, tensor, tensor), output)
        self.assertEqual(len(calls), count)


if __name__ == "__main__":
    unittest.main(verbosity=2)

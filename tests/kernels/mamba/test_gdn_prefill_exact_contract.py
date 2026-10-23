# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract smoke; executes extracted production functions, not CUDA."""
import ast
import functools
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / 'vllm/third_party/flashinfer_gdn_vsplit/adapter.py'
PRODUCER = ROOT / 'vllm/model_executor/layers/mamba/ops/gdn_conv_cuda.py'
GRAPH = ROOT / 'vllm/model_executor/layers/mamba/gdn/gdn_layer_graphs.py'
FLAGS = ('STAGED', 'STAGED_LOAD', 'VEC_STATE', 'EARLY_REL', 'PDL', 'HEAD_MAJOR')


def extract(path, names, ns):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def adapter():
    ns = dict(torch=torch, functools=functools, Optional=Optional,
              _CG0_SPLIT=False, _C1_REORDER=False)
    ns.update({'_' + f: False for f in FLAGS})
    return extract(ADAPTER, ['_dense_bf16_state', '_variants', '_cg0_split',
                            '_c1_reorder', '_cache', '_io', '_st', '_mark_state',
                            'chunk_gated_delta_rule_vsplit'], ns)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = adapter()
        self.raw = torch.empty(3, 2 * 128 * 128 + 64, dtype=torch.bfloat16)
        self.pool = self.raw[:, :2 * 128 * 128].view(3, 2, 128, 128)
        self.g = torch.empty(65, 2)
        self.idx = torch.tensor([1, 2], dtype=torch.int32)

    def variants(self, **kw):
        args = dict(initial_state=self.pool, output_state=self.pool,
                    state_indices=self.idx, v_split=2, gate=self.g,
                    beta=self.g, pdl=True, h0_late=True)
        args.update(kw)
        return self.ns['_variants'](**args)

    def test_default_off_and_padded_pool(self):
        self.assertTrue(self.ns['_dense_bf16_state'](self.pool))
        self.assertEqual(self.variants(), (False, False, False, False, False, False, ()))

    def test_state_geometry_rejections(self):
        for t in (self.pool.float(), self.pool.transpose(2, 3), self.pool[..., :64],
                  torch.empty(3, 2, 128, 129, dtype=torch.bfloat16)[..., 1:]):
            with self.subTest(shape=t.shape, stride=t.stride()):
                self.assertFalse(self.ns['_dense_bf16_state'](t))
        self.assertFalse(self.ns['_dense_bf16_state'](None))

    def test_staged_input_output_guards_independent(self):
        self.ns.update(_STAGED=True, _STAGED_LOAD=True)
        self.assertEqual(self.variants()[:3], (True, True, False))
        bad = torch.empty(3, 2, 128, 129, dtype=torch.bfloat16)[..., 1:]
        self.assertEqual(self.variants(initial_state=bad)[:3], (True, False, False))
        self.assertEqual(self.variants(output_state=bad)[:3], (False, False, False))
        self.assertFalse(self.variants(state_indices=None)[0])
        self.assertFalse(self.variants(v_split=1)[0])
        self.ns['_STAGED'] = False
        self.assertFalse(self.variants()[1])

    def test_vec_and_pdl_are_independent_optins(self):
        self.ns['_VEC_STATE'] = True
        self.assertTrue(self.variants(v_split=1)[2])
        self.assertFalse(self.variants(v_split=2)[2])
        self.assertFalse(self.variants(v_split=1, initial_state=self.pool.float())[2])
        self.ns['_PDL'] = True
        self.assertEqual(self.variants()[3:5], (True, True))
        self.assertEqual(self.variants(pdl=False)[3:5], (False, False))
        self.assertEqual(self.variants(h0_late=False)[3:5], (True, False))

    def test_head_major_stride_contract(self):
        self.ns['_HEAD_MAJOR'] = True
        hm = torch.empty(2, 128)[:, :65].t()
        self.assertTrue(self.variants(gate=hm, beta=hm)[5])
        with self.assertRaises(AssertionError):
            self.variants(gate=hm, beta=self.g)
        invalid = torch.empty(2, 66)[:, :65].t()
        with self.assertRaises(AssertionError):
            self.variants(gate=invalid, beta=invalid)

    def test_compile_cache_distinguishes_guarded_variants(self):
        # Exercise the actual normal and HOST_TRIM dispatchers. The fake compiler
        # observes specialization identities only; it does not assert numerics.
        class Desc:
            def mark_compact_shape_dynamic(self, **kw): return self
            def mark_layout_dynamic(self, **kw): return self
        built = []
        called = []
        class Kernel:
            def __init__(self, **kw): self.kw = kw
            @staticmethod
            def get_workspace_size(*args): return 16
        def compile_kernel(k, *args, **kw):
            built.append(k.kw)
            def run(*args): called.append(k.kw)
            return run
        self.ns.update(GDN_HOST_TRIM=True, _fast_compiled={}, _fast_ws={},
                       _num_sm=lambda dev: 1, GatedDeltaNetChunkedKernel=Kernel,
                       logger=SimpleNamespace(info=lambda *a: None),
                       cutlass=SimpleNamespace(BFloat16='bf16', Float16='fp16', Float32='fp32'),
                       cute=SimpleNamespace(compile=compile_kernel),
                       cuda=SimpleNamespace(CUstream=lambda h: h),
                       from_dlpack=lambda *a, **kw: Desc())
        q = torch.empty(65, 2, 128, dtype=torch.bfloat16)
        cu = torch.tensor([0, 65], dtype=torch.int32)
        def run(g=self.g, pool=None):
            p = self.pool if pool is None else pool
            self.ns['chunk_gated_delta_rule_vsplit'](
                q, q, q, g, g, q, cu, p, p, 128**-0.5,
                state_indices=self.idx, workspace=torch.empty(16, dtype=torch.int8))
        with patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=7)), \
             patch.object(torch._C, '_cuda_getCurrentRawStream', return_value=7, create=True):
            run(); run()
            self.assertEqual(len(built), 1)
            self.ns.update(_STAGED=True, _STAGED_LOAD=True)
            run(); run()
            self.assertEqual(len(built), 2)
            self.assertTrue(called[-1]['staged_store'])
            self.ns['_HEAD_MAJOR'] = True
            run(torch.empty(2, 128)[:, :65].t())
            self.assertEqual(len(built), 3)
            self.ns['_STAGED'] = False
            self.ns['_STAGED_LOAD'] = False
            run()
            self.assertEqual(len(built), 3)
            self.assertFalse(called[-1]['staged_store'])

    def test_producer_default_head_major_and_rejection(self):
        calls = []
        flags = SimpleNamespace(VLLM_GDN_VSPLIT_HEAD_MAJOR=False)
        ns = extract(PRODUCER, ['gdn_conv_cuda_prep'], dict(
            torch=torch, envs=flags, GDN_HOST_TRIM=False,
            _ext=[SimpleNamespace(run=lambda *a, **kw: calls.append(kw) or True)]))
        H, HV, P = 1, 2, 65
        x = torch.empty(P, (2*H+HV)*128, dtype=torch.bfloat16)
        args = (x, torch.empty(x.shape[1], 4), torch.empty(3),
                self.idx, torch.ones(2, dtype=torch.bool),
                torch.tensor([0, P], dtype=torch.int32),
                torch.empty(P, HV), torch.empty(P, HV),
                torch.empty(HV), torch.empty(HV), H, 128, 128)
        run = ns['gdn_conv_cuda_prep']
        base = run(*args)
        self.assertEqual(base[3].stride(), (HV, 1))
        self.assertEqual(calls[-1], {'gb_ts': 0, 'pdl': 0})
        flags.VLLM_GDN_VSPLIT_HEAD_MAJOR = True
        hm = run(*args, pdl=True)
        self.assertEqual(hm[3].shape, (P, HV))
        self.assertEqual(hm[3].stride(), (1, 128))
        self.assertEqual(calls[-1], {'gb_ts': 128, 'pdl': 1})
        run(*args, out=base)
        self.assertEqual(calls[-1]['gb_ts'], 0)
        n = len(calls)
        self.assertIsNone(run(*args, out=(*hm[:4], base[4])))
        self.assertEqual(len(calls), n)


if __name__ == '__main__':
    unittest.main(verbosity=2)

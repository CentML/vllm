#!/usr/bin/env python3
"""Inventory + one-GPU smoke of the DeepSeek-V4.1-Flash Rubin image (see README.md "Validate").

Run inside the image on a VR200 (sm_107) GPU, without any PYTHONPATH overlay:
    python3 docker/dsv41-rubin/validate_image.py
Exit status: number of failed checks.
"""
import collections
import importlib
import importlib.metadata as md
import os
import subprocess

import torch

failed = []


def check(ok: bool, what: str) -> None:
    print(("OK    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        failed.append(what)


import vllm  # noqa: E402
import vllm._flashmla_C  # noqa: E402,F401

print(f"vllm {vllm.__version__} from {os.path.dirname(vllm.__file__)}")
print("versions", {p: md.version(p) for p in ("torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl",
                                              "transformers", "ai-dynamo", "ai-dynamo-runtime")})
print("build commit", os.environ.get("VLLM_BUILD_COMMIT"), "TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST"))
cc = torch.cuda.get_device_capability()
check(cc == (10, 7), f"GPU {torch.cuda.get_device_name()} compute capability {cc} (Rubin = (10, 7))")

# FlashMLA: sm_100f + sm_107a cubins (CUTLASS 4.8 build); the driver runs the exact-match sm_107a cubin on Rubin.
so = vllm._flashmla_C.__file__
elf = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True).stdout.split()
cubins = collections.Counter(e.rsplit(".", 2)[-2] for e in elf if e.endswith(".cubin"))
check(cubins.get("sm_107a", 0) > 0 and cubins.get("sm_100", 0) > 0, f"_flashmla_C cubins {dict(cubins)}")
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import is_flashmla_mega_attn_supported  # noqa: E402
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported  # noqa: E402

check(is_flashmla_mega_attn_supported()[0] and is_flashmla_sparse_supported()[0], "FlashMLA mega/sparse attention supported")

# DeepGEMM: vendored package with the SM107 MegaMoE kernel and per-die (locality-domain) execution.
from vllm.utils.deep_gemm import _import_deep_gemm  # noqa: E402

dg = _import_deep_gemm()
check("vllm/third_party/deep_gemm" in dg.__file__, f"deep_gemm is the vendored copy ({dg.__file__})")
syms = sorted(n for n in dir(dg._C) if n.startswith("sm107_"))
check(len(syms) >= 6, f"deep_gemm SM107 symbols {syms}")
inc = os.path.join(os.path.dirname(dg.__file__), "include")
check(os.path.exists(os.path.join(inc, "deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh")), "SM107 MegaMoE kernel header")
with open(os.path.join(inc, "sm107_cutlass/cutlass/version.h")) as f:
    check("CUTLASS_MINOR 8" in f.read(), "include/sm107_cutlass is CUTLASS 4.8")
loc = importlib.import_module(dg.__name__ + ".mega.sm107_locality")
check(loc.is_localization_available(), f"{loc.get_num_locality_domains()} locality domains (2 dies)")
sm_map = loc.get_balanced_sm_locality_domains()
check(sm_map.numel() == torch.cuda.get_device_properties(0).multi_processor_count and set(sm_map.unique().tolist()) == {0, 1},
      f"balanced SM -> die map {sm_map.bincount().tolist()}")
w = torch.randint(-128, 127, (4, 512, 256), dtype=torch.int8, device="cuda")
lw = loc.localize(w)
check(tuple(lw.shape) == (2, 4, 256, 256) and dg._C.sm107_is_localized(lw)
      and [dg._C.sm107_get_locality_domain_of(lw[d]) for d in range(2)] == [0, 1]
      and torch.equal(lw.movedim(0, 1).reshape(4, 512, 256), w), "die-local weight copy (content preserved)")

# Dense opt-in kernels and Dynamo.
for name in ("vllm.model_executor.kernels.linear.mxfp8.rubin", "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
             "vllm.model_executor.layers.quantization.utils.mxfp8_utils"):
    try:
        importlib.import_module(name)
        check(True, f"import {name}")
    except Exception as e:  # noqa: BLE001 - report every failing import, keep checking
        check(False, f"import {name}: {type(e).__name__}: {e}")
import dynamo.frontend  # noqa: E402,F401
import dynamo.vllm  # noqa: E402,F401

check(True, "import dynamo.frontend, dynamo.vllm")
print(f"== {'ALL OK' if not failed else f'{len(failed)} FAILED'}")
raise SystemExit(len(failed))

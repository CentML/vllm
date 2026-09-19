# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backport FlashInfer's SM107 wheel import and CuTe DSL 4.8 compatibility fixes."""

import hashlib
import importlib.metadata
from pathlib import Path
from urllib.request import urlopen

REVISION = "2b16e3c765fd4bf79339a4b2536a2de68a9c85ee"
# Stock 0.6.18.post1 and upstream source hashes, respectively.
SOURCES = {
    "dense_blockscaled_gemm_sm100.py": (
        "aad93031b1145c43195d1af2cbeb310c91df5c000310df26dfea768be949f705",
        "64b83236fa69202c97e015fedf298fe5abdbf3f464609fb44a5e7a92158dd79b",
    ),
    "dense_blockscaled_gemm_sm107.py": (
        "c9def937d2bf76b363bb321aa4053ffd39055febdcef3c890572bd2c4f2946ad",
        "5959d66e7b8becc0e4169aa081a0886c829712323531aa655b02b2013a14ed2d",
    ),
}


def main() -> None:
    dist = importlib.metadata.distribution("flashinfer-python")
    if dist.version != "0.6.18.post1":
        raise RuntimeError(
            f"SM107 backport requires FlashInfer 0.6.18.post1, got {dist.version}"
        )
    root = Path(dist.locate_file("flashinfer/gemm/kernels"))
    updates = {}
    for name, (stock_hash, source_hash) in SOURCES.items():
        path = root / name
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_hash == source_hash:
            continue
        if current_hash != stock_hash:
            raise RuntimeError(
                f"Refusing to replace unexpected FlashInfer source: {path}"
            )
        url = (
            "https://raw.githubusercontent.com/flashinfer-ai/flashinfer/"
            f"{REVISION}/flashinfer/gemm/kernels/{name}"
        )
        with urlopen(url, timeout=60) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != source_hash:
            raise RuntimeError(f"FlashInfer source checksum mismatch: {name}")
        updates[path] = source
    # Validate both files before changing either installed source.
    for path, source in updates.items():
        path.write_bytes(source)
    print(f"FlashInfer 0.6.18.post1 with SM107 sources from {REVISION}")


if __name__ == "__main__":
    main()

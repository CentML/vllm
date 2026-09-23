# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.mm.rope import RopeState


class MRoPE(torch.nn.Module):
    def __init__(self, positions_device: str):
        super().__init__()
        self.positions_device = positions_device

    def get_mrope_input_positions(self, token_ids, mm_features):
        base = torch.tensor(token_ids, dtype=torch.int64, device=self.positions_device)
        return torch.stack((base, base * 2 + 1, base * 3 + 2)), 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("uva_target", [False, True])
@pytest.mark.parametrize("positions_device", ["cpu", "cuda"])
def test_rope_stages_mrope_positions(uva_target, positions_device, monkeypatch):
    """M-RoPE positions remain observable from CPU and GPU inputs."""
    if uva_target and not is_uva_available():
        pytest.skip("UVA is not available.")
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: uva_target)

    device = torch.device("cuda:0")
    state = RopeState(3, 2, 8192, 8192, device)
    assert (state.prefill_positions.write_contents is not None) == uva_target
    model = MRoPE(positions_device)
    token_ids = [list(range(1027)), list(range(4099, 8192))]
    expected = [model.get_mrope_input_positions(ids, [])[0] for ids in token_ids]
    for req_idx, ids in enumerate(token_ids):
        state.init_prefill_positions(req_idx, model, ids, [])
    state.apply_staged_writes()
    torch.accelerator.synchronize()

    for req_idx, positions in enumerate(expected):
        torch.testing.assert_close(
            state.read_prefill_positions(req_idx, len(token_ids[req_idx])).cpu(),
            positions.cpu().to(torch.int32),
            rtol=0,
            atol=0,
        )

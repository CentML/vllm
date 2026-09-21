# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor
from vllm.v1.worker.gpu.mm.rope import RopeState

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    cuda_view.mul_(2)
    assert cuda_view[0, 0] == 2
    assert cuda_view[2, 3] == 4
    assert cuda_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cuda_view[0, 0] = 1
    cuda_view[2, 3] = 2
    cuda_view[4, 5] = -1
    cuda_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_staged_write_uses_uva_contents_for_uva_target(device, monkeypatch):
    def fail_async_tensor_h2d(*args, **kwargs):
        pytest.fail("UVA-backed targets should not copy write contents to the GPU")

    monkeypatch.setattr(buffer_utils, "async_tensor_h2d", fail_async_tensor_h2d)
    staged = StagedWriteTensor(
        (3, 4096),
        dtype=torch.int32,
        device=torch.device(device),
        max_concurrency=2,
        uva_instead_of_gpu=True,
    )

    staged.stage_write(2, 3, [11, 12, 13])
    staged.apply_write()
    torch.accelerator.synchronize()
    staged.stage_write(1, 7, [21, 22])
    staged.apply_write()
    torch.accelerator.synchronize()
    staged.stage_write(0, 1020, range(1500))
    staged.apply_write()
    torch.accelerator.synchronize()

    assert staged.gpu[2, 3:6].tolist() == [11, 12, 13]
    assert staged.gpu[1, 7:9].tolist() == [21, 22]
    assert staged.gpu[0, 1020:2520].tolist() == list(range(1500))


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("input_type", ["list", "numpy", "tensor"])
@pytest.mark.parametrize("size", [(4,), (4, 3)])
def test_uva_pool_overwrites_exposed_prefix(input_type, size):
    """Both slots expose only current contents across growth and shorter reuse."""
    pool = buffer_utils.UvaBufferPool(size, torch.int32, max_concurrency=2)
    for buf in pool._uva_bufs:
        assert tuple(buf.cpu.shape) == size
        assert buf.cpu.is_pinned()
        assert torch.count_nonzero(buf.cpu).item() == 0
    lengths = [0, 0, 3, 3, 4, 4, 5, 3, 3, 5, 1024, 1024, 1025, 1025, 2, 2]
    for step, length in enumerate(lengths):
        if input_type == "list" and len(size) > 1 and length == 0:
            # An empty list has no trailing shape; preserve NumPy's rejection.
            with pytest.raises(ValueError, match="could not broadcast"):
                pool.copy_to_uva([])
            continue
        shape = (length, *size[1:])
        expected = (
            torch.arange(int(np.prod(shape)), dtype=torch.int32, device="cpu").reshape(
                shape
            )
            - step
        )
        values = expected.tolist()
        if input_type == "numpy":
            values = expected.numpy()
        elif input_type == "tensor":
            values = expected
        before = list(pool._uva_bufs)
        slot = (pool._curr + 1) % pool.max_concurrency
        result = pool.copy_to_uva(values)
        assert tuple(result.shape) == shape
        assert pool._curr == slot
        assert pool._uva_bufs[1 - slot] is before[1 - slot]
        if length <= before[slot].cpu.shape[0]:
            assert pool._uva_bufs[slot] is before[slot]
        else:
            assert tuple(pool._uva_bufs[slot].cpu.shape) == (
                1 << (length - 1).bit_length(),
                *size[1:],
            )
        assert pool.size == size
        # Blocking read also retires GPU readers before the next slot reuse.
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("input_type", ["numpy", "tensor"])
def test_uva_pool_copy_to_gpu_preserves_shape_and_out(use_out, input_type):
    pool = buffer_utils.UvaBufferPool((2, 3), torch.int32, max_concurrency=2)
    for length in (2, 5, 3, 6):
        expected = torch.arange(length * 3, dtype=torch.int32, device="cpu").reshape(
            length, 3
        )
        values = expected.numpy() if input_type == "numpy" else expected
        out = (
            torch.empty(expected.shape, dtype=torch.int32, device="cuda")
            if use_out
            else None
        )
        result = pool.copy_to_gpu(values, out=out)
        if use_out:
            assert result is out
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("uva_target", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_staged_write_inflight(uva_target, dtype):
    """Preserve every generation until its consumer finishes before slot reuse."""
    device = torch.device("cuda:0")
    with torch.accelerator.device_index(device.index):
        state = StagedWriteTensor(
            (4, 4096),
            dtype,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=uva_target,
        )
        assert (state.write_contents is not None) == uva_target
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        pending: list[tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]] = []
        expected = torch.zeros((4, 4096), dtype=dtype, device="cpu")
        for step in range(24):
            if len(pending) == 2:
                event, snapshot, reference = pending.pop(0)
                event.synchronize()
                torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
            # Growing and shrinking lengths exercise reallocation and reuse.
            length = [3, 17, 1500, 4090][step % 4]
            row = step % 3
            values = torch.arange(length, dtype=dtype, device="cpu") + step * 8192
            if dtype == torch.float32:
                values += 0.25
            expected[row, 2 : 2 + length] = values
            expected[3, 1:4] = step
            with torch.cuda.stream(stream):
                state.stage_write(row, 2, values.tolist())
                state.stage_write(3, 1, [step] * 3)
                state.apply_write()
                # A GPU consumer observes this generation before the next update.
                snapshot = state.gpu.clone()
                event = torch.cuda.Event()
                event.record(stream)
            pending.append((event, snapshot, expected.clone()))
        for event, snapshot, reference in pending:
            event.synchronize()
            torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
def test_staged_write_tensor_chunks_inflight():
    """Keep large CPU position tensors valid across pooled-slot reuse."""
    device = torch.device("cuda:0")
    with torch.accelerator.device_index(device.index):
        state = StagedWriteTensor(
            (6, 131072),
            torch.int32,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=True,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        pending: list[tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]] = []
        expected = torch.zeros((6, 131072), dtype=torch.int32, device="cpu")
        for step in range(24):
            if len(pending) == 2:
                event, snapshot, reference = pending.pop(0)
                event.synchronize()
                torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
            length = [16384, 65536, 131072, 65536][step % 4]
            base = torch.arange(length, dtype=torch.long) + step * 1_000_000
            positions = (
                base.unsqueeze(0).expand(3, -1)
                if step % 2
                else torch.stack((base, base * 2 + 1, base * 3 + 2))
            )
            row = (step % 2) * 3
            expected[row : row + 3, :length] = positions.to(torch.int32)
            with torch.cuda.stream(stream):
                for axis in range(3):
                    state.stage_write_tensor(row + axis, 0, positions[axis])
                state.apply_write()
                snapshot = state.gpu.clone()
                event = torch.cuda.Event()
                event.record(stream)
            pending.append((event, snapshot, expected.clone()))
        for event, snapshot, reference in pending:
            event.synchronize()
            torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
def test_fused_staged_writer_materializes_tensor_chunks():
    """Fused writes preserve tensor-staged chunks across staging modes."""
    device = torch.device("cuda:0")
    states = [
        StagedWriteTensor(
            (1, 64),
            torch.int32,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=True,
        )
        for _ in range(2)
    ]
    states[0].stage_write(0, 0, [7])
    states[0].stage_write_tensor(0, 1, torch.tensor([8, 9], dtype=torch.int64))
    states[1].stage_write_tensor(0, 0, torch.tensor([10, 11], dtype=torch.int64))
    writer = FusedStagedWriter(device, max_writes=2)
    output_ptrs = torch.tensor(
        [state.gpu.data_ptr() for state in states], dtype=torch.uint64, device=device
    )
    output_strides = torch.tensor(
        [state.gpu.stride(0) for state in states], dtype=torch.int64, device=device
    )
    writer.apply(states, output_ptrs, output_strides)
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        states[0].gpu[0, :3].cpu(), torch.tensor([7, 8, 9], dtype=torch.int32)
    )
    torch.testing.assert_close(
        states[1].gpu[0, :2].cpu(), torch.tensor([10, 11], dtype=torch.int32)
    )


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
def test_rope_cpu_positions_use_uva_tensor_staging(monkeypatch):
    """CPU M-RoPE positions use typed staging when their UVA path is available."""

    class CpuMRoPE(torch.nn.Module):
        def get_mrope_input_positions(self, token_ids, mm_features):
            base = torch.tensor(token_ids, dtype=torch.long)
            return torch.stack((base, base * 2 + 1, base * 3 + 2)), 0

    state = RopeState(3, 2, 8192, 8192, torch.device("cuda:0"))
    assert state.prefill_positions.write_contents is not None

    def fail_list_staging(*args, **kwargs):
        pytest.fail("CPU M-RoPE positions must use tensor-backed UVA staging")

    monkeypatch.setattr(state.prefill_positions, "stage_write", fail_list_staging)
    first_ids = list(range(1027))
    second_ids = list(range(4099, 8192))
    model = CpuMRoPE()
    first, _ = model.get_mrope_input_positions(first_ids, [])
    second, _ = model.get_mrope_input_positions(second_ids, [])
    state.init_prefill_positions(0, model, first_ids, [])
    state.init_prefill_positions(1, model, second_ids, [])
    state.apply_staged_writes()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        state.read_prefill_positions(0, len(first_ids)).cpu(),
        first.to(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state.read_prefill_positions(1, len(second_ids)).cpu(),
        second.to(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="CUDA is required.")
def test_rope_cpu_positions_falls_back_without_uva(monkeypatch):
    """CPU M-RoPE positions retain the list path when UVA staging is disabled."""

    class CpuMRoPE(torch.nn.Module):
        def get_mrope_input_positions(self, token_ids, mm_features):
            positions = torch.stack(
                (
                    torch.arange(len(token_ids), dtype=torch.long),
                    torch.arange(len(token_ids), dtype=torch.long) * 2 + 1,
                    torch.arange(len(token_ids), dtype=torch.long) * 3 + 2,
                )
            )
            return positions, 0

    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    positions, _ = CpuMRoPE().get_mrope_input_positions(list(range(64)), [])
    state = RopeState(3, 1, 64, 64, torch.device("cuda:0"))
    assert state.prefill_positions.write_contents is None
    state.init_prefill_positions(0, CpuMRoPE(), list(range(64)), [])
    state.apply_staged_writes()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        state.read_prefill_positions(0, 64).cpu(),
        positions.to(torch.int32),
        rtol=0,
        atol=0,
    )

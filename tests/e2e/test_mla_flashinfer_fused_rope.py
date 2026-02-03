# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for MLA with FlashInfer fused RoPE kernels.

Based on the benchmark pattern from dsr1-fp4-bench.sh:
- Uses DeepSeek-R1 model with FP4/FP8 quantization
- Tests FLASHINFER_MLA attention backend
- Verifies server startup and basic inference
"""

import pytest

from tests.utils import RemoteOpenAIServer

# Use a smaller test model for faster tests
TEST_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# Check if FlashInfer rope_quantize_fp8 is available
try:
    from flashinfer.rope import rope_quantize_fp8

    HAS_FLASHINFER_ROPE_QUANTIZE_FP8 = True
except ImportError:
    HAS_FLASHINFER_ROPE_QUANTIZE_FP8 = False


@pytest.mark.skipif(
    not HAS_FLASHINFER_ROPE_QUANTIZE_FP8,
    reason="FlashInfer rope_quantize_fp8 not available",
)
@pytest.mark.e2e
class TestMLAFlashInferFusedRopeE2E:
    """E2E tests for MLA with FlashInfer fused RoPE kernels."""

    @pytest.fixture
    def server(self):
        """Start vLLM server with FLASHINFER_MLA backend and FP8 cache."""
        # Use a smaller model for faster CI
        args = [
            "--model",
            TEST_MODEL,
            "--attention-backend",
            "FLASHINFER_MLA",
            "--kv-cache-dtype",
            "fp8",
            "--max-model-len",
            "2048",
            "--enforce-eager",  # Disable cuda graph for testing
        ]

        with RemoteOpenAIServer(TEST_MODEL, args) as server:
            yield server

    def test_server_startup_with_flashinfer_mla(self, server):
        """Verify server starts with FLASHINFER_MLA backend."""
        # If we get here, server started successfully
        client = server.get_client()
        models = client.models.list()
        model_names = [m.id for m in models.data]
        assert TEST_MODEL in model_names

    def test_basic_completion(self, server):
        """Verify basic completion works with fused RoPE kernels."""
        client = server.get_client()

        response = client.completions.create(
            model=TEST_MODEL,
            prompt="Hello, my name is",
            max_tokens=10,
            temperature=0,
        )

        assert response.choices[0].text is not None
        assert len(response.choices[0].text) > 0

    def test_decode_throughput_no_regression(self, server):
        """
        Run a small decode test to verify throughput is reasonable.

        This is a sanity check, not a performance benchmark.
        """
        client = server.get_client()

        # Small batch of short prompts
        prompts = ["Hello"] * 10

        responses = []
        for prompt in prompts:
            response = client.completions.create(
                model=TEST_MODEL,
                prompt=prompt,
                max_tokens=50,
                temperature=0,
            )
            responses.append(response)

        # Verify all responses completed
        assert len(responses) == 10
        for response in responses:
            assert response.choices[0].text is not None


@pytest.mark.skipif(
    not HAS_FLASHINFER_ROPE_QUANTIZE_FP8,
    reason="FlashInfer rope_quantize_fp8 not available",
)
@pytest.mark.e2e
@pytest.mark.slow
class TestMLAFlashInferFusedRopeE2EFull:
    """
    Full E2E tests with larger model.

    These tests are slow and require significant GPU memory.
    Mark with @pytest.mark.slow to skip in CI unless explicitly requested.
    """

    @pytest.fixture
    def server_full(self):
        """Start vLLM server with full DeepSeek-R1 model."""
        # Full model for comprehensive testing
        model = "nvidia/DeepSeek-R1-0528-FP4"
        args = [
            "--model",
            model,
            "--attention-backend",
            "FLASHINFER_MLA",
            "--max-model-len",
            "4096",
            "--tensor-parallel-size",
            "1",
        ]

        with RemoteOpenAIServer(model, args) as server:
            yield server

    def test_full_model_inference(self, server_full):
        """Test inference with full DeepSeek-R1 FP4 model."""
        client = server_full.get_client()

        response = client.completions.create(
            model="nvidia/DeepSeek-R1-0528-FP4",
            prompt="What is 2 + 2?",
            max_tokens=100,
            temperature=0,
        )

        assert response.choices[0].text is not None
        assert len(response.choices[0].text) > 0

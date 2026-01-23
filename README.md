<!-- markdownlint-disable MD001 MD041 -->

# FA4 Integration

### (1) Support fa4 in vllm.

From low-level to high-level:
1. Add `FLASH_ATTN_CUTE` (FA4 / `flash_attn.cute`) to `vllm/v1/attention/backends/registry.py` (`AttentionBackendEnum`).
2. Create a new file `vllm/v1/attention/backends/fa4_utils.py`, for the utils / imports for fa4 (keep imports lazy).
3. Register the new backend in `vllm/platforms/cuda.py` (FA4 is **Blackwell-only (CC 10.x)** and **opt-in** via `--mm-encoder-attn-backend FLASH_ATTN_CUTE`; default remains FA2/3 or Torch SDPA).
4. Add the fa4 custom op under `vllm/v1/attention/ops/vit_attn_wrappers.py`.
5. Update `vllm/model_executor/layers/attention/mm_encoder_attention.py` to add another _forward_impl method for fa4 (`FLASH_ATTN_CUTE`).
6. Update `vllm/model_executor/models/qwen3_vl.py` and (optionally) `qwen2_5_vl.py` to accept `FLASH_ATTN_CUTE` and compute `max_seqlen` for it.

Notes:
- FA4 (`flash_attn.cute`) is only considered on **Blackwell** (compute capability 10.x) in this vLLM fork.
- To force FA4 for ViT/MM encoder attention (Blackwell only): `--mm-encoder-attn-backend FLASH_ATTN_CUTE`.

### (2) Do the kernel_warmup in vllm.

- Add a FA4 ViT warmup in `vllm/model_executor/warmup/kernel_warmup.py` (see `vllm/model_executor/warmup/fa4_warmup.py`).
- Scope: **Qwen3-VL / Qwen3-VL-MoE** vision transformer only, **Blackwell-only**, and only when `--mm-encoder-attn-backend FLASH_ATTN_CUTE` is set.
- Candidate seqlens (only varying seqlen): `[64, 256, 576, 1024, 2304, 4096, 9216, 16384, 36864, 65536]` (filtered by `vision_config.num_position_embeddings` if smaller).

### (3) Minor fixes for FA4 integration.

- In `vllm/model_executor/layers/rotary_embedding/common.py`, there is a logic of `if find_spec("flash_attn") is not None:`
  However, flash_attn original package is actually not installed, not `flash_attn.cute` is installed.
  Therefore, minor fix is needed for the import error.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)

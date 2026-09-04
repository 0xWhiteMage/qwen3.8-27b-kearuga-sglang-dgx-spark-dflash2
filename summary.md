# 🧙‍♂️ Kearuga Model Suite: Release Summary

## 📌 Certified Production Checkpoints

* **Target Model (27B)**:
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)
  * **Format**: Hybrid GPTQ-4o6 + FP8 + NVFP4 quantization (24.85 GB, 3 shards + MTP draft head)
  * **Architecture**: Qwen3.5 (Gated DeltaNet + sliding-window attention hybrid)
  * **Fidelity**: Served Fidelity-40 mean KL 0.0165, top-1 40/40, exact 20/40. Held-out full-vocab KL 0.0208 (top-1 95.0%).
  * **Quality**: Quality-200 objective 157/180 (GSM8K: 66, HumanEval: 39, IFEval: 34, Agentic: 18).
  * **Context**: 262,144 tokens native context.

* **Speculative Drafter (DFlash 2)**:
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)
  * **Precision**: Native BF16 with fused KV materialization active (3.58 GiB).
  * **Context**: 262,144 tokens.
  * **SGLang Kernel**: `fused_kv_materialization` is 100% ENABLED.

---

## 🎛️ Hardware & Execution Profile

* **Hardware Platform**: NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory).
* **Serving Image**: `lmsysorg/sglang@sha256:616a3e97…` (official, digest-pinned, no overlay needed).
* **Serving VRAM Envelope**: 24.85 GiB (Target) + 3.58 GiB (Drafter) + 32.00 GiB (1M-Token BF16 KV Pool) + ~4.0 GiB (PyTorch / CUDA runtime) = ~64.4 GiB total (>63.6 GiB headroom).
* **Launch Profiles**:
  * `start-dflash2.sh`: Interactive low-latency profile (30.9 tok/s C1 decode, 98.3 tok/s C4).
  * `start-eagle.sh`: High-concurrency agent profile (527–539 tok/s C32 throughput).

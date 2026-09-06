# 📜 Changelog

All notable changes to the Kearuga model suite and DGX Spark deployment stack are documented in this file.

---

## [v0.5.0] - 2026-09-05

### 🎯 New Production Checkpoint — Hybrid GPTQ-4o6 + FP8
* **Promoted [Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)**: New hybrid quantization checkpoint combining GPTQ with Four-Over-Six group scales (120 MLP tensors), NVFP4 AWQ (60 down_proj tensors), FP8 e4m3 (195 GDN/attention projections + 10 boundary MLP modules). Total footprint 24.85 GB.
* **Fidelity Validation**: Served Fidelity-40 mean KL **0.0165** (−50.6% vs previous NVFP4-only checkpoint), top-1 agreement **40/40**, exact 32-token continuation **20/40**. Held-out full-vocab KL **0.0208** (top-1 95.0%). Quality-200 objective **157/180**.
* **Speed**: Empirical Spark DFlash (2048 draft window): C1 decode **57 tok/s** (57 tok/s/stream, TTFT 264ms), C2 aggregate **51 tok/s** (40 tok/s/stream, TTFT 416ms), C4 aggregate **94 tok/s** (39 tok/s/stream, TTFT 480ms).

### ⚡ Serving Migration — Official Image, No Overlay
* **Migrated to `lmsysorg/sglang@sha256:616a3e97…`**: The Kearuga checkpoint has no NVFP4_AWQ layers, so the 5-file kernel overlay is no longer needed. TTFT improved −6%.
* **K=10 Draft Block**: Sweep-validated across K=8/10/12/16 as the only value not "worse" on any C1 domain. K=8 loses code/math 7–12%; K=12/16 lose prose/ifeval and C4.
* **KV Cache bf16**: Fidelity-first default — KL 0.0170→0.0165, exact 19→20/40, speed cost within noise.

### 📊 Hugging Face Integration
* **Model card published** at [0xWhiteMage/Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) with full fidelity analysis, quantization architecture, serving instructions, and cross-references to this repository.
* **Repository and model card interconnected**: Hugging Face model card links here for the serving suite and verification harness; this repository links to Hugging Face for the checkpoint and fidelity analysis.

---

## [v0.4.1] - 2026-08-22

### 🎯 Target Model & Checkpoint Integrity
* **Target Model Specification (`8ea86bdc...`)**: Certified complete 2,194-tensor ModelOpt NVFP4 target checkpoint on `0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4` with verified dual scale matrices (`weight_scale_2`, `input_scale`).
* **Multimodal Architecture Verification**: Confirmed 27 Vision Transformer blocks producing all 333 visual tensors (`model.visual.*`) remain lossless in BF16 alongside image/video preprocessor configurations.

### ⚡ Speculative Decoding Architecture
* **DFlash 2 Speculative Integration**: Paired the target with stock `z-lab/Qwen3.8-27B-DFlash2` (3.58 GiB native BF16), maintaining low-noise FP8 tap points (`[5, 19, 33, 47, 61]`) and activating SGLang's `fused_kv_materialization` CUDA kernel.

### 📊 Community Benchmark Synchronization & Hardware
* **Weschera Recipe Synchronization**: Aligned benchmark matrix with Weschera's latest DFlash 2 Block 10 Speed Profile (42.04 tok/s dedicated C1) and Block 8 Capacity Profile (120.58 tok/s C8), incorporating findings on SGLang fp8_gemm autotuning.
* **MiaAI-Lab & r0b0tlab Parity**: Verified comparative throughput numbers against MiaAI-Lab DSpark (51.5 tok/s) and r0b0tlab SM121 click-run recipes.

### 🛠️ Hardware & Launch Robustness
* **SM121 JIT Auto-Detection**: Optimized compiler configuration for Blackwell GB10 SM121 native execution.
* **Dynamic Local Path Mounts**: Implemented `MODEL_MOUNT_ARGS` in launchers to automatically bind-mount local host paths (`/workspace/...` or `/volume2/...`).
* **Verification Suite**: Passed all 15 gates in `bench/verify_all.py` with 100% compliance.

---

## [v0.4.0] - 2026-08-21

### 🎛️ Operational Parameters & Quality Controls
* **Reasoning Effort Controls**: Standardized default `REASONING_EFFORT=medium`, `T=0.6`, and `Top-P=0.95` across launchers and benchmark suites to ensure consistent decoding behavior.
* **Quality Dataset Audit**: Integrated the 200-question multi-domain verification dataset across GSM8K, HumanEval, IFEval, and agentic coding.
* **Cross-Platform Manifest Hardening**: Enforced bit-exact Linux LF line endings via `.gitattributes` to guarantee 100% cross-platform parity.

---

## [v0.3.0] - 2026-08-18

### ⚡ Speculative Decoding Framework
* **DFlash 2 Daily Driver Profile**: Integrated block-diffusion speculative decoding delivering high-speed net decode on single stream.
* **1M-Token KV Pool**: Implemented shared KV cache sustaining 4 concurrent native 262K contexts.

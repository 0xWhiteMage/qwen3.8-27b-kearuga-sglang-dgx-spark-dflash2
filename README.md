# 🧙‍♂️ Qwen3.8-27B Kearuga on a Single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang, DFlash 2 and EAGLE" width="100%"><br><br>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/release-v0.5.0-blue.svg?style=for-the-badge" alt="Version 0.5.0"></a>
  <a href="#-benchmarks"><img src="https://img.shields.io/badge/C1_Net_Decode-30.9_tok%2Fs-success.svg?style=for-the-badge" alt="C1 Net Decode"></a>
  <a href="#-benchmarks"><img src="https://img.shields.io/badge/C32_Aggregate-535_tok%2Fs-purple.svg?style=for-the-badge" alt="C32 Aggregate"></a>
  <a href="#-runtime-envelope"><img src="https://img.shields.io/badge/KV_Pool-1%2C048%2C576_Tokens-orange.svg?style=for-the-badge" alt="KV Pool"></a>
  <a href="#-saturated-responsiveness--priority-scheduling"><img src="https://img.shields.io/badge/Priority_TTFT-~2.6s-red.svg?style=for-the-badge" alt="Priority TTFT"></a><br><br>
  <a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Model-yellow.svg?style=for-the-badge" alt="HuggingFace Model"></a>
  <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/Follow_on_X-@0xWhiteMage-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a> ·
  <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Kofi-Buy_me_a_coffee-1A9642?style=for-the-badge&logo=buymeacoffee&logoColor=white" alt="Ko-fi"></a>
</p>

Run **[Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** paired with the **[Kearuga DFlash 2 Drafter](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)** on **[SGLang](https://docs.sglang.io)** on a single 128 GB NVIDIA DGX Spark (GB10). This repository provides production container builds, hardware launchers, kernel overlays, on-target distillation tooling, and automated quality benchmarks.

* ⚡ **DFlash 2 (Daily Driver)**: Ultra-responsive C1–C4 profile (~31 tok/s net C1, ~98 tok/s C4) with full reasoning & tool calling.
* 🦅 **EAGLE 3/1/4 (Agent Swarms)**: 32-seat high-concurrency profile for agent pipelines (~535 tok/s at C32).
* 📜 **1M-Token KV Pool**: Sustains **4 simultaneous native 262K contexts** in shared KV memory.
* 🛡️ **Tiered Sensitivity Hierarchy**: EXL3-inspired mixed-precision GPTQ-4o6 / NVFP4 / FP8 / BF16 preserving head fidelity and draft tap states across all model tensors.

> 📖 **Deep Architectural Rationale**: Read **[Kearuga: Architecture Insights & Design Rationale](INSIGHTS.md)** to understand why tiered sensitivity quantization and dual-engine speculative inference outperform conventional approaches.

> 🤗 **Model checkpoint**: The Kearuga target model lives on Hugging Face at **[0xWhiteMage/Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)**. The model card there contains the full fidelity analysis, quantization architecture, and serving instructions. This repository provides the serving suite that runs it.

---

## 📢 Recent Updates

See the complete release history in **[CHANGELOG.md](CHANGELOG.md)**.

* 🌟 **v0.5.0 Release Highlights**:
  * 🎯 **New Production Checkpoint**: Promoted **[Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** — a hybrid GPTQ-4o6 + FP8 quantization that halves served first-token KL divergence (0.0334 → 0.0165) at identical speed and a smaller footprint (24.85 GB).
  * � **Measured Fidelity**: Served Fidelity-40 mean KL **0.0165**, top-1 agreement **40/40**, exact 32-token continuation **20/40**. Held-out full-vocab KL **0.0208** (top-1 95.0%). Quality-200 objective **157/180** (gsm8k 66, humaneval 39, ifeval 34, agentic 18).
  * ⚡ **Official Image, No Overlay**: Migrated to `lmsysorg/sglang@sha256:616a3e97…` — the V5 checkpoint has no NVFP4_AWQ layers, so the 5-file kernel overlay is no longer needed. TTFT improved −6%.
  * 🎛️ **K=10 Draft Block**: Sweep-validated as the only value not "worse" on any C1 domain (K=8 loses code/math 7–12%, K=12/16 lose prose/ifeval and C4).
  * 🧊 **KV Cache bf16**: Fidelity-first default — KL 0.0170→0.0165, exact 19→20/40, speed cost within noise.

---

## 📊 Benchmarks & Community Comparison

> 🚧 **In active development** — we're actively developing an upgraded on-target DFlash 2 drafter. Benchmarks below reflect the current stock `z-lab/Qwen3.8-27B-DFlash2` drafter; updated figures will be published when the new drafter lands.

> *"DFlash 2 delivers instant interactive feedback (~31 tok/s C1); EAGLE scales massive agent swarms (535 tok/s C32)."*

### ⚡ 1. Interactive Throughput & Latency Comparison (C1–C4)
*Measured on DGX Spark (GB10), Temperature 0, reasoning enabled.*

| Solution / Repository | Speculative Method | Dedicated C1 (tok/s) | Net Decode C1 (tok/s) | Saturated C4 (tok/s) | Ladder C8 (tok/s) |
|---|---|---:|---:|---:|---:|
| 🧙‍♂️ **Kearuga Model Suite** | **DFlash 2 (BF16)** | **41.7** | **~31.0** | **~98.3** | **~135.0** |
| 🔹 [Weschera (Latest)](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 (Block 10 Speed Profile) | 42.04 | — | 66.31 | 114.50 |
| 🔹 [Weschera (Capacity)](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 (Block 8 Capacity Profile) | 34.08 | 25.33 | 64.10 | 120.58 |
| 🔹 [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | DSpark / DFlash 2 | ~50.9–51.5 | ~29.0–35.0 | 111.60 | — |
| 🔹 [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | MTP (In-Checkpoint) | ~26.0 | 33.0–35.0 | ~95.0 | — |
| 🔹 [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) | SM121 Pin (DFlash 2 K8) | 28.38 | 23.47 | 54.99 | 92.05 |

### 🦅 2. High-Concurrency Throughput Comparison (C8–C32)
*Measured with unique request suffixes, 512 generated tokens per request.*

| Profile & Repository | Speculative Engine | C8 Aggregate | C16 Aggregate | C32 Aggregate | Saturated p95 TTFT |
|---|---|---:|---:|---:|---:|
| 🦅 **Kearuga Model Suite** | **EAGLE 3/1/4 (32 Seats)** | **181–193 tok/s** | **320–335 tok/s** | **527–539 tok/s** | **0.32s / 0.49s / 0.87s** |
| 🔹 Kearuga DFlash 2 | DFlash 2 (4 Seats Max) | ~118 tok/s* | — | — | — |
| 🔹 [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark) | vLLM 4-bit (MTP) | 246.0 tok/s | — | — | — |
| 🔹 [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 Capacity | 116.8 tok/s | 153.9 tok/s | 178.3 tok/s | — |

*\*DFlash queues requests above batch size 4.*

### ⏱️ 3. Saturated Responsiveness & Priority Scheduling

| Server Load State | Default Priority TTFT | Interactive Priority (`priority: 100`) | Latency Reduction |
|---|---:|---:|---:|
| **DFlash 2 (All 4 Seats Busy)** | ~43 s | **~2.6 s** | **~94% Faster** |
| **EAGLE (All 32 Seats Busy)** | 73.30 s | **2.76 s** | **96.2% Faster** |

```json
{
  "model": "qwen3.8-27b-sglang",
  "priority": 100,
  "messages": [{"role": "user", "content": "Instant interactive code request"}]
}
```

---

### ⚖️ 4. Checkpoint Fidelity (Kearuga vs. Original Base BF16)

| | **Qwen/Qwen3.8-27B** (BF16 base) | **Qwen3.8-27B-Kearuga** | delta |
|---|---:|---:|---|
| **Footprint** | 51.8 GiB | **24.85 GB** | **−52%** |
| **Fidelity-40 mean KL** vs BF16 | 0 (self) | **0.0165** | — |
| **Fidelity-40 top-1 agreement** | 40/40 | **40/40** | preserved |
| **Fidelity-40 exact 32-token continuation** | 40/40 | **20/40** | 50% of near-tie continuations preserved |
| **Held-out full-vocab KL** (72k positions) | 0 (self) | **0.0208** | — |
| **Held-out top-1 agreement** | 100% | **95.0%** | −5.0 pt |
| **Quality-200 objective** | — | **157/180** | — |
| **C1 decode tok/s** (DFlash2, K=10) | ~14 (no spec) | **30.9** | **+121%** |
| **C4 decode tok/s** (DFlash2, K=10) | ~45 (no spec) | **98.3** | **+118%** |

Full methodology on the [Hugging Face model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga).

---

## 🎛️ Runtime Envelope & Memory Allocation

> *"Four full native 262K contexts operating concurrently in a 1,048,576-token shared pool."*

| Architectural Dimension | Specification | Operational Details |
|---|---:|---|
| 🧠 **Per-Request Context** | `262,144` tokens | Native Qwen3.8 context length (YaRN disabled) |
| 📜 **Shared Target KV Pool** | `1,048,576` tokens | Four simultaneous 262K requests without memory exhaustion |
| 👥 **Admitted Concurrency** | `4` (DFlash) / `32` (EAGLE) | Governs maximum active parallel decode graphs |
| 💾 **Target KV Allocation** | `32.0 GiB` | Allocated in bf16 KV (fidelity-first) |
| ⚡ **Target Weight Footprint** | `24.85 GiB` | Hybrid GPTQ-4o6 + FP8 + NVFP4 (3 shards + MTP draft head) |
| 🏎️ **Drafter Footprint** | `3.58 GiB` | BF16 (fused KV materialization active) |
| 🛡️ **Total Serving VRAM** | **~60 GiB** | Fits with ~68 GiB headroom on 128 GB Unified Memory |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory)
* Docker with NVIDIA Container Toolkit (`--gpus all`)
* Linux kernel with unified memory support

### 2. Clone & Configure
```bash
git clone https://github.com/0xWhiteMage/Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2.git
cd Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2
cp .env.sample .env
```

### 3. Launch Serving Engines

* **Daily Driver (DFlash 2 Interactive C1–C4)**:
  ```bash
  ./start-dflash2.sh
  ```
* **High-Concurrency Agent Swarms (EAGLE C8–C32)**:
  ```bash
  ./start-eagle.sh
  ```
* **Stop Server**:
  ```bash
  ./stop.sh
  ```

The launcher automatically pulls the Kearuga checkpoint from [Hugging Face](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) on first run. No manual download or kernel overlay is needed — the V5 checkpoint boots on the official SGLang image with no patches.

---

## 🧪 Verification & Benchmarking

Run the complete verification harness:
```bash
python3 bench/verify_all.py
```

Run semantic canary and speculative decoding benchmarks:
```bash
python3 bench/semantic_gate.py
python3 bench/ndec.py
python3 bench/run_quality_set.py
```

---

## 🤗 Hugging Face Integration

The Kearuga model suite is split across two platforms:

| artifact | location | purpose |
|---|---|---|
| **Target checkpoint** | [`0xWhiteMage/Qwen3.8-27B-Kearuga`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) | The quantized 27B model weights (24.85 GB, 3 shards + MTP head) with the full model card, fidelity analysis, and serving instructions |
| **DFlash 2 drafter** | [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) | The speculative decoding draft model (BF16, 3.58 GiB) |
| **Serving suite** | This repository | Container builds, launchers, benchmarks, distillation tooling |

The Hugging Face model card and this repository cross-reference each other. The model card links here for the serving suite and verification harness; this repo links to Hugging Face for the checkpoint and fidelity analysis.

---

## 📄 License & Citations
Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

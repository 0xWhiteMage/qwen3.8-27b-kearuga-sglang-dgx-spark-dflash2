# 🧙‍♂️ Qwen3.8-27B Kearuga on a Single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang and DFlash 2" width="100%"><br><a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Release-v0.6.1-blue.svg" alt="Version 0.6.1"></a> <a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Target_Model-yellow.svg" alt="HuggingFace Model"></a> <a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Kearuga_DFlash2_Drafter-orange.svg" alt="Kearuga DFlash 2 Drafter"></a> <a href="https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Stock_Drafter_(fallback)-lightgrey.svg" alt="Stock DFlash 2 Drafter (fallback)"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License: Apache 2.0"></a> <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/X-@0xWhiteMage-000000?logo=x&logoColor=white" alt="Follow on X"></a> <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Ko--fi-Donate-FF5E5B?logo=ko-fi&logoColor=white" alt="Donate on Ko-fi"></a>
</p>

Serve **[Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** paired with its own DFlash 2 drafter — **[Qwen3.8-27B-Kearuga-DFlash2](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)** (Kearuga-distilled, Kearuga-calibrated NVFP4, 64K draft head) — on **[SGLang](https://docs.sglang.io)** on a single 128 GB NVIDIA DGX Spark (GB10 / SM121). `DRAFTER_PROFILE=stock` keeps the v0.5.0 stock `z-lab` recipe.

This repository provides certified production container launchers, hardware configurations, priority preemption queues, and automated multi-gate verification suites.

* ⚡ **DFlash 2 (Interactive Daily Driver)**: Ultra-responsive C1–C4 profile with full reasoning & tool-calling support — paired bake-off aggregate throughput of **35.33 tok/s C1 (+14.5 %)** and **108.90 tok/s C4 (+10.8 %)** vs the stock drafter (see the paired table below).
* 📜 **1M-Token KV Pool**: Sustains **4 simultaneous native 262K contexts** in shared unified memory without swapping or fragmentation.
* 🛡️ **Tiered Sensitivity Hierarchy**: EXL3-inspired mixed-precision (GPTQ-4o6 / NVFP4 AWQ / FP8 / BF16) preserving vocabulary logit tails and intermediate draft taps.
* ⏱️ **Priority Queue Preemption**: Sub-3-second interactive response under full multi-session saturation via native priority scheduling.
* 🚀 **Official digest-pinned image + a 5-file read-only Python overlay**, hash-verified at launch; no Docker build, no kernel patch. `DRAFTER_PROFILE=stock` runs the unmodified image.

> 📖 **Deep Architectural Rationale**: Read **[Kearuga: Architecture Insights & Design Rationale](INSIGHTS.md)** for an in-depth analysis of our tiered quantization map, Four-Over-Six group scaling, and DFlash 2 block-diffusion serving.
>
> 🤗 **Model Checkpoint**: The production target weights are hosted on Hugging Face at **[0xWhiteMage/Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** (24.85 GB, 3 shards + MTP head).

---

## 📢 Recent Updates

See the complete chronological release history in **[CHANGELOG.md](CHANGELOG.md)**.

### 🌟 v0.6.0 Release Highlights
* ⚡ **Kearuga's Own DFlash 2 Drafter ([Qwen3.8-27B-Kearuga-DFlash2](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2))**: distilled on Kearuga's own outputs, Kearuga-calibrated NVFP4 (1.55 GB vs 3.85 GB stock BF16), served with a frequency-pruned 64K draft head — **+14.5 % C1 / +10.8 % C4 aggregate** vs the stock drafter in a paired bake-off, better on all 10 domain cells.
* 🧩 **Vendored Draft-Head Overlay (`drafter/`)**: five read-only Python files mounted over the pinned image plus the 65,650-row token map; both `drafter/SHA256SUMS` and the image's base-file hashes are verified at launch. No Docker build, no kernel patch.
* 🎛️ **`DRAFTER_PROFILE` Launcher Switch**: `kearuga` (default, K=12 + overlay) or `stock` (the exact v0.5.0 recipe); `DRY_RUN=1` runs every pre-flight check and prints the full `docker run` command without starting anything.
* ✅ **Quality-200 155/180 & Fidelity-40 unchanged** (mean KL 0.0165, top-1 40/40): the target verifies every drafted token, so the drafter changes speed, not output.

---

## 📊 Benchmarks & Community Comparison

> *"DFlash 2 delivers instant interactive feedback with four simultaneous native 262K contexts in shared unified memory."*

### ⚡ 0. Kearuga drafter vs stock drafter (paired)

Paired bake-off, 2026-09-17 — control and candidates booted back-to-back on the same image, target, flags and clock (GB10 clock-capped at 2400 MHz, 262K context enabled). 50-prompt battery (10 prompts × 5 domains) × 2 runs, `temperature 0`, thinking off. *"Net decode tok/s"* = `(completion_tokens − 1) / (t_last_token − t_first_token)` per request, TTFT excluded; *"aggregate tok/s"* = Σ completion tokens / cell wall time.

| Serving configuration | C1 aggregate | C4 aggregate | Domains better (C1 + C4) |
|---|:---:|:---:|:---:|
| Stock BF16 `z-lab` drafter, K = 10 (baseline) | 30.85 tok/s | 98.31 tok/s | — |
| Kearuga drafter, K = 12, plain SGLang | 33.30 tok/s (+7.9 %) | 102.57 tok/s (+4.3 %) | 10 / 10 |
| **Kearuga drafter, K = 12, + 64K draft head (overlay)** | **35.33 tok/s (+14.5 %)** | **108.90 tok/s (+10.8 %)** | **10 / 10** |

| Concurrency | Domain | Stock drafter med. tok/s | Kearuga drafter + overlay med. tok/s | Paired Δ |
|:---:|---|:---:|:---:|:---:|
| C1 | code | 46.27 | **57.62** | **+23.1 %** |
| C1 | math | 51.21 | **62.18** | **+24.2 %** |
| C1 | tool calls | 80.82 | **114.63** | **+26.2 %** |
| C1 | prose | 19.45 | **21.64** | **+10.8 %** |
| C1 | instruction-following | 19.87 | **21.80** | **+9.4 %** |
| C4 | code | 38.84 | **45.65** | **+16.1 %** |
| C4 | math | 41.17 | **47.33** | **+13.0 %** |
| C4 | tool calls | 49.02 | **69.25** | **+31.5 %** |
| C4 | prose | 16.07 | **17.22** | **+8.6 %** |
| C4 | instruction-following | 15.42 | **17.68** | **+11.0 %** |

Median C1 TTFT moved from 0.28–0.30 s to 0.27–0.28 s (tool prompts 0.78 → 0.62 s). Quality-200 (frozen suite): **155/180** with the Kearuga drafter vs 157 for the target alone — inside the 154–159 near-tie band every gated drafter lands in. Fidelity-40 vs the BF16 base: mean KL 0.0165, top-1 40/40 — unchanged by the drafter, because the target verifies every token. Full methodology, acceptance lengths and the ranked candidate table are on the [drafter model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2).

### ⚡ 1. Interactive Throughput & Community Comparison (C1–C4)
*Measured on NVIDIA DGX Spark (GB10 / SM121), Temperature 0, reasoning enabled. Community numbers come from each author's own harness and are not paired with ours.*

| Solution / Repository | Speculative Method | Dedicated C1 (tok/s) | Net Decode C1 (tok/s) | Saturated C4 (tok/s) | Ladder C8 (tok/s) |
|---|---|---:|---:|---:|---:|
| 🧙‍♂️ **Kearuga Model Suite** | **DFlash 2 (stock drafter, v0.5.0 single-prompt probe)** | **57.0** | **57.0** | **94.0** | — |
| 🔹 [MiaAI-Lab (DFlash 2)](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | DFlash 2 / DSpark | ~50.9–51.5 | ~29.0–35.0 | 111.60 | — |
| 🔹 [MiaAI-Lab (MTP)](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | MTP (In-Checkpoint) | ~26.0 | 33.0–35.0 | ~95.0 | — |
| 🔹 [Weschera (Speed Profile)](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 (Block 10) | 42.04 | — | 66.31 | 114.50 |
| 🔹 [Weschera (Capacity Profile)](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 (Block 8) | 34.08 | 25.33 | 64.10 | 120.58 |
| 🔹 [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) | SM121 Pin (DFlash 2 K8) | 28.38 | 23.47 | 54.99 | 92.05 |

*Note: our row was measured with the stock DFlash 2 drafter, K=10 and 2048 draft window as a v0.5.0 single-prompt probe: C1 = 57 tok/s (57 tok/s/stream, TTFT 264ms), C2 = 51 tok/s aggregate (40 tok/s/stream, TTFT 416ms), C4 = 94 tok/s aggregate (39 tok/s/stream, TTFT 480ms). It is not measured with the same battery method as the paired table above. In pure unconstrained decode without reasoning traces (`enable_thinking=false`), DFlash 2 reaches 65+ tok/s.*

### ⏱️ 2. Saturated Responsiveness & Priority Scheduling

When running concurrent agent workers, interactive developer chats cannot wait for queue drain. SGLang's native priority scheduling preempts background requests:

```json
{
  "model": "qwen3.8-27b-sglang",
  "priority": 100,
  "messages": [{"role": "user", "content": "Fix this unit test immediately"}]
}
```

| Server Saturation State | Default Priority TTFT | Interactive Priority (`priority: 100`) | Latency Reduction |
|---|---:|---:|---:|
| **DFlash 2 (All 4 Seats Busy)** | ~43.15 s | **~2.63 s** | **93.9% Faster** |

---

### ⚖️ 3. Checkpoint Fidelity (Kearuga vs. Original Base BF16)

| Metric | Base Model: `Qwen/Qwen3.8-27B` (Native BF16) | **Qwen3.8-27B-Kearuga** (This Suite) | Delta / Rationale |
|---|:---:|:---:|:---|
| **Weight Footprint** | 51.8 GiB | **24.85 GB** | **−52.0%** footprint |
| **Fidelity-40 Mean KL** | 0.0000 | **0.0165** | Near-zero distribution divergence |
| **Fidelity-40 Mean JS Divergence** | 0.0000 | **0.0034** | High distributional stability |
| **Fidelity-40 Top-1 Agreement** | 40 / 40 | **40 / 40** | 100% exact argmax token match |
| **Fidelity-40 Exact 32-Token Match** | 40 / 40 | **20 / 40** | 50% byte-identical; ties flip to valid equivalents |
| **Held-Out Full-Vocab KL (72k pos)** | 0.0000 | **0.0208** | Evaluated on out-of-domain sequences |
| **Held-Out Top-1 Agreement** | 100.0% | **95.0%** | −5.0 pt drop over raw vocabulary |
| **C1 Decode tok/s (stock drafter, K=10, v0.5.0 single-prompt probe)** | ~14.0 (no spec) | **57.0** | **+307% speedup** (57 tok/s/stream, TTFT 264ms) |
| **C2 Decode tok/s (stock drafter, K=10, v0.5.0 probe)** | ~28.0 (no spec) | **51.0** | **+82% speedup** (40 tok/s/stream, TTFT 416ms) |
| **C4 Decode tok/s (stock drafter, K=10, v0.5.0 probe)** | ~45.0 (no spec) | **94.0** | **+109% speedup** (39 tok/s/stream, TTFT 480ms) |

---

## 🎛️ Runtime Envelope & Memory Math

> *"Four full native 262K contexts operating concurrently in a 1,048,576-token shared pool."*

| Allocation Layer | Memory Footprint | Format / Allocation Scope |
|---|:---:|---|
| **Target Model Weights** | 24.85 GiB | Hybrid GPTQ-4o6 + FP8 + NVFP4 (3 shards + MTP head) |
| **Speculative Drafter** | ~2.1 GiB | Kearuga DFlash 2 (NVFP4 weights 1.45 GiB + 64K BF16 draft head 0.63 GiB, computed from file sizes) |
| **Shared Target KV Cache** | 32.00 GiB | 1,048,576 shared tokens in Native BF16 (4 × 262K contexts) |
| **PyTorch & SGLang Runtime** | ~4.00 GiB | CUDA context, execution buffers, and scratchpad memory |
| **Total Active Serving Footprint** | **~62.9 GiB** | Allocated out of 128 GB Unified Memory |
| **Free System Headroom** | **~65.1 GiB** | Available for OS, page cache, and dynamic context growth |

*Stock (`DRAFTER_PROFILE=stock`) figures: BF16 drafter 3.58 GiB, total ~64.4 GiB, headroom ~63.6 GiB.*


| Architectural Dimension | Specification | Operational Details |
|---|---:|---|
| 🧠 **Per-Request Context** | `262,144` tokens | Native Qwen3.8 window (YaRN interpolation disabled) |
| 📜 **Shared Target KV Pool** | `1,048,576` tokens | Four simultaneous 262K requests without memory exhaustion |
| 👥 **Admitted Concurrency** | `4` concurrent streams | Governs maximum active parallel decode graphs (DFlash 2) |
| 💾 **Target KV Allocation** | `32.00 GiB` | Allocated in BF16 KV cache (fidelity-first profile) |
| ⚡ **Target Weight Footprint** | `24.85 GiB` | Hybrid GPTQ-4o6 + FP8 + NVFP4 (3 shards + MTP head) |
| 🏎️ **Drafter Footprint** | `~2.1 GiB` | Kearuga NVFP4 drafter + 64K BF16 draft head (computed from file sizes; stock BF16 = 3.58 GiB) |
| 🛡️ **Total Serving Footprint** | **~62.9 GiB** | Fits with **>65 GiB headroom** on 128 GB Unified Memory |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory)
* Docker with NVIDIA Container Toolkit (`--gpus all`)
* Linux kernel with unified memory support

### 2. Clone & Setup
```bash
git clone https://github.com/0xWhiteMage/Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2.git
cd Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2
cp .env.sample .env
```

### 3. Launch Serving Engine

* **Start Server (DFlash 2 Interactive C1–C4)** — defaults to the Kearuga drafter (`DRAFTER_PROFILE=kearuga`):
  ```bash
  ./start-dflash2.sh
  ```
* **Stop Server**:
  ```bash
  ./stop.sh
  ```

On the `kearuga` profile the launcher first verifies `drafter/SHA256SUMS` (vendored overlay + token map) and the pinned image's base-file hashes (`drafter/BASE-SHA256SUMS`), then mounts five overlay files read-only and passes the token map. A correct boot prints `[kearuga-drafthead] loaded hot token map: rows=65650 …` and `[kearuga-drafthead] draft head rows=65650 dtype=torch.bfloat16 bytes=672256000 fp8=False hot_map=True`.

**Preview without starting anything:**
```bash
DRY_RUN=1 ./start-dflash2.sh   # runs all checks, prints the expanded docker run command, exits
```

**Switching profiles / falling back:**
```bash
./stop.sh && DRAFTER_PROFILE=stock ./start-dflash2.sh   # exact v0.5.0 recipe: stock BF16 drafter, K=10, no overlay
```

*The launcher automatically pulls the verified Kearuga checkpoint and DFlash drafter from Hugging Face on first run. No manual weight conversion or kernel patching is required.*

---

## 🧪 Verification & Benchmarking

Run the complete 13-gate verification harness:
```bash
python3 bench/verify_all.py
```

Run specialized quality and latency suites:
```bash
python3 bench/semantic_gate.py   # Verify exact arithmetic & decimal comparison canaries
python3 bench/niah.py            # Test Needle-In-A-Haystack at 64K depth
python3 bench/run_quality_set.py # Run 200-sample multi-domain benchmark
python3 bench/ndec.py            # Measure net decode throughput
python3 bench/priority_ttft.py   # Benchmark priority queue preemption latency
```

### 📐 Fidelity & Quality Evaluation Set

This repository includes a **frozen, reproducible evaluation suite** under [`eval/`](eval/) so others can compare their quantized Qwen3.8-27B checkpoints against Kearuga on equal footing:

- **KLD-40 (Fidelity-40)** — 40 prompts that measure top-20 token KL/JS divergence, top-1 agreement, and exact 32-token continuation match against a served BF16 reference. The BF16 reference capture is included so you can score without running BF16 yourself.
- **Quality-200** — 200 prompts across GSM8K, HumanEval, IFEval, agentic coding, and hard reasoning, with per-family objective grading.

All files are SHA-256 pinned and verified by `check_kld_manifest.py`. See **[eval/README.md](eval/README.md)** for full usage instructions.

Quick start:
```bash
# Capture KLD-40 from your served model
python3 eval/frozen/kld_capture_score.py capture \
  --prompts eval/frozen/kld-prompts-40.json \
  --base-url http://localhost:8890 \
  --output my-capture.json

# Score against the included BF16 reference
python3 eval/frozen/kld_capture_score.py score \
  --reference eval/frozen/kld-bf16-reference-20260824.json \
  --candidate my-capture.json \
  --output my-score.json
```

---

## 🤗 Hugging Face Model Suite

The Kearuga deployment suite is interconnected across GitHub and Hugging Face:

| Artifact | Location | Purpose |
|---|---|---|
| **Target Checkpoint (27B)** | [`0xWhiteMage/Qwen3.8-27B-Kearuga`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) | Production weights (24.85 GB, 3 shards + MTP head), model card, and fidelity proofs |
| **Kearuga DFlash 2 Drafter** | [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) | Kearuga-distilled, Kearuga-calibrated NVFP4 drafter (1.55 GB) + 64K draft-head map and SGLang overlay |
| **Stock DFlash 2 Drafter** | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Fallback / baseline speculative draft model (BF16, 3.58 GiB) — `DRAFTER_PROFILE=stock` |
| **Serving Suite & Harness** | This Repository | Production container scripts, benchmarks, and hardware launchers |

---

## 🙏 Credits

This project stands on other people's work:

* **[z-lab](https://huggingface.co/z-lab)** — DFlash / DFlash 2, the block-diffusion drafter architecture and the stock [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) weights our drafter is fine-tuned from (Apache-2.0).
* **[Inco](https://inco.ai) / [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)** — the published per-task acceptance table used as the public yardstick, and the DFlash 2 write-up.
* **[SGLang](https://github.com/sgl-project/sglang) (LMSYS / RadixArk)** — the serving engine, its DFLASH spec-v2 worker and overlap scheduler.
* **[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** — the NVFP4 checkpoint layout our drafter quantization reproduces bit-for-bit.
* **[maurienne-ai](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)**, **[tcclaviger](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8/discussions/1)**, **[lued](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8)**, **[josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8)**, **[syvai](https://huggingface.co/syvai)**, **[gratex](https://huggingface.co/gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ)**, **[YourHighnessLA](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4)**, **[TechPrototyper](https://huggingface.co/TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm)**, **[Mia-AiLab](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)** — the community drafter quantizations and measurements that showed the bytes curve flattens past FP4.
* **[DaoCloud](https://huggingface.co/DaoCloud/Qwen3.8-27B-DFlash2-Exp)**, **[alphakek](https://huggingface.co/alphakek/Qwen3.8-27B-heretic-ara-DFlash2)**, **[mrchuy](https://huggingface.co/mrchuy/Qwen3.8-27B-DFlash-drafter-bootstrap-GGUF)**, **[AEON-7](https://github.com/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED)**, **[sxuff](https://github.com/sxuff/qwen38-27b-nvfp4-dflash2-dgx-spark)** — drafter training and adaptive-K recipes we studied.
* **FR-Spec** — Zhao et al., *Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling*, ACL 2025 ([arXiv 2502.14856](https://arxiv.org/abs/2502.14856), [thunlp/FR-Spec](https://github.com/thunlp/FR-Spec)) — the frequency-pruned draft vocabulary behind the 64K draft head.
* **Spec-AUF** ([arXiv 2607.01893](https://arxiv.org/abs/2607.01893)), **D-PACE** ([arXiv 2605.18810](https://arxiv.org/abs/2605.18810)) and **Draft-OPD** ([arXiv 2605.29343](https://arxiv.org/abs/2605.29343)) — drafter-training objectives implemented and piloted.
* Serving-stack community credits are in [INSIGHTS.md §7](INSIGHTS.md#-7-acknowledgements--community-credits) (malaiwah, MiaAI-Lab, Weschera, r0b0tlab, 0xBakeer, z-lab, SGLang).

---

## 📄 License & Citations
Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details. Base model weights are governed by Alibaba Cloud's original license terms.

If you build upon this work, please cite:
```bibtex
@misc{kearuga-2026,
  title={Kearuga: Hybrid GPTQ-4o6 + FP8 Quantization of Qwen3.8-27B for Speculative-Decoding Serve on NVIDIA DGX Spark},
  author={0xWhiteMage},
  year={2026},
  publisher={GitHub},
  url={https://github.com/0xWhiteMage/qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2}
}
```

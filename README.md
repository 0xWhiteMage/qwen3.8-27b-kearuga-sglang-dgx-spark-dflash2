# 🧙‍♂️ Qwen3.8-27B Kearuga on a Single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang and DFlash 2" width="100%"><br><a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Release-v0.6.4-blue.svg" alt="Version 0.6.4"></a> <a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Target_Model-yellow.svg" alt="HuggingFace Model"></a> <a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Kearuga_DFlash2_Drafter-orange.svg" alt="Kearuga DFlash 2 Drafter"></a> <a href="https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Stock_Drafter_(fallback)-lightgrey.svg" alt="Stock DFlash 2 Drafter (fallback)"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License: Apache 2.0"></a> <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/X-@0xWhiteMage-000000?logo=x&logoColor=white" alt="Follow on X"></a> <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Ko--fi-Donate-FF5E5B?logo=ko-fi&logoColor=white" alt="Donate on Ko-fi"></a>
</p>

Serve **[Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** paired with its own DFlash 2 drafter — **[Qwen3.8-27B-Kearuga-DFlash2](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) v1.0** (Kearuga-distilled, Kearuga-calibrated NVFP4, 64K draft head) — on **[SGLang](https://docs.sglang.io)** on a single 128 GB NVIDIA DGX Spark (GB10 / SM121). `DRAFTER_PROFILE=stock` keeps the v0.5.0 stock `z-lab` recipe.

This repository provides certified production container launchers, hardware configurations, priority preemption queues, and automated multi-gate verification suites.

* ⚡ **DFlash 2 (Interactive Daily Driver)**: Ultra-responsive C1–C4 profile with full reasoning & tool-calling support — paired bake-off aggregate throughput of **35.33 tok/s C1 (+14.5 %)** and **108.90 tok/s C4 (+10.8 %)** vs the stock drafter (see the paired table below).
* 📜 **818K-Token Shared KV Pool** (measured at boot): four seats, each with the native 262K window, sharing ≈ 818,294 tokens of BF16 KV — no swapping, no fragmentation.
* 🛡️ **Tiered Sensitivity Hierarchy**: EXL3-inspired mixed-precision (GPTQ-4o6 / NVFP4 AWQ / FP8 / BF16) preserving vocabulary logit tails and intermediate draft taps.
* ⏱️ **Priority Queue Preemption**: Sub-3-second interactive response under full multi-session saturation via native priority scheduling.
* 🚀 **Official digest-pinned image + a 5-file read-only Python overlay**, hash-verified at launch; no Docker build, no kernel patch. `DRAFTER_PROFILE=stock` runs the unmodified image.

> 📖 **Deep Architectural Rationale**: Read **[Kearuga: Architecture Insights & Design Rationale](INSIGHTS.md)** for an in-depth analysis of our tiered quantization map, Four-Over-Six group scaling, and DFlash 2 block-diffusion serving.
>
> 🤗 **Model Checkpoint**: The production target weights are hosted on Hugging Face at **[0xWhiteMage/Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** (24.85 GB, 3 shards + MTP head).

---

## 📢 Recent Updates

See the complete chronological release history in **[CHANGELOG.md](CHANGELOG.md)**.

### 🌟 v0.6.4 Release Highlights (v0.6.0 → v0.6.4, all 2026-09-17)
* ⚡ **Kearuga's Own DFlash 2 Drafter v1.0 ([Qwen3.8-27B-Kearuga-DFlash2](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2))**: distilled on Kearuga's own outputs, Kearuga-calibrated NVFP4 (1.55 GB vs 3.85 GB stock BF16), served with a frequency-pruned 64K draft head — **+14.5 % C1 / +10.8 % C4 aggregate** vs the stock drafter in a paired bake-off, better on all 10 domain cells.
* 🧩 **Vendored Draft-Head Overlay (`drafter/`)**: five read-only Python files mounted over the pinned image plus the 65,650-row token map; both `drafter/SHA256SUMS` and the image's base-file hashes are verified at launch. No Docker build, no kernel patch.
* 🎛️ **`DRAFTER_PROFILE` Launcher Switch**: `kearuga` (default, K=12 + overlay) or `stock` (the exact v0.5.0 recipe); `DRY_RUN=1` runs every pre-flight check and prints the full `docker run` command without starting anything.
* ✅ **Quality-200 155/180 & Fidelity-40 unchanged** (mean KL 0.0165, top-1 40/40): the target verifies every drafted token, so the drafter changes speed, not output.
* 📊 **v0.6.2 — Community comparison refreshed and memory math corrected**: 17 GB10 / Qwen3.8-27B sources re-read on 2026-09-17 with a per-row metric column; our row re-measured with the repo's own `bench/scale.py` on the Kearuga profile; the KV-pool / footprint figures now come from boot logs (818,294-token pool, ~100 GB static) — the earlier 1,048,576-token / 32 GiB / ~63 GiB figures were wrong. **v0.6.3**: re-measured with the GPU clock lock lifted — identical numbers; the v0.5.0 57 / 51 / 94 row is marked historical (not reproduced).

---

## 📊 Benchmarks & Community Comparison

> *"DFlash 2 delivers instant interactive feedback across four concurrent seats in shared unified memory."*

### ⚡ 1. Kearuga drafter vs stock drafter (paired)

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

Median C1 TTFT moved from 0.28–0.30 s to 0.27–0.28 s (tool prompts 0.78 → 0.62 s). Exact acceptance on the production resident (K=12, 64K head; n=2 prompts per domain, 2026-09-17): greedy code 7.5 · math 7.7 · tool 10.5 · prose 2.5 · IFEval 2.4 tokens/cycle; with the model's default sampling (T 1.0, top-p 0.95, top-k 20) 6.5 · 7.4 · 9.7 · 2.5 · 2.3; **with thinking on, code falls to 3.1** (the reasoning trace behaves like prose) — the drafter's largest remaining headroom. Quality-200 (frozen suite): **155/180** with the Kearuga drafter vs 157 for the target alone — inside the 154–159 near-tie band every gated drafter lands in. Fidelity-40 vs the BF16 base: mean KL 0.0165, top-1 40/40 — unchanged by the drafter, because the target verifies every token. Full methodology, acceptance lengths and the ranked candidate table are on the [drafter model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2).

### ⚡ 2. Interactive Throughput & Community Comparison (C1–C4)

Every number below is what its author measured with their own harness, prompt set, target quantization and clock — indicative, not paired. The **Metric** column says exactly what each number is; do not compare across rows with different definitions. Sources were re-read on 2026-09-17 (raw copies archived); dates are each repo's last commit. Our row was measured on the owner's DGX Spark twice on 2026-09-17 — once under the owner's 2400 MHz SM clock lock and once with the lock lifted — with identical results (C1 44.7 vs 44.5, C4 131.3 vs 131.1): unlocked, the GPU ran this workload at 2.40–2.48 GHz by itself (GB10 nominal max 3003 MHz, no throttle flags), so the clock is not a factor in these numbers.

**Single DGX Spark (GB10) · SGLang + DFlash 2 — the comparable lane**

| Solution (last update) | Target checkpoint | Drafter · K | C1 (tok/s) | C4 aggregate | C8+ aggregate | Metric (as measured) |
|---|---|---|---|---:|---:|---|
| 🧙‍♂️ **Kearuga — drafter v1.0, serving stack v0.6.4** (2026-09-17) | **Kearuga hybrid GPTQ-4o6 + FP8 + NVFP4, 24.85 GB** (Fidelity-40 KL 0.0165) | **Kearuga NVFP4 DFlash 2 + 64K draft head · K=12** | **44.7** aggregate (45.5 net decode) on `bench/scale.py`; battery medians: code **57.6** · math 62.2 · tool calls 114.6 · prose 21.6 | **131.3** (≈ 36/stream, TTFT 0.34 s) | — (4 seats by design) | `scale.py`: fixed code prompt, 512 forced tokens, T=0, thinking off, aggregate incl. TTFT (0.23 s); C2 = 78–94 across runs. Battery = 10 prompts × 5 domains, net decode, TTFT excluded. Clock lock on or off: same result |
| 🧙‍♂️ Kearuga v0.5.0 (2026-09-05, historical) | same target | stock z-lab BF16 · K=10 | 57.0 | 94.0 (39/stream) | — | `scale.py`, same prompt; C2 = 51. **Not reproduced**: the raw run was not kept and today the same script on the current drafter — which is +23 % on code vs this drafter in the paired bake-off (§1) — gives 44.7; treat this row as documentation of the v0.5.0 release, not as a current measurement |
| [hasso5703](https://github.com/hasso5703/dgx-spark-qwen38) (2026-09-15) | Qwen3.8-27B NVFP4 (uniform) | maurienne calibrated NVFP4 DFlash 2 · depth 16 | 65 greedy median (code 64–66, math peak 71; free prose 18–25) | — | c8 148+, c32 258 | `bench.sh` streaming decode rate net of TTFT; deterministic kernels (`--disable-flashinfer-autotune`) |
| [sxuff](https://github.com/sxuff/qwen38-27b-nvfp4-dflash2-dgx-spark) (2026-09-10) | RadixArk NVFP4 | maurienne NVFP4 DFlash 2 · D=16 | 71.50 median whole-request (57.11 with z-lab BF16 at D=8) | — | c10 smoke 146 | completion tokens ÷ whole-request wall incl. prefill; 27 frozen requests × 3; the author notes drafter, depth and image changed together |
| [anliang0306](https://github.com/anliang0306/dgx-spark-qwen38) (2026-09-14) | RadixArk NVFP4 | z-lab BF16 · 8 draft tokens | code 43.2 · math 51.3 · prose 22.1 | — | c8 157.7 (10-round soak 143–168); c16 161.4 (TTFT 7.6 s) | single-stream greedy, thinking off |
| [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) (2026-09-09) | RadixArk NVFP4 | z-lab BF16 | `ndec` code 50.9 · essay 25.4; ladder c1 56.6 | 111.6 (33/stream) | c8 184.9, c16 227.6 (TTFT 4.2 s) | `ndec.py` two-call net-decode delta; ladder = synthetic structural-decode fixture, aggregate |
| MiaAI-Lab (MTP / EAGLE) | RadixArk NVFP4 | in-checkpoint MTP | `ndec` code 34.5 · essay 24.1 | — | — | same `ndec.py` |
| [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) speed (2026-08-21) | RadixArk NVFP4 + lm_head adapter | z-lab BF16 · block 10 | 42.04 dedicated (512 in / 2048 out) | 66.31 | c8 114.5 | fixed-length synthetic, thinking off, gate-qualified boots; documents a 42-vs-33 "boot lottery" from the fp8 autotuner |
| Weschera capacity | same | z-lab BF16 · block 8 | 34.08 | 64.10 | c8 120.58, c32 176–178 | same |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) (2026-08-19) | NVFP4 (uniform) | z-lab BF16 · K=8 | 28.38 dedicated; ladder c1 23.47 | 54.99 | c8 92.05 | thinking off; paired against DSpark K7 on the same card |

**Single DGX Spark (GB10) · other engines**

| Solution | Engine | Target · drafter | Numbers (tok/s) | Metric |
|---|---|---|---|---|
| [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark) | vLLM 0.27.1 | Unsloth NVFP4 · DSpark k=7 / k=14 | C1 "fresh" 29.2 / "edit" 58.6–59.1 (k=7), "edit" 72.6–75.0 (k=14); c4 167.7, c8 246 (k=7); c16 256.5 | fresh = 400-token module, edit = ~2K-in / 3K-out rewrite; T=0, thinking off; gpu-memory-utilization 0.85 |
| [tcclaviger](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8/discussions/1) et al. | vLLM (PR #52816) | RadixArk NVFP4 · z-lab BF16 / lued W8A16 / josch FP8 / syvai INT4, n_spec 7 | C1 41.3 / 43.2 / 43.8 / 44.6 at acceptance 4.1–4.2 | T=0.6, n=24 code + prose prompts — the drafter-bytes curve that shows the format lever flattens past FP8/FP4 |
| [Mia-AiLab](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) | ExLlamaV3 | EXL3 3.5 bpw · EXL3 5.0 bpw DFlash 2 | C1 47.5 (HumanEval, T=0.6; 35.7 without the drafter) | single stream |
| [AEON-7](https://github.com/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED) | vLLM 0.29 (omni image) | NVFP4-MIXED uncensored fine-tune · z-lab DFlash 2 "dynamic lattice" (K 10 → 3 by batch size) | peak 237.67 aggregate @ c16 (coding) | "AEON Bench" wave-peak; harness not published |

*Not on this table: H200 results (z-lab / incoai: C1 184–236, C32 1.5–2K tok/s) and RTX 5090 results (maurienne-ai 228 e2e, YourHighnessLA 616 aggregate @ c4) — the 5090's 1.8 TB/s against the GB10's 273 GB/s makes them a different machine, and several of those authors say so themselves.*

**Reading the table — where Kearuga stands**
1. **Target quantization is the largest single lever on this bandwidth-bound box, and ours is a deliberate trade.** Every other SGLang row serves a *uniform* NVFP4 target (≈ 16 GB of weights read per decode cycle). Kearuga reads 24.85 GB — `lm_head` and embeddings in BF16, attention and GDN projections in FP8 — for Fidelity-40 KL 0.0165 (our earlier all-NVFP4 build measured 0.0334). That is roughly a quarter to a third more bytes per cycle at equal acceptance; it accounts for most of the gap between our 45 and the 50–65 of the fastest uniform-NVFP4 rows (the GPU clock does not: locking it at 2400 MHz or leaving it free changed nothing here).
2. **On the drafter side we are at the front of the pack.** The two fastest community rows (hasso5703, sxuff) got there the way we did — a calibrated NVFP4 DFlash 2 drafter at depth 16 (+25 % whole-request for sxuff). Ours is Kearuga-distilled, Kearuga-calibrated, K=12 with a 64K draft head, and measures +14.5 % C1 / +10.8 % C4 over the stock drafter paired on one clock.
3. **Prose is everyone's floor:** 18–25 tok/s on every GB10 row that reports it (ours 21.6); block drafters accept ≈ 2.4 tokens per cycle on free prose against 7–10 on code and tool calls.
4. **Concurrency:** our C4 aggregate of 131 sits in the GB10 SGLang band (MiaAI 111.6, Weschera 64–66; 0xBakeer's vLLM 168). We admit 4 seats by design (see the memory math) and do not publish c8+.
5. **Reproducibility:** identical launches on this image can serve ~20 % apart (Weschera's boot lottery; hasso5703's deterministic-kernels note). Our launcher pins `--disable-flashinfer-autotune`, and every Kearuga number above was measured with it. Run-to-run spread on this probe is real, though: the two C2 repetitions of the unlocked run came out 78.1 and 93.6 (each repetition uses a different request marker, so the forced 512-token tail differs).

### ⏱️ 3. Saturated Responsiveness & Priority Scheduling

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

### ⚖️ 4. Checkpoint Fidelity (Kearuga vs. Original Base BF16)

| Metric | Base Model: `Qwen/Qwen3.8-27B` (Native BF16) | **Qwen3.8-27B-Kearuga** (This Suite) | Delta / Rationale |
|---|:---:|:---:|:---|
| **Weight Footprint** | 51.8 GiB | **24.85 GB** | **−52.0%** footprint |
| **Fidelity-40 Mean KL** | 0.0000 | **0.0165** | Near-zero distribution divergence |
| **Fidelity-40 Mean JS Divergence** | 0.0000 | **0.0034** | High distributional stability |
| **Fidelity-40 Top-1 Agreement** | 40 / 40 | **40 / 40** | 100% exact argmax token match |
| **Fidelity-40 Exact 32-Token Match** | 40 / 40 | **20 / 40** | 50% byte-identical; ties flip to valid equivalents |
| **Held-Out Full-Vocab KL (72k pos)** | 0.0000 | **0.0208** | Evaluated on out-of-domain sequences |
| **Held-Out Top-1 Agreement** | 100.0% | **95.0%** | −5.0 pt drop over raw vocabulary |

---

## 🎛️ Runtime Envelope & Memory Math

> *"Measured, not assumed: the allocations SGLang logged at the production boot (2026-09-17, --mem-fraction-static 0.85, BF16 KV)."*

| Allocation (Kearuga profile, `--mem-fraction-static 0.85`, BF16 KV) | Measured | Where it comes from |
|---|:---:|---|
| **Target weights** | 24.9 GB | Hybrid GPTQ-4o6 + FP8 + NVFP4 (`Load weight end … mem usage=24.87 GB`) |
| **Drafter weights** | 1.4 GB (+ 0.7 GB pruned draft head) | Kearuga NVFP4 DFlash 2; the 65,650-row BF16 head is built at CUDA-graph capture |
| **GDN / Mamba state pool** | 5.8 GB | 20 slots (4 requests × 5), `--mamba-full-memory-ratio 4.21`, `extra_buffer` radix strategy |
| **Target KV cache (BF16)** | 49.9 GB | **818,294 tokens** × ~60 KiB — sized by the memory fraction; the `--max-total-tokens 1048576` cap is not reached |
| **Drafter KV cache (BF16)** | 15.6 GB | the same 818,294 tokens for the 5-layer drafter (~19 KiB/token) |
| **CUDA graphs + workspace** | ~2.0 GB | target and draft verify graphs, batch sizes 1–4 |
| **Total allocated by the server** | **~99.7 GB** | of 121 GB usable unified memory; `available_gpu_mem` after boot 15.5 GB, ~11 GB left to the OS (`free -g`) |

*Stock profile (`DRAFTER_PROFILE=stock`, same fraction): BF16 drafter 3.0 GB; KV pool 802,746 tokens (49.0 GB target + 15.3 GB drafter); `available_gpu_mem` 14.7 GB.*

| Architectural Dimension | Specification | Operational Details |
|---|---:|---|
| 🧠 **Per-Request Context** | `262,144` tokens | Native Qwen3.8 window (YaRN interpolation disabled) |
| 📜 **Shared KV Pool** | `818,294` tokens (measured) | Shared by the 4 seats: ≈ 3.1 full 262K contexts at once, or 4 requests of ≈ 200K each. `--max-total-tokens 1048576` is a ceiling, not the allocation |
| 👥 **Admitted Concurrency** | `4` concurrent streams | Sets the decode CUDA-graph batch sizes (1–4) and the 20-slot GDN state pool |
| 💾 **KV Precision** | BF16 | Fidelity-first: FP8 KV measured within noise on speed, BF16 kept for fidelity |
| ⚡ **Target Weights** | `24.85 GB` | Hybrid GPTQ-4o6 + FP8 + NVFP4 (3 shards + MTP head) |
| 🏎️ **Drafter** | `1.55 GB` + 0.67 GB head | Kearuga NVFP4 DFlash 2 (stock BF16 drafter: 3.58 GiB) |
| 🛡️ **Static allocation** | **~100 GB** | of 121 GB unified memory at `--mem-fraction-static 0.85` |

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
python3 bench/scale.py           # C1/C2/C4 aggregate + TTFT probe (the harness behind our comparison rows)
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
| **Kearuga DFlash 2 Drafter v1.0** | [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) | Kearuga-distilled, Kearuga-calibrated NVFP4 drafter (1.55 GB) + 64K draft-head map and SGLang overlay — release v1.0, pinned by the launcher at files revision `4a109695` |
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
* **[hasso5703](https://github.com/hasso5703/dgx-spark-qwen38)**, **[anliang0306](https://github.com/anliang0306/dgx-spark-qwen38)**, **[0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)** — the newest GB10 recipes and measurements we compare against (see §2).
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

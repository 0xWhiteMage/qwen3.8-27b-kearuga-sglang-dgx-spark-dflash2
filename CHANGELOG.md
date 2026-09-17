# 📜 Changelog

All notable changes to the Kearuga model suite and DGX Spark deployment stack are documented in this file.

---

## [v0.6.4] - 2026-09-17

### 🏷️ Version coherence across GitHub and Hugging Face
* The drafter is now named **v1.0** everywhere (HF drafter card badge + Release row, this README, `summary.md`, `drafter/README.md`); the launcher's `DFLASH_REV` default pins the v1.0 card commit `ae61d9b8` (weights `9a39463f…` and token map `cadaee01…` unchanged since the release). The HF target card names this serving stack **v0.6.4** and the drafter **v1.0**.
* Version scheme: the serving stack (this repo) carries `v0.x.y`; the drafter carries `v1.x` for releases of the same architecture and `v2` for a retrained drafter; the target checkpoint is unversioned (`Qwen3.8-27B-Kearuga`).

---

## [v0.6.3] - 2026-09-17

### 🔎 Docs — clock-lock check, historical row relabelled
* **The GPU clock is not a factor.** `bench/scale.py` re-run with the owner's 2400 MHz SM lock lifted: C1 44.5 / C2 85.9 (78.1–93.6) / C4 131.1 tok/s vs 44.7 / 77.9 / 131.3 locked. Unlocked, the GB10 ran this decode workload at 2.40–2.48 GHz on its own (nominal max 3003 MHz; no thermal or power throttle flags during the run) — the decode loop is memory-bandwidth-bound. v0.6.2 attributed the difference from the v0.5.0 figures to the lock; that attribution is withdrawn.
* **v0.5.0 row marked historical.** The 57 / 51 / 94 figures cannot be reproduced (raw run not kept; today's same-script number on the current, paired-faster drafter is 44.7). They stay in the table as documentation of that release, not as a current measurement.

---

## [v0.6.2] - 2026-09-17

### 📊 Docs — community comparison refreshed, measurements corrected
* **Community comparison rebuilt**: 17 GB10 / Qwen3.8-27B sources re-read on 2026-09-17 with a per-row metric column; non-GB10 results (H200, RTX 5090) excluded from the table; new GB10 recipes added (hasso5703, anliang0306, 0xBakeer, tcclaviger et al., Mia-AiLab, AEON-7).
* **Our row re-measured** with `bench/scale.py` on the Kearuga profile: C1 44.7 / C2 77.9 / C4 131.3 tok/s aggregate under the owner's 2400 MHz SM clock cap — the v0.5.0 figures (57 / 51 / 94) predate the cap.
* **Memory math corrected to boot-log measurements**: KV pool 818,294 tokens (49.9 GB target + 15.6 GB drafter KV), ~100 GB static at mem-fraction 0.85 — the previous 1,048,576-token / 32 GiB / ~63 GiB figures were carried over from an earlier configuration and were wrong.
* **"Reasoning enabled" header corrected** (`bench/scale.py` measures thinking-off) and the unmeasured "no-spec" speed-up rows removed from the fidelity table.
* **INSIGHTS.md** executive summary, §1 and §5 refreshed with the same measured figures; `summary.md` and `.env.sample` aligned.

---

## [v0.6.1] - 2026-09-17

### 🩹 Fix — fresh clones failed the launch-time hash check
* `drafter/sglang-overlay/MANIFEST.json` and `PATCH.diff` were pinned as CRLF bytes but stored LF by git's eol normalization, so every Linux clone failed `sha256sum -c drafter/SHA256SUMS` and the kearuga profile refused to boot. Both files are now LF at the source and on Hugging Face (drafter commit `6eaa58f3`); `drafter/**` is stored verbatim (`-text`); `drafter/SHA256SUMS` re-pinned; `DFLASH_REV` default → `6eaa58f3` (same weights and token map — only the overlay text files changed). Caught by the end-to-end boot test of the published launcher, before it reached the resident.
* `PATCH.diff` usage documented: apply with `git apply -p1` in `sglang/python` and copy the new `dflash_head_utils.py` alongside (it is not part of the diff).

---

## [v0.6.0] - 2026-09-17

### ⚡ Kearuga DFlash 2 Drafter — Default Profile
* **Released [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)** (`39a26bd8`): distilled on 24,644 Kearuga-answered conversations, quantized to ModelOpt NVFP4 with activation scales calibrated on Kearuga's own features (1.55 GB vs 3.85 GB stock BF16), and served through a frequency-pruned 65,650-row draft head (`draft-vocab-v4-top65650.pt`, sha256 `cadaee01…`).
* **Paired bake-off vs the stock BF16 K=10 drafter** (same image, target, flags, clock; 50-prompt battery × 2 runs, temperature 0): **+14.5 % C1 / +10.8 % C4 aggregate** with the 64K draft head, better on all 10 domain cells; plain SGLang (no overlay) is +7.9 % / +4.3 %.
* **Quality & fidelity unchanged**: Quality-200 155/180 (near-tie band 154–159; target alone 157); Fidelity-40 mean KL 0.0165, top-1 40/40 — the target verifies every drafted token.

### 🧩 Launcher & Integrity
* **`DRAFTER_PROFILE=kearuga|stock`** in `start-dflash2.sh` — `kearuga` is the new default (K=12, `modelopt_fp4`, overlay + token map); `stock` reproduces the exact v0.5.0 recipe (z-lab BF16, K=10, no overlay).
* **Vendored `drafter/` directory**: the 5-file read-only SGLang overlay, the 64K token map (`.pt` + `.json`), `SHA256SUMS` (vendored bytes) and `BASE-SHA256SUMS` (in-image base hashes). Launch verifies both before mounting; `OVERLAY_SKIP_BASE_CHECK=1` opts out of the image check, `DRY_RUN=1` prints the fully expanded `docker run` command after all checks and exits.
* **Defaults updated to the validated production profile**: `MEM_FRACTION=0.85` (launcher and `.env.sample`), `DFLASH_DRAFT_TOKENS=12` on the kearuga profile.
* **Launcher flags aligned with the measured production configuration**: `--disable-flashinfer-autotune` is now always passed (every published number was measured with it), and `--revision local` is added automatically when `TARGET_MODEL` is a local directory.
* **`bench/verify_all.py`**: new gates for the Kearuga drafter checkpoint (cached-or-hosted) and `drafter/SHA256SUMS` byte verification (13 gates total); stock drafter check retained as the fallback profile.

### 📚 Documentation
* README paired-benchmark section, memory-math update (drafter ~2.1 GiB computed, total ~62.9 GiB), profile-switching quick start, credits; INSIGHTS §3/§4 rewritten for the NVFP4 drafter + draft head (fused KV materialization is disabled on this profile — it requires a BF16 `qkv_proj`).

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

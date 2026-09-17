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
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) (Kearuga Drafter, revision `6eaa58f3…`)
  * **Precision**: NVFP4 W4A4 (ModelOpt layout, Kearuga-calibrated activation scales), 1.55 GB; selector / fc / conv projections stay BF16.
  * **Draft head**: frequency-pruned 64K head (65,650 rows, 0.63 GiB BF16) via the vendored 5-file read-only overlay (`drafter/`), hash-verified at launch; K=12.
  * **SGLang Kernel**: fused KV materialization is **disabled** on this profile (it requires a BF16 `qkv_proj`); the win is weight bandwidth (1.55 GB vs 3.85 GB read per cycle). On `DRAFTER_PROFILE=stock` it remains active.
  * **Status**: +14.5 % C1 / +10.8 % C4 aggregate vs the stock drafter in a paired bake-off (all 10 domain cells better). Fallback: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (stock BF16, 3.58 GiB, K=10, fused KV materialization active) — v0.5.0 single-prompt probe: 57 tok/s C1 · 264ms TTFT, 51 tok/s C2 agg, 94 tok/s C4 agg · 480ms TTFT.
  * **Context**: 262,144 tokens.

---

## 🎛️ Hardware & Execution Profile

* **Hardware Platform**: NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory).
* **Serving Image**: `lmsysorg/sglang@sha256:616a3e97…` (official, digest-pinned; the kearuga profile mounts a 5-file read-only Python overlay, hash-verified at launch — no image rebuild).
* **Serving VRAM Envelope**: 24.85 GiB (Target) + ~2.1 GiB (Kearuga drafter + 64K head, computed from file sizes) + 32.00 GiB (1M-Token BF16 KV Pool) + ~4.0 GiB (PyTorch / CUDA runtime) = ~62.9 GiB total (>65 GiB headroom). Stock profile: 3.58 GiB drafter → ~64.4 GiB.
* **Launch Profile**:
  * `start-dflash2.sh` (`DRAFTER_PROFILE=kearuga`, default): paired-battery aggregate 35.33 tok/s C1 / 108.90 tok/s C4 (+14.5 % / +10.8 % vs the stock drafter); stock-drafter v0.5.0 single-prompt probe: 57 tok/s C1 · 264ms TTFT, 51 tok/s C2 agg, 94 tok/s C4 agg · 480ms TTFT. `DRAFTER_PROFILE=stock` reproduces the v0.5.0 recipe.

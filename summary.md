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
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) (Kearuga Drafter **v1.0**, files revision `4a109695…`)
  * **Precision**: NVFP4 W4A4 (ModelOpt layout, Kearuga-calibrated activation scales), 1.55 GB; selector / fc / conv projections stay BF16.
  * **Draft head**: frequency-pruned 64K head (65,650 rows, 0.63 GiB BF16) via the vendored 5-file read-only overlay (`drafter/`), hash-verified at launch; K=12.
  * **SGLang Kernel**: fused KV materialization is **disabled** on this profile (it requires a BF16 `qkv_proj`); the win is weight bandwidth (1.55 GB vs 3.85 GB read per cycle). On `DRAFTER_PROFILE=stock` it remains active.
  * **Status**: +14.5 % C1 / +10.8 % C4 aggregate vs the stock drafter in a paired bake-off (all 10 domain cells better). Fallback: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (stock BF16, 3.58 GiB, K=10, fused KV materialization active) — v0.5.0 single-prompt probe: 57 tok/s C1 · 264ms TTFT, 51 tok/s C2 agg, 94 tok/s C4 agg · 480ms TTFT.
  * **Context**: 262,144 tokens.

---

## 🎛️ Hardware & Execution Profile

* **Hardware Platform**: NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory).
* **Serving Image**: `lmsysorg/sglang@sha256:616a3e97…` (official, digest-pinned; the kearuga profile mounts a 5-file read-only Python overlay, hash-verified at launch — no image rebuild).
* **Serving VRAM Envelope**: 24.9 GB (target) + 1.4 GB (NVFP4 drafter) + 0.7 GB (draft head) + 5.8 GB (GDN state pool) + 49.9 GB (target KV, 818,294 tokens BF16) + 15.6 GB (drafter KV) + ~2.0 GB (CUDA graphs) ≈ 99.7 GB of 121 GB unified memory at --mem-fraction-static 0.85 (available_gpu_mem after boot 15.5 GB). Stock profile: 802,746-token pool, 3.0 GB drafter.
* **Launch Profile**:
  * `start-dflash2.sh` (`DRAFTER_PROFILE=kearuga`, default): paired-battery aggregate 35.33 tok/s C1 / 108.90 tok/s C4 (+14.5 % / +10.8 % vs the stock drafter); scale.py C1 44.7 / C2 77.9 / C4 131.3 tok/s aggregate (thinking off; identical with the GPU clock lock lifted); the v0.5.0 stock-drafter figures 57 / 51 / 94 are historical and were not reproduced. `DRAFTER_PROFILE=stock` reproduces the v0.5.0 recipe.

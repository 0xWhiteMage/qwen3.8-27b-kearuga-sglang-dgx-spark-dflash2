# 🧙‍♂️ Kearuga: Architecture Insights & Design Rationale

> A deep dive into the engineering choices, quantization hierarchy, and speculative inference techniques powering Qwen3.8-27B on the NVIDIA DGX Spark.

---

## 🎯 Executive Summary: What Kearuga Solves

Serving a 27-billion parameter dense model like **Qwen3.8-27B** locally on a single machine requires balancing interactive latency, multi-request capacity, and output fidelity without GPU memory exhaustion.

Kearuga achieves this on a **single 128 GB NVIDIA DGX Spark (GB10 / SM121)** by combining **EXL3-inspired tiered sensitivity quantization** with **DFlash 2 block-diffusion speculative decoding**:

| Benchmark / Capability | Kearuga Profile | Measured Performance | Operational Significance |
|---|---|---:|---|
| ⚡ **Single-Stream (C1)** | Kearuga NVFP4 drafter **v1.0**, K=12 + 64K head | **44.7 tok/s** agg (45.5 net) on the forced-512 code probe; battery medians code 57.6 · tool calls 114.6 · prose 21.6 | Interactive daily driver |
| 👥 **Dual-Stream (C2)** | same | **78–94 tok/s agg** across runs (77.9 and 85.9 mean; ≈ 40–48/stream, TTFT 0.28 s) | Two interactive sessions |
| 👷 **Saturated Interactive (C4)** | same | **131.3 tok/s agg** (≈ 36/stream, TTFT 0.34 s) | Four seats busy |
| 📜 **Shared KV Pool** | BF16 target KV + FP8 drafter KV (v0.6.6) | **932,355 tokens** (measured at the production boot; 818,294 with BF16 drafter KV) | 4 seats × 262K window; ≈ 3.6 full contexts at once |
| ⏱️ **Saturated Priority TTFT** | **Preemption Mode** | **43.15s → 2.63s** | **93.9% latency reduction** under full load |

*Measured 2026-09-17 with `bench/scale.py` (T=0, thinking off, 512 forced tokens, aggregate incl. TTFT); repeated with the GPU clock lock lifted — identical (C1 44.5 / C2 85.9 / C4 131.1; the GPU ran at 2.40–2.48 GHz by itself). The v0.5.0 figures (57 / 51 / 94 with the stock drafter) are historical and were not reproduced. Paired drafter comparison: [README §1](README.md).*

---

## 🏗️ 1. Why Block-Diffusion Speculative Decoding (DFlash 2) Outperforms Sequential Drafters

> *"Sequential speculative drafters saturate memory bandwidth. Block-diffusion predicts candidate blocks in a single forward pass, unlocking instant interactive responsiveness."*

| Architectural Dimension | Traditional Sequential Drafters | ⚡ DFlash 2 Block-Diffusion Profile |
|---|---|---|
| **Draft Prediction Complexity** | Sequential $O(K)$ autoregressive forward passes | Parallel single-step $O(1)$ block diffusion |
| **Memory Bus Overhead** | $K$ sequential memory round-trips per step | Single memory fetch per candidate block |
| **Drafter Footprint** | Often multi-billion parameter autoregressive model | 1.55 GB NVFP4 Kearuga drafter (+ 0.67 GB pruned draft head); stock BF16 3.58 GiB |
| **Kernel Materialization** | Separate draft KV cache allocations | Fused KV projection kernel with the stock BF16 drafter; unfused with the NVFP4 drafter (§3) |
| **Empirical Throughput** | High per-step latency overhead | **44.7 / 78–94 / 131 tok/s agg at C1 / C2 / C4** (clock lock on or off: same) |

### ⚡ DFlash 2: The Interactive Engine (C1–C4)
* **How It Works**: Traditional speculative drafters draft tokens sequentially (generating one candidate token at a time). DFlash 2 uses a non-causal **block-diffusion architecture** that predicts candidate token blocks (block size K=12 with the Kearuga drafter, K=10 with the stock drafter) in a single forward pass (single-step O(1)).
* **The Benefit**: Eliminates sequential draft latency entirely, unlocking steady-state interactive decode speeds of **44.7 tok/s C1**, **78–94 tok/s aggregate C2**, and **131 tok/s aggregate C4** on DGX Spark unified memory (identical with the GPU clock lock on or off — the decode loop is memory-bandwidth-bound).
* **Unified Memory Optimization**: Because Grace-Blackwell GB10 utilizes unified high-bandwidth memory, eliminating sequential kernel launches and memory ping-pong is paramount. DFlash 2 reduces GPU memory bus traffic by amortizing draft overhead into a single parallel tensor operation.

---

## 🔬 2. Why Our Quantization Strategy Is Superior

> *"Uniform quantization compromises model reasoning. Tiered sensitivity quantization preserves intelligence while maximizing hardware speed."*

### ❌ The Limitation of Uniform Quantization
Applying a single quantization format across all layers (e.g. uniform INT4 or uniform NVFP4) introduces severe, uneven numerical degradation across sensitive architectural components:
1. **Vocabulary Logit Fidelity**: Quantizing `embed_tokens` and `lm_head` causes loss of precision on code syntax and rare vocabulary tokens.
2. **Recurrent State Sensitivity**: Gated DeltaNet linear attention projections (`in_proj`, `conv1d`) degrade over long context windows (>64K tokens) if quantized too aggressively.
3. **Speculative Feature Quality**: Intermediate tapped layers (`[5, 19, 33, 47, 61]`) require clean, low-noise representations to maximize candidate block acceptance.

### ✅ The Kearuga Solution: EXL3-Inspired Tiered Sensitivity Hierarchy

Applying sensitivity lessons from mixed-precision research ([`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)), **[`Qwen3.8-27B-Kearuga`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga)** splits model weights into four distinct precision tiers:

| Tier | Layers & Tensors | Precision | Architectural Purpose |
|---|---|:---:|---|
| **Tier 1 (Protect)** | `embed_tokens`, `lm_head`, all norms, 27 Vision Blocks (333 tensors), MTP draft head (15 tensors) | Native BF16 | Protects vocabulary logit tails, multimodal reasoning, and MTP speculative decoding |
| **Tier 2 (Medium)** | Attention Projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), GDN Recurrence (`in_proj`), Boundary MLPs (Layers 0, 1, 62, 63) | FP8 E4M3 | Preserves draft feature taps `[5, 19, 33, 47, 61]` and recurrence stability |
| **Tier 3 (Core)** | MLP `gate_proj` + `up_proj` (Layers 2–61, 120 tensors) | GPTQ-4o6 (W4A16 NVFP4) | Four-Over-Six group scales remove 21.7% of weight KL divergence |
| **Tier 4 (Down)** | MLP `down_proj` (Layers 2–61, 60 tensors) | NVFP4 AWQ (Pre-quantized) | ModelOpt AWQ export, retained to handle activation outliers |


* **Outcome**: A compact **24.85 GB** model running with full Blackwell Tensor Core acceleration while preserving **40/40 top-1 token agreement** with the BF16 base and passing **157/180 Quality-200 objective gates** (GSM8K, HumanEval, IFEval, agentic coding; 155/180 served with the Kearuga drafter — near-tie band).

#### Four-Over-Six (4o6) Group Scales
Standard GPTQ uses a single group scale per block (amax → code 6). Four-Over-Six instead evaluates dynamic range per block and chooses the better of:
- **amax → 6** (standard: larger dynamic range, slightly lower precision)
- **amax → 4** (alternative: higher precision for blocks with tight local distribution)

In our calibration, **44.7% of blocks chose Code 4**. This reduces Hessian-weighted MSE by 16% (0.529 → 0.445) and cuts held-out KL divergence by 21.7% at identical bytes/step and identical serving format.

#### Fused-Shared Global Scale (Serving Contract)
SGLang fuses `gate_proj` + `up_proj` into a single linear kernel and uses `weight_scale_2.max()` for the pair. Kearuga writes identical `weight_scale_2` for gate and up of every layer by computing a shared global scale:

```text
scale_shared = max(amax(gate), amax(up))
```

Without this synchronization, one half of every fused MLP would be dequantized with a ~1.9× wrong scale, corrupting generation.

---

## 🏎️ 3. Drafter Architecture: SGLang Fused KV Materialization

> *"Drafter precision must preserve SGLang's fused CUDA graph materialization while minimizing memory bus traffic."*

### 3.1 Fused KV Materialization Contract
In SGLang's DFlash engine, the draft model projects target hidden states into the draft KV cache using a specialized CUDA kernel (`fused_dflash_kv_kernel`).
* SGLang's high-speed kernel requires `self_attn.qkv_proj` in native **BF16**.
* The stock [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter keeps `qkv_proj`/`out_proj` in BF16 and gets the fused kernel — still active on `DRAFTER_PROFILE=stock`.
* **With the NVFP4 Kearuga drafter the kernel is disabled**: its `qkv_proj` is quantized (`quant_method=ModelOptFp4LinearMethod`), so the boot log prints `DFLASH fused KV materialization disabled …`. The drafter still wins because the decode cycle on GB10 is weight-bandwidth-bound — 1.55 GB NVFP4 read vs 3.85 GB BF16 — which more than pays for the unfused projection.

---

## 🏎️ 4. Drafter: Kearuga's Own DFlash 2 Drafter (v1.0)

> *"The stock drafter delivers instant speedup out of the box; the Kearuga drafter is distilled on Kearuga's own outputs and is 2.5× smaller."*

The released drafter [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) **v1.0** (files revision `4a109695`, pinned by this repo's launcher) was built in three stages (full recipe on its model card):

1. **Distillation on Kearuga's own outputs** — 24,644 conversations answered greedily by the Kearuga target over 8 domains, teacher-forced block-diffusion CE, initialized from `z-lab/Qwen3.8-27B-DFlash2` (architecture unchanged).
2. **NVFP4 weights + Kearuga-calibrated activation scales** — ModelOpt NVFP4 (E2M1, group 16) on 35 linears; input scales calibrated on Kearuga features over 100 conversations (a stratified 8-domain, 50 % thinking-on subsample of the 400-conversation Kearuga teacher set); selector, `fc` and conv projections stay BF16. Loads as `--speculative-draft-model-quantization modelopt_fp4`.
3. **The 64K draft head (FR-Spec-style)** — Kearuga's actual output tokens were counted (14.74 M over 20,000 responses): the top 65,536 ids cover 99.56 %, plus 114 `=identifier` fusion tokens = **65,650 rows**. An even row count matters for the head GEMM (an odd count pushed cuBLAS onto a 3×-slower unaligned kernel). The draft scores candidates through those rows; the target verifies over the full vocabulary — lossless by construction.

**What did not work**: K = 14/16 (C1 tie, C4 −2 to −6 %), a 32K map (acceptance loss eats the bytes), an FP8 draft head (−18 %), draft window 1024/4096 and FP8 draft KV (±0.2 %).

**Where the remaining headroom is** (exact acceptance on the production resident, 2026-09-17, n = 2 prompts per domain): greedy thinking-off code 7.5 · math 7.7 · tool calls 10.5 · prose 2.5 · IFEval 2.4 tokens per cycle; the model's default sampling (T 1.0 · top-p 0.95 · top-k 20) costs only 0.1–1.1 tokens per cycle; **thinking-on code falls to 3.1** because the reasoning trace behaves like prose, so a thinking-on code request decodes at roughly prose speed (≈ 27 tok/s incl. TTFT vs 56–65 thinking-off). Block drafters share this prose floor across the GB10 community (18–25 tok/s); raising acceptance on reasoning text — not more bytes saved — is the next drafter lever. Two follow-up measurements (2026-09-22) sharpen this. *Paired, thinking on:* the v1.0 drafter is still 7–10 % faster than the stock drafter at C1 (32.90 vs 30.61 / 29.61 tok/s at stock K 10 / 12) and 8 % at C4, in all five domains — the byte advantage carries into reasoning traces, but acceptance there is 3.15 (code) · 2.50 (prose) · 5.68 (math) · 7.14 (tool) tokens per cycle at K = 12, about half of answer-text acceptance, and the stock drafter accepts up to 0.5 more per cycle on four of the five domains at a higher per-cycle cost. *Reasoning effort:* the template default `xhigh` is the expensive setting — on the live resident both code prompts used the whole 4,096-token budget without answering, while `medium` answered in ~90 s and cut math wall time 2.6× with higher acceptance (6.27 → 7.56). The per-token speed of reasoning text is the drafter's problem; the number of reasoning tokens is a request setting (`chat_template_kwargs.reasoning_effort`), and the latter is the larger lever today. The community evidence on how to move the former points at *trained* block expansion with reasoning-trace data (a head trained for longer blocks on thinking-mode completions improves the late positions; naive widening of K at inference does not — our own K = 14 sweep and independent GB10 depth sweeps agree), which is where our drafter v2 work goes. Two byte-side ideas were measured and closed: quantizing the drafter's last BF16 projections (+0.6 % C1 / −1.1 % C4, below the gate) and FP8 draft KV (a capacity knob, ≈ −1 % speed).

**What the drafter buys, measured against no speculation (2026-09-23).** The same target with every `--speculative-*` flag removed decodes at a flat ≈ 10.3 tok/s per stream (C1 aggregate 10.30, C4 39.56 on the 50-prompt battery) — the weight-streaming bound: 24.85 GB of target weights per token over the GB10's 273 GB/s is ≈ 11 tok/s, and the domain-independence of that number is the signature of a bandwidth-bound loop. The production configuration (drafter v1.0, K = 12, 64K head, FP8 draft KV) measures 35.66 / 108.83 on the same line: 3.46× at C1, 2.75× at C4, and per domain 5.6× (code), 6.0× (math), 8.5× (tool calls), 2.1× (prose, instruction-following). The C4 ratio is lower than C1's because the no-speculation loop batches almost perfectly (3.84× from C1 to C4) while the speculative loop's verify blocks already fill the memory pipe. Speculation also costs capacity: without the drafter the KV pool hits the 4 × 262,144-token cap with 21 GB unallocated; with it, 932,355 tokens.

**Lossless up to floating-point numerics — and why the offline reference loop is not.** Greedy verification reproduces the target's argmax in exact arithmetic; in bf16 it reproduces it wherever the top-2 margin exceeds rounding. Measured on 40 prompts × 256 greedy tokens: speculative vs no-speculation continuations diverge on 28/40 prompts, every first divergence at a top-2 logprob margin ≤ 0.25 (0, 0.125 or 0.25 — one or two bf16 ulps of the logits); two runs of the speculative server diverge on 6/40 with the same margins. Verify blocks drive the target through different GEMM shapes than one-token decode, so more near-ties flip than between two same-shape runs — but none flips at a confident position. Contrast the HF/z-lab reference loop (`dflash_generate` on transformers): `Cache.crop` trims the Gated-DeltaNet conv state but leaves the recurrent state as written after the whole verify block, so rejected drafts stay in the target's memory. On this hybrid that is a real distribution shift (first-divergence margins 0.6–2.4 in our teacher-forced check) and, because the target's memory then contains the drafter's own rejected continuation, it *inflates* measured acceptance — 1.9× over 100 prompts (8,465 cycles) in our diagnostic before we added a snapshot-and-restore. Anyone evaluating a drafter for a Qwen3.5-class target with the reference loop should expect optimistic numbers until huggingface/transformers#49036 lands; SGLang's own verify path does not have this problem, which is what the 0.25-margin result shows.

---

## 💾 5. Hardware Memory Math: Serving Envelope

> *"The pool is sized by the memory fraction, not by the cap: 932,355 KV tokens at --mem-fraction-static 0.85 with FP8 drafter KV (818,294 with BF16 drafter KV)."*

### Serving on a Single 128 GB DGX Spark (Measured at the Production Boot)
* **Target weights**: 24.9 GB (hybrid GPTQ-4o6 + FP8 + NVFP4; boot log `mem usage=24.87 GB`)
* **Kearuga NVFP4 DFlash 2 drafter**: 1.4 GB weights + 0.7 GB pruned 65,650-row BF16 draft head (built at CUDA-graph capture)
* **GDN / Mamba state pool**: 5.8 GB (20 slots = 4 requests × 5)
* **Target KV cache (BF16)**: 56.9 GB = **932,355 tokens** (~60 KiB/token)
* **Drafter KV cache (FP8 E4M3, default since v0.6.6)**: 8.9 GB for the same 932,355 tokens (~10 KiB/token). With BF16 drafter KV the pool was 818,294 tokens (49.9 GB target + 15.6 GB drafter); the FP8 drafter KV costs ≈ −1 % decode (paired sweep) and buys +13.9 % pool.
* **CUDA graphs + workspace**: ~2.0 GB
* **Total allocated by the server**: **~100.6 GB of 121 GB usable unified memory** at `--mem-fraction-static 0.85`; `available_gpu_mem` after boot 15.7 GB, ~14 GB left to the OS. Stock profile: 3.0 GB BF16 drafter, 802,746-token KV pool (49.0 + 15.3 GB), `available_gpu_mem` 14.7 GB.

---

## ⏱️ 6. Saturated Responsiveness & Priority Scheduling

> *"In real-world multi-agent deployments, priority preemption is the difference between an instant response and a 40-second freeze."*

| Load Scenario | Default Priority TTFT | Interactive Priority (`priority: 100`) | Latency Improvement |
|---|---:|---:|---:|
| **DFlash 2 (All 4 Seats Full)** | ~43.15 s | **~2.63 s** | **93.9% faster** |

Passing `"priority": 100` in the OpenAI-compatible API request preempts background agent batches, delivering sub-3-second responses even when the GPU is 100% saturated.

---

## 🤝 7. Acknowledgements & Community Credits

We gratefully acknowledge the researchers, engineers, and creators whose open-source repositories and insights made this project possible:

* 🔬 **[malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3)**: For the groundbreaking mixed-precision sensitivity research that inspired our Tiered Sensitivity Map.
* 🚀 **[MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)**: For pioneering SGLang DGX Spark deployment recipes, empirical DSpark / DFlash benchmarks, and CPU core affinity optimizations.
* 📦 **[Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark)**: For pioneering DFlash 2 block size parameter sweeps (Block 10 Speed Profile @ 42.04 tok/s vs. Block 8 Capacity Profile @ 120.58 tok/s C8) and analyzing SGLang AutoTuner kernel selection.
* ⚙️ **[r0b0tlab/qwen38-27b-nvfp4-sm121-sglang](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang)**: For SM121 hardware image pinning, CPU core affinity contracts, and system stability flags.
* 📊 **[0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)**: For vLLM 4-bit memory allocation analysis and throughput benchmarks.
* ⚡ **[z-lab/dflash](https://github.com/z-lab/dflash)**: For inventing the revolutionary block-diffusion speculative decoding architecture.
* 🌐 **[SGLang Project](https://github.com/sgl-project/sglang)**: For the high-throughput inference engine, radix attention, and speculative decoding framework.

The drafter-specific credits (base weights, quantization yardsticks, training recipes, papers) are listed in full in [README §Credits](README.md#-credits) and on the [drafter model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2).

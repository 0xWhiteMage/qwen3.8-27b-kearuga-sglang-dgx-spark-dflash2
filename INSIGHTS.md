# 🧙‍♂️ Kearuga: Architecture Insights & Design Rationale

> A deep dive into the engineering choices, quantization hierarchy, and speculative inference techniques powering Qwen3.8-27B on the NVIDIA DGX Spark.

---

## 🎯 Executive Summary: What Kearuga Solves

Serving a 27-billion parameter dense model like **Qwen3.8-27B** locally on a single machine requires balancing interactive latency, multi-request capacity, and output fidelity without GPU memory exhaustion.

Kearuga achieves this on a **single 128 GB NVIDIA DGX Spark (GB10 / SM121)** by combining **EXL3-inspired tiered sensitivity quantization** with **DFlash 2 block-diffusion speculative decoding**:

| Benchmark / Capability | Kearuga Profile | Measured Performance | Operational Significance |
|---|---|---:|---|
| ⚡ **Single-Stream Net Decode (C1)** | **DFlash 2 (C1)** | **57 tok/s** (57/str, TTFT 264ms) | Zero-latency interactive daily driver |
| 👥 **Dual-Stream Decode (C2)** | **DFlash 2 (C2)** | **51 tok/s agg** (40/str, TTFT 416ms) | Balanced dual-stream interactive sessions |
| 👷 **Saturated Interactive (C4)** | **DFlash 2 (C4)** | **94 tok/s agg** (39/str, TTFT 480ms) | Quad-stream simultaneous interactive tasks |
| 📜 **Shared KV Cache Capacity** | **BF16 KV Pool** | **1,048,576 tokens**| 4 × full 262K native contexts concurrently |
| ⏱️ **Saturated Priority TTFT** | **Preemption Mode** | **43.15s → 2.63s** | **93.9% latency reduction** under full load |

*Throughput figures above are stock-drafter, K=10, v0.5.0 single-prompt probes. Paired-battery aggregate throughput with the Kearuga drafter (K=12 + 64K draft head): 35.33 tok/s C1 / 108.90 tok/s C4 — see [README §0](README.md).*

---

## 🏗️ 1. Why Block-Diffusion Speculative Decoding (DFlash 2) Outperforms Sequential Drafters

> *"Sequential speculative drafters saturate memory bandwidth. Block-diffusion predicts candidate blocks in a single forward pass, unlocking instant interactive responsiveness."*

| Architectural Dimension | Traditional Sequential Drafters | ⚡ DFlash 2 Block-Diffusion Profile |
|---|---|---|
| **Draft Prediction Complexity** | Sequential $O(K)$ autoregressive forward passes | Parallel single-step $O(1)$ block diffusion |
| **Memory Bus Overhead** | $K$ sequential memory round-trips per step | Single memory fetch per candidate block |
| **Drafter Footprint** | Often multi-billion parameter autoregressive model | Compact stock drafter (3.58 GiB native BF16) |
| **Kernel Materialization** | Separate draft KV cache allocations | Fused CUDA graph KV projection (`fused_dflash_kv_kernel`) |
| **Empirical Throughput** | High per-step latency overhead | **57 tok/s C1**, **51 tok/s C2 agg**, **94 tok/s C4 agg** |

### ⚡ DFlash 2: The Interactive Engine (C1–C4)
* **How It Works**: Traditional speculative drafters draft tokens sequentially (generating one candidate token at a time). DFlash 2 uses a non-causal **block-diffusion architecture** that predicts candidate token blocks (block size K=10) in a single forward pass (single-step O(1)).
* **The Benefit**: Eliminates sequential draft latency entirely, unlocking steady-state interactive decode speeds of **57 tok/s C1 (57 tok/s/stream, TTFT 264ms)**, **51 tok/s aggregate C2 (40 tok/s/stream)**, and **94 tok/s aggregate C4 (39 tok/s/stream, TTFT 480ms)** on DGX Spark unified memory.
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


* **Outcome**: A compact **24.85 GB** model running with full Blackwell Tensor Core acceleration while preserving **40/40 top-1 token agreement** with the BF16 base and passing **157/180 Quality-200 objective gates** (GSM8K, HumanEval, IFEval, agentic coding).

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

## 🏎️ 4. Drafter: Kearuga's Own DFlash 2 Drafter

> *"The stock drafter delivers instant speedup out of the box; the Kearuga drafter is distilled on Kearuga's own outputs and is 2.5× smaller."*

The released drafter [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) was built in three stages (full recipe on its model card):

1. **Distillation on Kearuga's own outputs** — 24,644 conversations answered greedily by the Kearuga target over 8 domains, teacher-forced block-diffusion CE, initialized from `z-lab/Qwen3.8-27B-DFlash2` (architecture unchanged).
2. **NVFP4 weights + Kearuga-calibrated activation scales** — ModelOpt NVFP4 (E2M1, group 16) on 35 linears; input scales calibrated on Kearuga features over 400 conversations; selector, `fc` and conv projections stay BF16. Loads as `--speculative-draft-model-quantization modelopt_fp4`.
3. **The 64K draft head (FR-Spec-style)** — Kearuga's actual output tokens were counted (14.74 M over 20,000 responses): the top 65,536 ids cover 99.56 %, plus 114 `=identifier` fusion tokens = **65,650 rows**. An even row count matters for the head GEMM (an odd count pushed cuBLAS onto a 3×-slower unaligned kernel). The draft scores candidates through those rows; the target verifies over the full vocabulary — lossless by construction.

**What did not work**: K = 14/16 (C1 tie, C4 −2 to −6 %), a 32K map (acceptance loss eats the bytes), an FP8 draft head (−18 %), draft window 1024/4096 and FP8 draft KV (±0.2 %).

---

## 💾 5. Hardware Memory Math: Serving Envelope

> *"A single 128 GB DGX Spark serves the 27-billion parameter dense model with abundant headroom."*

### Serving on a Single 128 GB DGX Spark (Comfortable Headroom)
* **Target Model (Hybrid GPTQ-4o6 + FP8 + NVFP4)**: 24.85 GiB
* **Kearuga DFlash 2 Drafter (NVFP4 + 64K BF16 draft head)**: ~2.1 GiB (computed from file sizes: 1.45 GiB weights + 0.63 GiB head)
* **1M-Token KV Cache Pool (BF16, fidelity-first)**: 32.00 GiB
* **SGLang & PyTorch Runtime Overhead**: ~4.00 GiB
* **Total Serving Footprint**: **~62.9 GiB (fits easily within 128 GB Unified Memory with >65 GiB headroom)**. Stock BF16 drafter profile: 3.58 GiB drafter → ~64.4 GiB total.

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

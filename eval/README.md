# Kearuga Evaluation Suite

A frozen, reproducible evaluation suite for comparing quantized Qwen3.8-27B checkpoints against the BF16 reference. Two independent sets are included:

| Set | What it measures | Prompts | Reference |
|---|---|---:|---|
| **KLD-40** (Fidelity-40) | Top-20 token KL divergence, JS divergence, top-1 agreement, exact 32-token continuation match | 40 | Served BF16 capture (2026-08-24) |
| **Quality-200** | Objective task correctness across GSM8K, HumanEval, IFEval, agentic coding, hard reasoning | 200 | Per-family pass/fail grading |

Both sets are **frozen** — the prompt files, tools, and BF16 reference capture are byte-pinned by `frozen/kld-manifest.json` and verified by `frozen/check_kld_manifest.py`. Do not edit the files in `frozen/`.

---

## File layout

```
eval/frozen/
  kld-prompts-40.json              40 prompts for the KLD fidelity set
  kld_capture_score.py             capture + score tool (stdlib only)
  kld-bf16-reference-20260824.json BF16 reference capture (top-20 logprobs + 32-token continuations)
  quality-200-prompts.jsonl        200 prompts across 5 families
  run_quality_200.py               Quality-200 response runner (stdlib only)
  kld-manifest.json                SHA-256 manifest of all frozen files
  check_kld_manifest.py            integrity verifier (exit 1 on any mismatch)
```

---

## KLD-40 (Fidelity-40)

### What it measures

For each of the 40 prompts, the tool sends two requests to an OpenAI-compatible endpoint:

1. **Top-20 logprob capture** — `max_tokens=1`, `logprobs=True`, `top_logprobs=20`.
2. **32-token greedy continuation** — `max_tokens=32`, no logprobs.

Scoring compares the candidate capture against the BF16 reference:

- **Mean KL** — symmetric KL divergence normalized over the union of captured top-20 tokens (floor logprob -20).
- **Mean JS** — Jensen-Shannon divergence over the same union.
- **Top-1 agreement** — fraction of prompts where the highest-logprob token matches the reference.
- **Exact continuation** — fraction of prompts where the 32-token greedy continuation matches byte-for-byte.
- **Probability cosine** — cosine similarity of the top-20 probability distributions.
- **Top-20 overlap** — mean fraction of overlapping tokens in the top-20 sets.

> **Note:** KL/JS are normalized over the union of captured top-20 tokens with a floor logprob of -20. They are approximations of the full-vocabulary divergence, not exact full-vocab KL. For full-vocabulary KL, use the held-out set in the Kearuga source repo.

### Capturing from a served model

```bash
python eval/frozen/kld_capture_score.py capture \
  --prompts eval/frozen/kld-prompts-40.json \
  --base-url http://localhost:8890 \
  --output my-capture.json
```

The endpoint must be an OpenAI-compatible `/v1/chat/completions` server (SGLang, vLLM, etc.) serving a Qwen3.8-27B checkpoint. The capture protocol uses `temperature=0`, `top_p=1`, `seed=1234`, `enable_thinking=False`.

### Scoring against the BF16 reference

```bash
python eval/frozen/kld_capture_score.py score \
  --reference eval/frozen/kld-bf16-reference-20260824.json \
  --candidate my-capture.json \
  --output my-score.json
```

The score JSON contains aggregate metrics and per-prompt rows. The summary is also printed to stdout.

### Using your own BF16 reference

If you want to capture your own BF16 reference instead of using the provided one, run `capture` against your BF16 server and use that as `--reference`. The provided reference was captured on a DGX Spark (GB10) with the official BF16 `Qwen/Qwen3.8-27B` checkpoint on 2026-08-24.

---

## Quality-200

### What it measures

200 prompts across 5 families, graded for objective correctness:

| Family | Count | Max tokens | Grading |
|---|---:|---:|---|
| GSM8K (flexible) | 80 | 1024 | Final-answer numeric match |
| HumanEval | 40 | 2048 | Executable test pass |
| IFEval | 40 | 1024 | Structural instruction compliance |
| Agentic coding | 20 | 2048 | Executable test pass |
| Hard reasoning | 20 | 2048 | Written (not auto-graded) |

### Running the Quality-200 set

```bash
python eval/frozen/run_quality_200.py \
  --prompts eval/frozen/quality-200-prompts.jsonl \
  --base-url http://localhost:8890 \
  --output my-quality-200.jsonl
```

The runner uses `temperature=0`, `top_p=1`, `seed=1234`, `enable_thinking=False`. It records per-prompt responses, finish reasons, and SHA-256 hashes for reproducibility.

> **Grading note:** HumanEval and agentic coding require executable test harnesses. IFEval requires the `ifeval_vendor` package. GSM8K flexible grading uses a final-answer regex. See the Kearuga source repo's `eval/quality/PROVENANCE.md` for third-party dependency setup.

---

## Integrity verification

After cloning or before publishing results, verify that all frozen files are byte-identical to the manifest:

```bash
python eval/frozen/check_kld_manifest.py
```

This exits 0 if all hashes match, 1 on any mismatch. Add `--strict` to also fail if the directory contains unlisted files.

---

## Comparing your model against Kearuga

1. Serve your Qwen3.8-27B checkpoint on an OpenAI-compatible endpoint.
2. Run the KLD-40 capture: `kld_capture_score.py capture --base-url http://your-server:port ...`
3. Score against the provided BF16 reference: `kld_capture_score.py score --reference .../kld-bf16-reference-20260824.json --candidate ...`
4. Run the Quality-200 set: `run_quality_200.py --base-url http://your-server:port ...`
5. Report mean KL, top-1 agreement, exact continuation, and Quality-200 per-family scores alongside your throughput numbers.

### Kearuga reference results

| Metric | Kearuga (promoted) |
|---|---:|
| Served Fidelity-40 mean KL | 0.0165 |
| Served Fidelity-40 top-1 agreement | 40/40 |
| Served Fidelity-40 exact 32-token continuation | 20/40 |
| Held-out full-vocab KL | 0.0208 |
| Held-out top-1 | 95.0% |
| Quality-200 objective | 157/180 |

These were measured on `Qwen3.8-27B-Kearuga` served on a single DGX Spark (GB10) with SGLang, DFlash2 K=10, BF16 KV cache, on the digest-pinned official image. See the [model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) for full details.

---

## License

The evaluation prompts and tools are released under Apache 2.0, consistent with the Kearuga project. The 40 prompts and 200 prompts are original to the Kearuga project. GSM8K and HumanEval prompts are derived from their respective public datasets.

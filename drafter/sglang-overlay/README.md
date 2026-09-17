# serve/overlay-drafthead — DFlash2 draft-only hot-token lm_head pruning + FP8

This overlay lets SGLang's DFLASH speculative worker score draft candidates
through (a) a frequency-pruned subset of the target `lm_head` rows
(FR-Spec-style "hot token" map, `N` of the full `V` rows) and/or (b) an FP8
per-tensor copy of those rows — **for the DRAFT only**. The target's own BF16
`lm_head` and its verify path are untouched.

## What it changes

Patched copies (byte-identical base, then minimal commented edits; every edited
block is marked `# [kearuga-drafthead]`):

- `sglang/srt/server_args.py` — two new server args:
  - `--speculative-dflash-token-map <path>`: a torch-saved 1-D int64 tensor of
    full-vocab hot token ids. When set, the DFlash2 draft scores candidates
    through only those `N` target `lm_head` rows and remaps the local top-k back
    to global ids.
  - `--speculative-dflash-head-quantization {none,fp8_e4m3}` (default `none`):
    quantize the draft `lm_head` rows to FP8-E4M3 per-tensor.
- `sglang/srt/models/dflash.py` — `DFlash2DraftModel` gets `self.hot_token_id`;
  `compute_candidates` remaps the pruned top-k ids to global ids before
  `build_lattice`/`sample_path` (codebooks stay full-vocab sized), and raises a
  clear error when a map is set with `tp>1`.
- `sglang/srt/speculative/dflash_worker_v2.py` — loads the map in `__init__`,
  builds/caches the pruned/FP8 draft head and assigns it to `draft_model.lm_head`
  before capture (in `_maybe_build_draft_sampler` and `_propose_selector_block`),
  publishes `hot_token_id` onto the draft model, logs the resulting head
  shape/dtype/bytes, and remaps local argmax to global in the greedy fallbacks.
- `sglang/srt/speculative/dflash_utils.py` — copied unchanged (no edits needed).
- `sglang/srt/speculative/dflash_head_utils.py` — **new** helper module
  (`load_hot_token_ids`, `gather_head_rows`, `quantize_fp8_per_tensor`,
  `default_input_scale`, `estimate_input_scale`, `build_dflash_draft_head`,
  `remap_local_to_global`). Imports only `torch` at module level; SGLang imports
  are lazy so it is unit-testable on a CPU host without SGLang installed.

`PATCH.diff` holds a unified diff per changed file (generated with `difflib`; apply with
`git -c core.autocrlf=false apply -p1 PATCH.diff` from the `sglang/python` directory, then copy the
new module `sglang/srt/speculative/dflash_head_utils.py` alongside — it is a new file and not part of the diff);
`MANIFEST.json` records the base and patched sha256 of every file so drift is
detected.

## How to boot

The serving script (`serve/start-dflash2.sh`) mounts overlay files read-only
over the image's sglang tree via `OVERLAY_DIR`/`OVERLAY_FILES` (lines 63-87).

```bash
OVERLAY_DIR=serve/overlay-drafthead \
OVERLAY_FILES="sglang/srt/models/dflash.py \
               sglang/srt/speculative/dflash_worker_v2.py \
               sglang/srt/speculative/dflash_utils.py \
               sglang/srt/speculative/dflash_head_utils.py \
               sglang/srt/server_args.py" \
EXTRA_SGLANG_ARGS="--speculative-dflash-token-map <dir>/draft-vocab-top65536.pt \
                   [--speculative-dflash-head-quantization fp8_e4m3]" \
bash serve/start-dflash2.sh
```

The token-map file is a torch-saved 1-D int64 tensor (or list) of full-vocab
hot token ids, e.g. `torch.save(torch.tensor([...], dtype=torch.int64), path)`.
It is validated to be 1-D, unique, and sorted ascending.

## Losslessness argument

The target model verifies **every** drafted token with its own untouched BF16
`lm_head` and full-vocab softmax. The draft only *proposes* candidates; a token
outside the hot set is never accepted unless the target independently confirms
it. Pruning/quantizing the draft head therefore changes which tokens are
*proposed* (and thus the acceptance rate / speedup), never the correctness of
the emitted text. The target's `lm_head`, its verify logits, and the
accept/bonus computation are not modified.

## Byte savings (target lm_head, hidden=5120)

| Head rows        | BF16 bytes | FP8-E4M3 bytes |
|------------------|------------|----------------|
| 248,320 (full)   | 2.54 GB    | 1.27 GB        |
| 65,536 (64K)     | 0.67 GB    | 0.34 GB        |

Full = 248320 * 5120 * 2 = 2.54 GB; 64K BF16 = 65536 * 5120 * 2 = 0.67 GB;
64K FP8 = 65536 * 5120 * 1 = 0.34 GB.

## Known limitations

- **tp=1 only** when a token map is set (the pruned head has no shard metadata;
  `compute_candidates` raises a clear error at `tp>1`).
- **Rejection-sampling verify modes are unsupported with a map** (the selector
  `q_rows` are computed over the pruned candidate set; the sampling-accept path
  is unchanged but only sees the hot candidates).
- **Acceptance drops for tokens outside the map**: if the target's greedy
  continuation is a token not in the hot set, the draft cannot propose it and
  that step accepts 0 draft tokens. Choose the hot set from a frequency
  analysis of the target workload.
- The FP8 arm is **experimental**: it uses a portable dequant-matmul shim
  (`_Fp8HeadShimMethod`) when the real `ModelOptFp8LinearMethod` cannot be
  constructed without a quant config. The static activation scale defaults to
  `8.0/448` when no warm-up hidden states are available. The BF16-pruned arm
  does not depend on it.
- The head is captured inside the draft CUDA graph, so the pruned/FP8 head
  object is assigned **before** capture and has a fixed shape.

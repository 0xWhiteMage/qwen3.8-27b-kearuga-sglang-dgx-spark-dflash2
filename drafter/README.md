# drafter/ — vendored Kearuga DFlash 2 draft-head assets

Byte-identical copies of the artifacts published on
[`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)
at commit `ae61d9b8863e145f1df343003347a79d7b0c90de` (**drafter release v1.0**). Nothing here is generated
at build or launch time; every file's SHA-256 is pinned in `SHA256SUMS` and
re-verified by `./start-dflash2.sh` before the server boots.

## Contents

| Path | What it is |
|---|---|
| `draft-vocab-v4-top65650.pt` | Torch-saved 1-D int64 tensor of the 65,650 hot token ids (sorted, unique) — the FR-Spec-style frequency-ranked draft vocabulary |
| `draft-vocab-v4-top65650.json` | The same ids as a JSON list |
| `sglang-overlay/` | 5 SGLang source files (4 patched + 1 new `dflash_head_utils.py`) plus `PATCH.diff`, `MANIFEST.json`, `README.md` |
| `SHA256SUMS` | `sha256sum -c` manifest of everything above |
| `BASE-SHA256SUMS` | SHA-256 of the *unpatched* files inside the pinned image, referenced by their in-image absolute paths |

## What the overlay does

The DFlash 2 drafter has no output head of its own: stock SGLang scores every
draft position through the target's full 248,320 × 5120 BF16 `lm_head`
(2.54 GB read per decode cycle). The overlay makes the **draft** score
candidates through only the 65,650 hot rows (0.67 GB) and remaps the local
top-k back to global vocabulary ids. The target model verifies every drafted
token with its own untouched full-vocabulary `lm_head`, so the emitted
distribution is unchanged — this changes *speed*, never *what the model says*.

Limitations: `tp = 1` only when a token map is set, and the overlay is pinned
to the image digest recorded in `BASE-SHA256SUMS`. On any other SGLang build, apply
`sglang-overlay/PATCH.diff` to your own tree (`git -c core.autocrlf=false apply -p1` from
`sglang/python`) and copy the new module `sglang/srt/speculative/dflash_head_utils.py`
alongside — it is a new file and not part of the diff — instead of mounting these files.

## How the launcher uses it

With `DRAFTER_PROFILE=kearuga` (the default), `start-dflash2.sh`:

1. runs `sha256sum -c drafter/SHA256SUMS` (fails closed on any mismatch);
2. runs the pinned image once with `sha256sum -c` against
   `drafter/BASE-SHA256SUMS` to prove the image's base files are the exact
   versions the overlay was built against (skippable with
   `OVERLAY_SKIP_BASE_CHECK=1` if you applied `PATCH.diff` yourself);
3. bind-mounts the five overlay files read-only over
   `/sgl-workspace/sglang/python/sglang/srt/...` inside the container and the
   token map at `/drafter/draft-vocab-v4-top65650.pt`;
4. passes `--speculative-dflash-token-map /drafter/draft-vocab-v4-top65650.pt`.

A correct boot prints
`[kearuga-drafthead] loaded hot token map: rows=65650 …` and
`[kearuga-drafthead] draft head rows=65650 dtype=torch.bfloat16 bytes=672256000 fp8=False hot_map=True`.

## Re-verify by hand

```bash
(cd drafter && sha256sum -c SHA256SUMS)
# and, against the image (no GPU needed for the check):
docker run --rm --entrypoint sha256sum \
  -v "$PWD/drafter/BASE-SHA256SUMS:/tmp/kearuga-base.sums:ro" \
  lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe \
  -c --quiet /tmp/kearuga-base.sums
```

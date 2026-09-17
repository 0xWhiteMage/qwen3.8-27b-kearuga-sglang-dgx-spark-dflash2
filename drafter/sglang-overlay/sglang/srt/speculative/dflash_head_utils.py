# [kearuga-drafthead] FR-Spec-style "hot token" head pruning + optional FP8
# per-tensor quantization for the DFlash2 DRAFT lm_head only.
#
# The target model verifies every drafted token with its own untouched BF16
# lm_head, so losslessness is preserved regardless of how the draft scores its
# candidates. This module builds a small stand-in ``lm_head`` module that the
# DFlash2 draft worker assigns to ``draft_model.lm_head`` before cuda-graph
# capture: it exposes only the ``N`` "hot" target lm_head rows (a
# frequency-pruned subset of the full ``V``-row vocabulary) and optionally
# stores them as FP8-E4M3 with per-tensor scales. ``compute_candidates`` then
# top-k's over the ``N``-row pruned logits and remaps the local indices back to
# global token ids via the hot-token map before building the selector lattice.
#
# Only ``torch`` is imported at module level so this file is unit-testable on a
# CPU host without the rest of SGLang installed; SGLang imports are lazy.

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# E4M3FN representable max; per-tensor FP8 scales are amax / 448.
_FP8_E4M3_MAX = 448.0


def load_hot_token_ids(path: str) -> torch.Tensor:
    """Load a 1-D int64 tensor of full-vocabulary "hot" token ids.

    Accepts a torch-saved tensor or list (``weights_only=True``). The result is
    validated to be 1-D, unique, and is returned sorted ascending as int64.
    """
    raw = torch.load(path, weights_only=True)
    if isinstance(raw, (list, tuple)):
        ids = torch.tensor(raw, dtype=torch.int64)
    elif isinstance(raw, torch.Tensor):
        ids = raw.to(torch.int64)
    else:
        raise ValueError(
            "dflash hot token map must be a 1-D int64 tensor or list, got "
            f"{type(raw).__name__}"
        )
    if ids.ndim != 1:
        raise ValueError(
            f"dflash hot token map must be 1-D, got shape={tuple(ids.shape)}"
        )
    if ids.numel() == 0:
        raise ValueError("dflash hot token map must be non-empty")
    ids = torch.unique(ids)
    if not torch.equal(ids, torch.sort(ids).values):
        # torch.unique already returns sorted; keep an explicit sort for clarity.
        ids = torch.sort(ids).values
    return ids.to(torch.int64)


def gather_head_rows(weight: torch.Tensor, hot_ids: torch.Tensor) -> torch.Tensor:
    """Return a contiguous [N, H] copy of the selected lm_head rows."""
    if weight.ndim != 2:
        raise ValueError(
            f"dflash lm_head weight must be 2-D, got shape={tuple(weight.shape)}"
        )
    idx = hot_ids.to(torch.long)
    return weight.index_select(0, idx).contiguous()


def quantize_fp8_per_tensor(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8-E4M3 quantization. Returns (w_fp8, weight_scale).

    ``weight_scale = amax(|w|) / 448`` as a float32 scalar tensor; ``w_fp8`` is
    ``float8_e4m3fn``. A zero scale (all-zero weight) is replaced by 1.0 so the
    cast never divides by zero.
    """
    w = w.float()
    amax = w.abs().max()
    scale = (amax / _FP8_E4M3_MAX).to(torch.float32)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    w_fp8 = (w / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def default_input_scale(hidden_amax: Optional[float] = None) -> torch.Tensor:
    """Static per-tensor FP8 activation scale.

    ``input_scale = hidden_amax / 448``. When the hidden-state magnitude is
    unknown (no warm-up forward available before cuda-graph capture), a
    conservative default amax of 8.0 is used: typical post-RMSNorm draft
    hidden states for this family sit well below 8.0, so 8.0/448 ~= 0.0179
    keeps activations inside the E4M3 range without saturating the scale. The
    FP8 arm is experimental; the BF16-pruned arm does not depend on it.
    """
    amax = 8.0 if hidden_amax is None else float(hidden_amax)
    return torch.tensor(amax / _FP8_E4M3_MAX, dtype=torch.float32)


def estimate_input_scale(
    draft_model: nn.Module, sample_hidden: Optional[torch.Tensor]
) -> torch.Tensor:
    """Compute the static FP8 activation scale once, before capture.

    ``amax(|h|) / 448`` from a warm-up draft hidden states tensor when one is
    available; otherwise fall back to :func:`default_input_scale`. This must
    run outside cuda-graph capture (it reads a tensor value onto the host).
    """
    del draft_model  # reserved for a future warm-up forward hook
    if sample_hidden is None:
        return default_input_scale(None)
    amax = float(sample_hidden.abs().max().item())
    if amax <= 0.0:
        amax = 8.0
    return default_input_scale(amax)


class _Fp8HeadShimMethod:
    """Capture-safe per-tensor FP8 lm_head GEMM used when the real
    ``ModelOptFp8LinearMethod`` cannot be constructed without a quant config.

    The apply path is a dequant-matmul: it dequantizes the FP8 weight with its
    per-tensor scale and statically quantizes the activation with the captured
    per-tensor ``input_scale``. It performs no data-dependent Python branching
    on tensor values, so it is safe to capture inside a cuda graph. The real
    ModelOpt kernel (``torch._scaled_mm`` / CUTLASS) is preferred at runtime
    when importable; this shim is the portable fallback and the path the CPU
    unit tests exercise.
    """

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = layer.weight.to(torch.float32) * layer.weight_scale
        qx = (x.float() / layer.input_scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        qx = qx.to(torch.float8_e4m3fn).to(torch.float32) * layer.input_scale
        out = qx.to(w.dtype) @ w.t()
        if bias is not None:
            out = out + bias
        return out


def _new_fp8_quant_method():
    """Try to construct the real ModelOpt FP8 method; fall back to the shim."""
    try:
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp8LinearMethod,
        )
        from sglang.srt.layers.quantization.modelopt_config import ModelOptFp8Config

        return ModelOptFp8LinearMethod(ModelOptFp8Config())
    except Exception:
        return _Fp8HeadShimMethod()


class _DFlashDraftHead(nn.Module):
    """A minimal lm_head stand-in: a weight buffer plus the attributes the
    DFlash2 draft code paths read (``org_vocab_size``, ``quant_method``,
    ``weight_scale``/``input_scale`` for FP8, ``hot_token_id``). It carries no
    parameters and no shard metadata (tp=1 only when a map is set)."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        org_vocab_size: int,
        quant_method,
        weight_scale: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor],
        hot_token_id: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.weight = weight
        self.org_vocab_size = int(org_vocab_size)
        # Code paths that read shard offsets fall back to tp=1 / start=0.
        self.org_vocab_start_index = 0
        self.vocab_start_index = 0
        self.shard_indices = None
        self.quant_method = quant_method
        self.weight_scale = weight_scale
        self.input_scale = input_scale
        self.hot_token_id = hot_token_id


def build_dflash_draft_head(
    target_lm_head: nn.Module,
    hot_ids: Optional[torch.Tensor],
    fp8: bool,
    device: torch.device,
    *,
    sample_hidden: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build a pruned and/or FP8 draft lm_head stand-in.

    - ``hot_ids is None and not fp8``: return ``target_lm_head`` unchanged (the
      existing zero-overhead path; no copy is made).
    - ``hot_ids is not None``: gather the ``N`` selected rows into a new
      ``[N, H]`` weight; ``org_vocab_size = N``.
    - ``fp8``: quantize the (possibly already pruned) rows to FP8-E4M3 with a
      per-tensor weight scale and a static input scale.

    The returned module is usable under cuda-graph capture: it has a fixed
    weight shape and its FP8 ``apply`` does no data-dependent branching.
    """
    if hot_ids is None and not fp8:
        return target_lm_head

    src_weight = target_lm_head.weight
    target_dtype = src_weight.dtype
    hot_on_device = None if hot_ids is None else hot_ids.to(device)

    if hot_on_device is not None:
        weight = gather_head_rows(src_weight.to(device), hot_on_device)
        org_vocab_size = int(weight.shape[0])
    else:
        weight = src_weight.to(device).contiguous()
        org_vocab_size = int(weight.shape[0])

    quant_method = None
    weight_scale = None
    input_scale = None
    if fp8:
        weight, weight_scale = quantize_fp8_per_tensor(weight)
        weight_scale = weight_scale.to(device)
        input_scale = estimate_input_scale(None, sample_hidden).to(device)
        quant_method = _new_fp8_quant_method()
    else:
        weight = weight.to(target_dtype)

    head = _DFlashDraftHead(
        weight=weight,
        org_vocab_size=org_vocab_size,
        quant_method=quant_method,
        weight_scale=weight_scale,
        input_scale=input_scale,
        hot_token_id=hot_on_device,
    )
    return head.to(device)


def remap_local_to_global(
    local_ids: torch.Tensor, hot_ids: Optional[torch.Tensor]
) -> torch.Tensor:
    """Map pruned-head local row indices back to full-vocabulary token ids.

    Identity when no hot-token map is set; otherwise ``hot_ids[local_ids]``.
    """
    if hot_ids is None:
        return local_ids
    return hot_ids.to(local_ids.device).index_select(0, local_ids.to(torch.long))

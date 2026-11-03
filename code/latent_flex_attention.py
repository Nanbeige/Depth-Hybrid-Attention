"""Inference attention over compact MLA K/V, without per-query-head copies."""

from functools import lru_cache

import torch
from torch.nn import functional as F


def _kernel(query, key, value, block_mask, scale, return_lse=False):
    from torch.nn.attention.flex_attention import flex_attention

    from transformers.integrations.flex_attention import get_flex_attention_lse_kwargs

    if not torch.compiler.is_compiling():
        raise RuntimeError("Latent FlexAttention requires torch.compile; dense fallback is disabled.")
    for tensor in (query, key, value):
        for axis in (0, 1, 3):
            torch._dynamo.mark_static(tensor, axis)
    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        # Torch 2.6's decoding specialization rejects 48:1 GQA. Small tiles also
        # keep the 512-wide accumulators within Ampere shared-memory limits.
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": 16, "BLOCK_N": 16},
        **get_flex_attention_lse_kwargs(return_lse),
    )


@lru_cache(maxsize=1)
def _compiled_kernel():
    return torch.compile(_kernel, fullgraph=True, dynamic=True)


def _make_block_mask(query, key, padding, offset, window_size=None):
    batch, _, q_length, _ = query.shape
    kv_length = key.shape[2]
    lengths = torch.tensor([q_length, kv_length], device=query.device)

    def mask_mod(b, h, q, k):
        position = q + offset
        allowed = (q < lengths[0]) & (k < lengths[1]) & (k <= position)
        if window_size is not None:
            left, right = window_size
            if left >= 0:
                allowed = allowed & (k >= position - left)
            if right >= 0:
                allowed = allowed & (k <= position + right)
        if padding is not None:
            allowed = allowed & padding[b, k.clamp(max=lengths[1] - 1)]
            allowed = allowed & padding[b, position.clamp(max=lengths[1] - 1)]
        return allowed

    return _compiled_mask()(
        mask_mod,
        B=batch,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=query.device,
        _compile=False,
    )


@lru_cache(maxsize=1)
def _compiled_mask():
    from torch.nn.attention.flex_attention import create_block_mask

    return torch.compile(create_block_mask, fullgraph=True, dynamic=True)


def latent_flex_attention(
    query,
    key,
    value,
    attention_mask,
    past_length,
    scale,
    window_size=None,
    sink_logits=None,
):
    """Causal attention with a bottom-right cache offset and optional 2D padding.

    K and V may be independent or shared latents. No head expansion occurs here.
    Non-power-of-two feature widths are zero-padded for older FlexAttention;
    the original model scale is retained and the output width is restored.
    """
    if query.device.type != "cuda":
        raise ValueError("Latent FlexAttention requires CUDA; select sdpa or eager on CPU.")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        raise ValueError("Latent FlexAttention currently supports inference only; use torch.no_grad().")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("Latent FlexAttention accepts only a 2D padding mask or None; use eager for 4D masks.")
    scale = float(scale)
    batch, heads, q_length, width = query.shape
    kv_length, v_width = value.shape[-2:]
    if key.shape[:3] != value.shape[:3] or key.shape[-1] != width or heads % key.shape[1]:
        raise ValueError("Incompatible latent FlexAttention Q/K/V shapes.")
    if min(q_length, kv_length, width, v_width) <= 0 or max(width, v_width) > 1024:
        raise ValueError("Latent FlexAttention requires nonempty inputs and head dimensions <= 1024.")
    # Triton dot lowering on the supported CUDA/PyTorch versions requires every
    # matrix dimension to be at least 16. Keep the original dimensions for the
    # returned slice while padding only the compiled Q/K/V operands.
    padded_width = max(16, 1 << (width - 1).bit_length())
    padded_v_width = max(16, 1 << (v_width - 1).bit_length())
    if padded_width != width:
        query = F.pad(query, (0, padded_width - width))
        key = F.pad(key, (0, padded_width - width))
    if padded_v_width != v_width:
        value = F.pad(value, (0, padded_v_width - v_width))
    if attention_mask is not None:
        if attention_mask.shape[0] != batch or attention_mask.shape[1] < kv_length:
            raise ValueError("Padding mask must cover the batch and the complete KV cache.")
        attention_mask = attention_mask[:, :kv_length].to(device=query.device, dtype=torch.bool)

    # Capture offsets as tensors so token-by-token decoding need not specialize
    # a new mask function for every Python integer cache length.
    offset = torch.tensor(past_length, device=query.device)

    # Put sequence first in storage: all Q strides then depend only on static
    # batch/head/feature sizes. Torch 2.6 can sort these strides without fixing
    # the sequence length (its FlexAttention output-layout heuristic otherwise
    # compares symbolic strides using Python bool).
    query = query.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)
    # Keep the existing budget for distinct model widths/dtypes/mask modes, not
    # sequence lengths. Scale is a model constant, not a dynamic scalar tensor.
    with torch._dynamo.config.patch(
        specialize_float=True,
        cache_size_limit=max(64, torch._dynamo.config.cache_size_limit),
    ):
        block_mask = _make_block_mask(query, key, attention_mask, offset, window_size)
        result = _compiled_kernel()(query, key, value, block_mask, scale, sink_logits is not None)
    if sink_logits is None:
        output = result
    else:
        output, auxiliary = result
        normalizer = (auxiliary.lse if hasattr(auxiliary, "lse") else auxiliary).float().unsqueeze(-1)
        # Apply the head-specific sink before V decoding or head merging.
        renorm = torch.exp(normalizer - torch.logaddexp(normalizer, sink_logits.float()))
        renorm = torch.where(torch.isneginf(normalizer), 0.0, renorm)
        output = (output.float() * renorm).to(query.dtype)
    return output[..., :v_width]

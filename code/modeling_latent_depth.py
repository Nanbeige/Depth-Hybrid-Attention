"""Latent depth attention with a separate shared positional K branch."""

import torch
from torch.nn import functional as F

from transformers.cache_utils import DynamicLayer
from .configuration_mla_depth import get_mla_depth_head_counts


_MATH_SDPA_MAX_SCORE_ELEMENTS = 64 * 1024 * 1024


def _materialize_latent_heads(tensor, num_heads):
    """Make the broadcast latent head dimension safe for fused SDPA kernels."""
    if tensor.shape[1] != num_heads:
        tensor = tensor.expand(-1, num_heads, -1, -1)
    # Fused CUDA kernels do not reliably support a zero-stride head dimension.
    # Materialize only this temporary attention input; the compact cache remains
    # [batch, 1, sequence, latent_rank].
    if tensor.stride(1) == 0 or not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _latent_scaled_dot_product_attention(query, key, value, **kwargs):
    """Run latent MLA with memory-efficient SDPA and a bounded math fallback.

    The latent cache is intentionally kept compact, so K/V arrive here as
    broadcast views. We materialize the head dimension only for the attention
    call, which avoids the fused-kernel illegal access without duplicating the
    long-lived cache. Math SDPA is retained for short or unsupported shapes,
    but is excluded for large score matrices because it materializes O(QK)
    attention weights and can exhaust GPU memory during needle evaluation.
    """
    key = _materialize_latent_heads(key, query.shape[1])
    value = _materialize_latent_heads(value, query.shape[1])
    score_elements = query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2]
    allow_math = score_elements <= _MATH_SDPA_MAX_SCORE_ELEMENTS

    attention = getattr(torch.nn, "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        backends = [attention.SDPBackend.EFFICIENT_ATTENTION]
        if allow_math:
            backends.append(attention.SDPBackend.MATH)
        with attention.sdpa_kernel(backends):
            return F.scaled_dot_product_attention(query, key, value, **kwargs)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=allow_math,
        enable_mem_efficient=True,
    ):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


class SharedLatentCacheLayer(DynamicLayer):
    """One allocation exposed through both K/V views for absorbed MLA attention."""

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = self.values = key_states.new_empty(0)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states is not value_states:
            raise ValueError("Shared latent cache requires the same K/V tensor.")
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = self.values = torch.cat([self.keys, key_states], dim=-2)
        return self.keys, self.values

    def reorder_cache(self, beam_idx):
        if self.get_seq_length() > 0:
            self.keys = self.values = self.keys.index_select(0, beam_idx.to(self.keys.device))

    def batch_repeat_interleave(self, repeats):
        if self.get_seq_length() > 0:
            self.keys = self.values = self.keys.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.get_seq_length() > 0:
            self.keys = self.values = self.keys[indices, ...]

    def crop(self, *args, **kwargs):
        super().crop(*args, **kwargs)
        self.values = self.keys

    def offload(self):
        if self.is_initialized:
            self.keys = self.values = self.keys.to("cpu", non_blocking=True)

    def prefetch(self):
        if self.is_initialized and self.keys.device != self.device:
            self.keys = self.values = self.keys.to(self.device, non_blocking=True)

    def reset(self):
        if self.is_initialized:
            self.keys.zero_()


class _RopeKeyCacheMixin:
    """Keep MLA's shared K positional branch alongside the latent cache."""

    def update(self, key_states, value_states, cache_kwargs=None):
        # Cache.update waits for prefetch before entering here and offloads after
        # returning. Keep all three branches inside that same transfer boundary.
        rope_key = cache_kwargs["mla_rope_key"]
        previous = getattr(self, "mla_rope_key", None)
        if previous is not None and previous.numel():
            rope_key = torch.cat([previous, rope_key], dim=-2)
        self.mla_rope_key = rope_key
        # The caller retains this device tensor for attention even after the
        # persistent cache tensors have been offloaded to CPU.
        cache_kwargs["mla_rope_key"] = rope_key
        return super().update(key_states, value_states, cache_kwargs)

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key = self.mla_rope_key.index_select(0, beam_idx.to(self.mla_rope_key.device))

    def crop(self, *args, **kwargs):
        super().crop(*args, **kwargs)
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key = self.mla_rope_key[..., : self.get_seq_length(), :]

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key = self.mla_rope_key.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key = self.mla_rope_key[indices, ...]

    def offload(self):
        super().offload()
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key = self.mla_rope_key.to("cpu", non_blocking=True)

    def prefetch(self):
        super().prefetch()
        if getattr(self, "mla_rope_key", None) is not None and self.mla_rope_key.device != self.device:
            self.mla_rope_key = self.mla_rope_key.to(self.device, non_blocking=True)

    def reset(self):
        super().reset()
        if getattr(self, "mla_rope_key", None) is not None:
            self.mla_rope_key.zero_()


class RopeKeyDynamicLayer(_RopeKeyCacheMixin, DynamicLayer):
    pass


class RopeKeySharedLatentCacheLayer(_RopeKeyCacheMixin, SharedLatentCacheLayer):
    pass


def _prepare_rope_key_cache(cache, cache_idx):
    while len(cache.layers) <= cache_idx:
        if cache.layer_class_to_replicate is None:
            raise ValueError("MLA shared RoPE key cache slot is missing.")
        cache.layers.append(cache.layer_class_to_replicate())
    layer = cache.layers[cache_idx]
    if isinstance(layer, (RopeKeyDynamicLayer, RopeKeySharedLatentCacheLayer)):
        return
    if layer.get_seq_length() != 0:
        raise ValueError("Cannot add MLA shared RoPE key cache to a populated layer.")
    if isinstance(layer, SharedLatentCacheLayer):
        cache.layers[cache_idx] = RopeKeySharedLatentCacheLayer()
    elif type(layer) is DynamicLayer:
        cache.layers[cache_idx] = RopeKeyDynamicLayer()
    else:
        raise ValueError("MLA shared RoPE key cache requires a full DynamicLayer cache.")


def _prepare_shared_cache(cache, cache_idx):
    """Upgrade only an empty full-attention slot, including hybrid/loop caches."""
    while len(cache.layers) <= cache_idx:
        if cache.layer_class_to_replicate is None:
            raise ValueError("Shared latent cache slot is missing.")
        cache.layers.append(cache.layer_class_to_replicate())
    layer = cache.layers[cache_idx]
    if isinstance(layer, SharedLatentCacheLayer):
        return
    if type(layer) is not DynamicLayer or layer.get_seq_length() != 0:
        raise ValueError("Shared latent attention requires an empty full DynamicLayer cache.")
    cache.layers[cache_idx] = SharedLatentCacheLayer()


def latent_depth_forward(
    module,
    hidden,
    attention_mask,
    cache,
    output_attentions,
    cache_position,
    cache_idx,
    depth_cache,
    position_ids=None,
    linear_attention_mask=None,
):
    from .modeling_nanbeige import (
        _append_depth_attention_anchor,
        _get_depth_attention_cache_group,
        _is_depth_attention_enabled,
        _resolve_depth_attention_cache,
    )

    cfg = module.config
    use_flex = cfg._attn_implementation == "flex_attention"
    if use_flex:
        if output_attentions:
            raise ValueError("Latent FlexAttention does not return attention weights; use eager.")
        if module.training:
            raise ValueError("Latent FlexAttention currently supports inference only; call model.eval().")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Latent FlexAttention requires a 2D padding mask or None.")
    batch, length, _ = hidden.shape
    cache_idx = module.layer_idx if cache_idx is None else cache_idx
    if cache is not None:
        # Dynamic full-attention layers also live inside the hybrid KDA cache.
        if cache_idx < len(cache.layers) and type(cache.layers[cache_idx]) not in (
            DynamicLayer,
            SharedLatentCacheLayer,
            RopeKeyDynamicLayer,
            RopeKeySharedLatentCacheLayer,
        ):
            raise ValueError("Latent depth attention requires a full DynamicLayer cache.")
    past_length = cache.get_seq_length(cache_idx) if cache is not None else 0
    if position_ids is None:
        positions = (
            cache_position
            if cache_position is not None
            else torch.arange(past_length, past_length + length, device=hidden.device)
        )
        position_ids = positions.unsqueeze(0)
    if cache is not None and cache_idx < len(cache.layers):
        previous = getattr(cache.layers[cache_idx], "mla_kv_share_latent", cfg.kv_share_latent)
        if past_length and previous != cfg.kv_share_latent:
            raise ValueError("Cannot change kv_share_latent on a populated cache.")
    cq = hidden if cfg.q_lora_rank is None else module.q_a_layernorm(module.q_a_proj(hidden))
    q_flat = module.q_proj(cq) if cfg.q_lora_rank is None else module.q_b_proj(cq)
    q, q_pe = (
        q_flat.view(batch, length, module.num_heads, module.q_head_dim)
        .transpose(1, 2)
        .split((cfg.qk_head_dim, cfg.qk_pos_emb_head_dim), dim=-1)
    )
    q_pe = module.q_pos_layernorm(q_pe)
    q_flat = q.transpose(1, 2).reshape(batch, length, -1)
    k_down = module.k_a_proj(hidden)
    if cfg.mla_kpos_from_mix_ck:
        ck, k_pe = k_down, None
    else:
        ck, k_pe = k_down.split((cfg.k_lora_rank, cfg.qk_pos_emb_head_dim), dim=-1)
    ck = module.k_a_layernorm(ck)
    if k_pe is not None:
        k_pe = module.k_pos_layernorm(k_pe)
    cv = ck if cfg.kv_share_latent else module.v_a_proj(hidden)
    k_flat = module.k_b_proj(ck)
    key = k_flat.view(batch, length, module.num_heads, cfg.qk_head_dim).transpose(1, 2)
    heads, k_heads, groups = get_mla_depth_head_counts(cfg)
    if depth_cache is None:
        raise ValueError("Combined MLA depth attention requires a per-forward depth cache.")
    entries, layer_id = _resolve_depth_attention_cache(
        depth_cache,
        module.layer_idx,
        _get_depth_attention_cache_group(cfg, module.layer_idx, "attention"),
    )
    apply_depth_attention = _is_depth_attention_enabled(cfg, "attention", module.layer_idx)
    if apply_depth_attention:
        outputs = []
        for kind, recovered, latent, count in (("q", q_flat, cq, heads), ("k", k_flat, ck, k_heads)):
            mode = getattr(cfg, f"depth_attention_mla_{kind}")
            if mode == "recover":
                projected = recovered
            elif mode == "latent":
                projected = latent
            elif mode == "independent":
                projected = getattr(module, f"depth_{kind}_layernorm")(getattr(module, f"depth_{kind}_proj")(hidden))
            else:
                projected = module.depth_q_layernorm(module.depth_query.view(1, -1))
                projected = projected.to(hidden.dtype).expand(batch, length, -1)
            outputs.append(projected.reshape(batch, length, count, -1))
        depth_query, depth_key = outputs
        if not entries:
            # Constant single-source routing still participates in autograd/DDP.
            cv = cv + ((depth_query.float().sum() + depth_key.float().sum()) * 0).to(cv.dtype)
        if entries:
            keys = [entry[1] for entry in entries] + [depth_key]
            values = [entry[2] for entry in entries] + [cv]
            if any(k.shape != depth_key.shape or v.shape != cv.shape for k, v in zip(keys, values)):
                raise ValueError("Incompatible latent depth anchor shapes.")
            grouped_q = depth_query.float().reshape(batch, length, groups, heads // groups, -1)
            if cfg.depth_attention_mla_group_mode == "qk":
                logits = torch.stack([(grouped_q * k.float().reshape_as(grouped_q)).sum(-1) for k in keys])
            else:
                logits = torch.stack([(grouped_q * k.float().unsqueeze(-2)).sum(-1) for k in keys])
            # Each head softmaxes over depth independently before group probability averaging.
            shared = (logits * depth_query.shape[-1] ** -0.5).softmax(dim=0).mean(dim=-1)
            source_values = torch.stack(values).reshape(len(values), batch, length, groups, cfg.v_lora_rank // groups)
            cv = (shared.to(cv.dtype).unsqueeze(-1) * source_values).sum(dim=0).reshape_as(cv)
        _append_depth_attention_anchor(entries, layer_id, cfg.depth_attention_stride, depth_key, cv)

    if cfg.mla_kpos_from_mix_ck and cfg.qk_pos_emb_head_dim:
        k_pe = module.k_pos_layernorm(module.k_pos_proj(cv))

    if cfg.mla_depth_attention_mode == "latent":
        # No historical K/V expansion. Absorb K per head and aggregate latent V;
        # restore only the current attention output through each head's V decoder.
        uk = module.k_b_proj.weight.view(module.num_heads, cfg.qk_head_dim, cfg.k_lora_rank)
        query = torch.einsum("bhtd,hdr->bhtr", q, uk)
        temporal_value = cv.unsqueeze(1)
        temporal_key = temporal_value if cfg.kv_share_latent else ck.unsqueeze(1)
    elif cfg.mla_depth_attention_mode == "expanded":
        if cfg.kv_share_latent:
            key = module.k_b_proj(cv).view(batch, length, module.num_heads, cfg.qk_head_dim).transpose(1, 2)
        query, temporal_key = q, key
        temporal_value = module.v_b_proj(cv).view(batch, length, module.num_heads, cfg.v_head_dim).transpose(1, 2)
    else:
        raise ValueError("mla_depth_attention_mode must be latent or expanded.")
    query_pe = module._apply_rope(q_pe, position_ids)
    key_pe = module._apply_rope(k_pe.unsqueeze(1), position_ids)

    if cache is not None:
        signature = (heads, k_heads, groups) + tuple(
            getattr(cfg, name)
            for name in (
                "depth_attention_mla_q",
                "depth_attention_mla_k",
                "depth_attention_mla_q_rank",
                "depth_attention_mla_k_rank",
                "depth_attention_mla_group_mode",
                "qk_layernorm",
                "qk_pos_emb_norm",
                "kv_share_latent",
                "q_lora_rank",
                "k_lora_rank",
                "v_lora_rank",
            )
        )
        if past_length:
            layer = cache.layers[cache_idx]
            if getattr(layer, "mla_depth_signature", None) != signature:
                raise ValueError("Cannot change depth MLA configuration on a populated cache.")
            if (
                getattr(layer, "mla_depth_attention_mode", None) != cfg.mla_depth_attention_mode
                or layer.keys.shape[1] != temporal_key.shape[1]
                or layer.keys.shape[-1] != temporal_key.shape[-1]
                or layer.values.shape[-1] != temporal_value.shape[-1]
            ):
                raise ValueError("Cannot change MLA latent/expanded mode on a populated cache.")
        if cfg.kv_share_latent and cfg.mla_depth_attention_mode == "latent":
            _prepare_shared_cache(cache, cache_idx)
        if cfg.qk_pos_emb_head_dim:
            _prepare_rope_key_cache(cache, cache_idx)
        cache_kwargs = {"cache_position": cache_position, "mla_rope_key": key_pe}
        temporal_key, temporal_value = cache.update(temporal_key, temporal_value, cache_idx, cache_kwargs)
        cache.layers[cache_idx].mla_kv_share_latent = cfg.kv_share_latent
        cache.layers[cache_idx].mla_depth_attention_mode = cfg.mla_depth_attention_mode
        cache.layers[cache_idx].mla_depth_signature = signature
    if query.shape[1] != module.num_heads:
        query = query.expand(-1, module.num_heads, -1, -1)
    if not use_flex:
        temporal_key = _materialize_latent_heads(temporal_key, module.num_heads)
    if cfg.qk_pos_emb_head_dim:
        if cache is not None:
            rope_key = cache_kwargs["mla_rope_key"]
        else:
            rope_key = key_pe
        if rope_key.shape[1] != temporal_key.shape[1]:
            rope_key = rope_key.expand(-1, temporal_key.shape[1], -1, -1)
        query = torch.cat((query, query_pe), dim=-1)
        temporal_key = torch.cat((temporal_key, rope_key), dim=-1)

    if use_flex:
        from .latent_flex_attention import latent_flex_attention

        output = latent_flex_attention(
            query,
            temporal_key,
            temporal_value,
            attention_mask,
            past_length,
            module.scaling,
            window_size=module.window_size,
            sink_logits=module._sink_logits(query, past_length, cache_position),
        )
        weights = None
    else:
        kv_length = temporal_key.shape[-2]
        positions = torch.arange(length, device=hidden.device) + past_length
        causal = torch.arange(kv_length, device=hidden.device)[None, :] <= positions[:, None]
        if module.window_size is not None:
            left, right = module.window_size
            keys = torch.arange(kv_length, device=hidden.device)[None, :]
            if left >= 0:
                causal &= keys >= positions[:, None] - left
            if right >= 0:
                causal &= keys <= positions[:, None] + right
        mask = query.new_zeros(batch, 1, length, kv_length).masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask.masked_fill_(~attention_mask[:, None, None, :kv_length].bool(), float("-inf"))
            elif attention_mask.dtype == torch.bool:
                mask.masked_fill_(~attention_mask[..., :kv_length], float("-inf"))
            else:
                mask = mask + attention_mask[..., :kv_length]
        dropout = cfg.attention_dropout if module.training else 0.0
        weights = None
        has_dynamic_padding = (
            linear_attention_mask is not None
            and linear_attention_mask.ndim == 2
            and bool(torch.any(linear_attention_mask == 0).item())
        )
        if module.sinks is not None:
            output, weights = module._attention_with_sink(
                query,
                temporal_key,
                temporal_value,
                mask,
                module._sink_logits(query, past_length, cache_position),
                dropout,
                output_attentions,
            )
        elif cfg._attn_implementation == "eager" or output_attentions:
            scores = (query @ temporal_key.transpose(-2, -1)) * module.scaling + mask
            weights = torch.nan_to_num(scores.float().softmax(dim=-1)).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=module.training)
            output = weights @ temporal_value
        elif has_dynamic_padding and past_length == 0 and module.window_size is None:
            # The fused CUDA SDPA kernels can fault when a broadcast latent K/V view
            # is combined with a 4-D padding mask.  Remove padding per sequence and
            # use the causal SDPA path without an explicit mask. The helper keeps the
            # score tensor fused and materializes only temporary per-head K/V inputs.
            # Evaluation collators use right padding, but selecting valid positions
            # also handles left padding without changing semantics.
            valid = linear_attention_mask.to(device=query.device).bool()
            outputs = []
            for batch_index in range(batch):
                positions = valid[batch_index].nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    outputs.append(query.new_zeros((1, query.shape[1], length, temporal_value.shape[-1])))
                    continue
                q_valid = query[batch_index : batch_index + 1].index_select(2, positions)
                k_valid = temporal_key[batch_index : batch_index + 1].index_select(2, positions)
                v_valid = temporal_value[batch_index : batch_index + 1].index_select(2, positions)
                valid_output = _latent_scaled_dot_product_attention(
                    q_valid,
                    k_valid,
                    v_valid,
                    attn_mask=None,
                    dropout_p=dropout,
                    is_causal=True,
                    scale=module.scaling,
                )
                outputs.append(
                    query.new_zeros((1, query.shape[1], length, temporal_value.shape[-1])).index_copy(
                        2, positions, valid_output
                    )
                )
            output = torch.cat(outputs, dim=0)
        else:
            # Keep the cache compact; the SDPA helper materializes temporary per-head
            # K/V inputs before selecting the memory-efficient backend.
            output = _latent_scaled_dot_product_attention(
                query,
                temporal_key,
                temporal_value,
                attn_mask=mask,
                dropout_p=dropout,
                scale=module.scaling,
            )
    if cfg.mla_depth_attention_mode == "latent":
        uv = module.v_b_proj.weight.view(module.num_heads, cfg.v_head_dim, cfg.v_lora_rank)
        output = torch.einsum("bhtr,hdr->bhtd", output, uv)
    output = output.transpose(1, 2).reshape(batch, length, module.num_heads * cfg.v_head_dim)
    if cfg.attention_output_gate:
        output = (output * module.g_proj(hidden).float().sigmoid()).to(output.dtype)
    return module.o_proj(output), weights if output_attentions else None, cache

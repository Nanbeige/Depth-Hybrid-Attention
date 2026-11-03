"""Megatron MLA with optional channel-wise output gating for exported Nanbeige models."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import StaticCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class NanbeigeMLAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        from .modeling_nanbeige import NanbeigeRMSNorm, _get_layer_window_size, _is_depth_attention_enabled

        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.window_size = _get_layer_window_size(config, layer_idx)
        use_sink = (config.window_layer_sink and self.window_size is not None) or (
            config.global_layer_sink and self.window_size is None
        )
        if use_sink:
            self.sinks = nn.Parameter(torch.empty(self.num_heads))
            nn.init.normal_(self.sinks, mean=0.0, std=config.initializer_range)
        else:
            self.register_parameter("sinks", None)
        self.combine_mla_depth_attention = config.combine_mla_depth_attention and (
            config.combine_window_mla_depth_attention or self.window_size is None
        )
        if self.window_size is not None and not config.combine_mla_depth_attention:
            raise ValueError("HF MLA currently requires full causal attention.")
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_layernorm = (
                NanbeigeRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps) if config.qk_layernorm else nn.Identity()
            )
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        if self.combine_mla_depth_attention:
            self.k_a_proj = nn.Linear(
                config.hidden_size,
                config.k_lora_rank + (0 if config.mla_kpos_from_mix_ck else config.qk_pos_emb_head_dim),
                bias=False,
            )
            if config.mla_kpos_from_mix_ck and config.qk_pos_emb_head_dim:
                self.k_pos_proj = nn.Linear(config.k_lora_rank, config.qk_pos_emb_head_dim, bias=False)
            if not config.kv_share_latent:
                self.v_a_proj = nn.Linear(config.hidden_size, config.v_lora_rank, bias=False)
            self.k_a_layernorm = (
                NanbeigeRMSNorm(config.k_lora_rank, eps=config.rms_norm_eps) if config.qk_layernorm else nn.Identity()
            )
            self.q_pos_layernorm = (
                NanbeigeRMSNorm(config.qk_pos_emb_head_dim, eps=config.rms_norm_eps)
                if config.qk_pos_emb_norm and config.qk_pos_emb_head_dim
                else nn.Identity()
            )
            self.k_pos_layernorm = (
                NanbeigeRMSNorm(config.qk_pos_emb_head_dim, eps=config.rms_norm_eps)
                if config.qk_pos_emb_norm and config.qk_pos_emb_head_dim
                else nn.Identity()
            )
            self.k_b_proj = nn.Linear(config.k_lora_rank, self.num_heads * config.qk_head_dim, bias=False)
            self.v_b_proj = nn.Linear(config.v_lora_rank, self.num_heads * config.v_head_dim, bias=False)
            depth_trainable = _is_depth_attention_enabled(config, "attention", layer_idx)
            for kind in ("q", "k"):
                mode = getattr(config, f"depth_attention_mla_{kind}")
                rank = getattr(config, f"depth_attention_mla_{kind}_rank")
                if mode == "fix":
                    self.depth_query = nn.Parameter(torch.empty(rank), requires_grad=depth_trainable)
                    nn.init.normal_(self.depth_query, mean=0.0, std=config.initializer_range)
                elif mode == "independent":
                    setattr(
                        self,
                        f"depth_{kind}_proj",
                        nn.Linear(config.hidden_size, rank, bias=False).requires_grad_(depth_trainable),
                    )
                if mode in ("fix", "independent"):
                    setattr(
                        self,
                        f"depth_{kind}_layernorm",
                        NanbeigeRMSNorm(rank, eps=config.rms_norm_eps).requires_grad_(depth_trainable),
                    )

        else:
            self.kv_a_proj_with_mqa = nn.Linear(
                config.hidden_size, config.kv_lora_rank + config.qk_pos_emb_head_dim, bias=False
            )
            self.kv_a_layernorm = (
                NanbeigeRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps) if config.qk_layernorm else nn.Identity()
            )
            self.q_pos_layernorm = (
                NanbeigeRMSNorm(config.qk_pos_emb_head_dim, eps=config.rms_norm_eps)
                if config.qk_pos_emb_norm and config.qk_pos_emb_head_dim
                else nn.Identity()
            )
            self.k_pos_layernorm = (
                NanbeigeRMSNorm(config.qk_pos_emb_head_dim, eps=config.rms_norm_eps)
                if config.qk_pos_emb_norm and config.qk_pos_emb_head_dim
                else nn.Identity()
            )
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank, self.num_heads * (config.qk_head_dim + config.v_head_dim), bias=False
            )
        self.o_proj = nn.Linear(self.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)
        if config.attention_output_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * config.v_head_dim, bias=False)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self):
        config = self.config

        def yarn_mscale(value):
            return (
                1.0 if config.rotary_scaling_factor <= 1 else 1 + 0.1 * value * math.log(config.rotary_scaling_factor)
            )

        self.scaling = yarn_mscale(config.mscale_all_dim) ** 2 / math.sqrt(self.q_head_dim)
        if self.combine_mla_depth_attention and config.qk_pos_emb_head_dim == 0:
            self.scaling = self.q_head_dim**-0.5
        layer_theta = (
            config.window_rope_theta
            if self.combine_mla_depth_attention and self.window_size is not None
            else config.global_rope_theta
        )
        theta = config.rope_theta if layer_theta is None else layer_theta
        self.rope_scale = 1.0
        if config.qk_pos_emb_head_dim == 0:
            return torch.empty(0, device="cpu", dtype=torch.float32)
        if config.mla_rope_type == "yarn":
            rope_config = copy.copy(config)
            rope_config.head_dim = config.qk_pos_emb_head_dim
            self.rope_scale = yarn_mscale(config.mscale) / yarn_mscale(config.mscale_all_dim)
            rope_config.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": theta,
                "factor": config.rotary_scaling_factor,
                "original_max_position_embeddings": config.original_max_position_embeddings,
                "beta_fast": config.beta_fast,
                "beta_slow": config.beta_slow,
                "attention_factor": self.rope_scale,
                "truncate": True,
            }
            with torch.device("cpu"):
                inv_freq, _ = ROPE_INIT_FUNCTIONS["yarn"](rope_config, device=torch.device("cpu"))
        else:
            dim = config.qk_pos_emb_head_dim
            if config.rotary_percent != 0:
                dim = int(dim * config.rotary_percent)
            if dim <= 0 or dim % 2:
                raise ValueError("MLA rotary dimension must be positive and even.")
            inv_freq = (
                torch.zeros(dim // 2, device="cpu", dtype=torch.float32)
                if config.rotary_percent == 0
                else 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
            )
        return inv_freq

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Frequency rounding in BF16 grows into a large phase error at long contexts.
        if self.inv_freq.device.type != "meta":
            self.inv_freq = self._compute_inv_freq().to(device=self.inv_freq.device)
        return self

    def _apply_rope(self, tensor, position_ids):
        if self.config.qk_pos_emb_head_dim == 0:
            return tensor
        # Megatron stores interleaved coordinates but rotates in half-split order.
        # Generation may provide positions as [B, 1, L] while prefill uses
        # [B, L]. Normalize both forms before adding the head broadcast axis.
        if position_ids.ndim == 3:
            if position_ids.shape[1] == 1:
                position_ids = position_ids[:, 0]
            elif position_ids.shape[0] == 1:
                position_ids = position_ids[0]
            else:
                position_ids = position_ids.reshape(position_ids.shape[0], -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != tensor.shape[-2]:
            position_ids = position_ids[..., -tensor.shape[-2] :]
        dim = self.inv_freq.numel() * 2
        rotated, tail = tensor[..., :dim], tensor[..., dim:]
        rotated = torch.cat((rotated[..., 0::2], rotated[..., 1::2]), dim=-1)
        angles = position_ids.float()[..., None] * self.inv_freq.float()[None, None, :]
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos = (angles.cos() * self.rope_scale).to(tensor.dtype)
        sin = (angles.sin() * self.rope_scale).to(tensor.dtype)
        left, right = rotated.chunk(2, dim=-1)
        return torch.cat((rotated * cos + torch.cat((-right, left), dim=-1) * sin, tail), dim=-1)

    def _sink_logits(self, query, past_length, cache_position=None):
        if self.sinks is None:
            return None
        batch, _, length, _ = query.shape
        logits = self.sinks.float().view(1, -1, 1, 1).expand(batch, -1, length, 1)
        if self.config.window_layer_sink_only_when_truncated and self.window_size is not None:
            positions = (
                cache_position[-length:].to(query.device)
                if cache_position is not None
                else torch.arange(length, device=query.device) + past_length
            )
            logits = logits.masked_fill(~(positions > self.window_size[0]).view(1, 1, length, 1), float("-inf"))
        return logits

    def _attention_with_sink(self, query, key, value, mask, sink_logits, dropout, output_attentions):
        if self.config._attn_implementation == "eager" or output_attentions:
            scores = (query @ key.transpose(-2, -1)) * self.scaling + mask
            logits = torch.cat((scores.float(), sink_logits), dim=-1)
            empty = torch.isneginf(logits).all(dim=-1, keepdim=True)
            weights = logits.masked_fill(empty, 0.0).softmax(dim=-1)[..., :-1]
            weights = weights.masked_fill(empty, 0.0).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            return weights @ value, weights

        from .modeling_latent_depth import _latent_scaled_dot_product_attention

        # A temporary zero K/V slot implements the output-less sink in SDPA.
        # It is added after cache.update, so the cache and absorption stay intact.
        mask = torch.cat((mask.expand(*query.shape[:3], key.shape[-2]), sink_logits.to(query.dtype)), dim=-1)
        output = _latent_scaled_dot_product_attention(
            query,
            F.pad(key, (0, 0, 0, 1)),
            F.pad(value, (0, 0, 0, 1)),
            attn_mask=mask,
            dropout_p=dropout,
            scale=self.scaling,
        )
        return output, None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        loop_cache_layer_idx=None,
        **kwargs,
    ):
        if self.combine_mla_depth_attention:
            from .modeling_latent_depth import latent_depth_forward

            linear_attention_mask = kwargs.pop("linear_attention_mask", None)
            return latent_depth_forward(
                self,
                hidden_states,
                attention_mask,
                past_key_value,
                output_attentions,
                cache_position,
                loop_cache_layer_idx,
                kwargs.pop("depth_attention_kv_cache", None),
                position_ids=position_ids,
                linear_attention_mask=linear_attention_mask,
            )
        batch, length, _ = hidden_states.shape
        config = self.config
        if isinstance(past_key_value, StaticCache):
            raise ValueError("HF MLA uses DynamicCache; StaticCache assumes equal K/V head dimensions.")
        cache_idx = self.layer_idx if loop_cache_layer_idx is None else loop_cache_layer_idx
        past_length = past_key_value.get_seq_length(cache_idx) if past_key_value is not None else 0
        if position_ids is None:
            positions = (
                cache_position
                if cache_position is not None
                else torch.arange(past_length, past_length + length, device=hidden_states.device)
            )
            position_ids = positions.unsqueeze(0)
        q = (
            self.q_proj(hidden_states)
            if config.q_lora_rank is None
            else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        )
        q = q.view(batch, length, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split((config.qk_head_dim, config.qk_pos_emb_head_dim), dim=-1)
        q_pe = self.q_pos_layernorm(q_pe)
        compressed, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            (config.kv_lora_rank, config.qk_pos_emb_head_dim), dim=-1
        )
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed))
            .view(batch, length, self.num_heads, config.qk_head_dim + config.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value = kv.split((config.qk_head_dim, config.v_head_dim), dim=-1)
        k_pe = self.k_pos_layernorm(k_pe)
        query = torch.cat((q_nope, self._apply_rope(q_pe, position_ids)), dim=-1)
        k_pe = self._apply_rope(k_pe.unsqueeze(1), position_ids).expand(-1, self.num_heads, -1, -1)
        key = torch.cat((k_nope, k_pe), dim=-1)
        from .modeling_nanbeige import _maybe_apply_depth_attention

        key, value = _maybe_apply_depth_attention(
            config,
            self.layer_idx,
            kwargs.pop("depth_attention_kv_cache", None),
            query,
            key,
            value,
            softmax_scale=self.scaling,
        )
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, cache_idx, {"cache_position": cache_position})
        dropout = self.attention_dropout if self.training else 0.0
        weights = None
        if config._attn_implementation == "flex_attention" and self.sinks is not None and not output_attentions:
            from .latent_flex_attention import latent_flex_attention

            if self.training:
                raise ValueError("MLA FlexAttention currently supports inference only; call model.eval().")
            output = latent_flex_attention(
                query,
                key,
                value,
                attention_mask,
                past_length,
                self.scaling,
                window_size=self.window_size,
                sink_logits=self._sink_logits(query, past_length, cache_position),
            ).transpose(1, 2)
        elif (
            config._attn_implementation == "flash_attention_2"
            and not output_attentions
            and self.window_size is None
            and self.sinks is None
        ):
            dim = max(self.q_head_dim, config.v_head_dim)
            output, _ = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
                self,
                F.pad(query, (0, dim - self.q_head_dim)),
                F.pad(key, (0, dim - self.q_head_dim)),
                F.pad(value, (0, dim - config.v_head_dim)),
                attention_mask,
                dropout=dropout,
                scaling=self.scaling,
            )
            output = output[..., : config.v_head_dim]
        else:
            kv_length = key.shape[-2]
            query_positions = torch.arange(length, device=query.device) + past_length
            causal = torch.arange(kv_length, device=query.device)[None, :] <= query_positions[:, None]
            mask = torch.zeros((batch, 1, length, kv_length), device=query.device, dtype=query.dtype)
            if self.window_size is not None:
                left, right = self.window_size
                if left >= 0:
                    causal &= torch.arange(kv_length, device=query.device)[None, :] >= query_positions[:, None] - left
            mask.masked_fill_(~causal, float("-inf"))
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    mask.masked_fill_(~attention_mask[:, None, None, :kv_length].bool(), float("-inf"))
                elif attention_mask.dtype == torch.bool:
                    mask.masked_fill_(~attention_mask[..., :kv_length], float("-inf"))
                else:
                    mask = mask + attention_mask[..., :kv_length]
            if self.sinks is not None:
                output, weights = self._attention_with_sink(
                    query,
                    key,
                    value,
                    mask,
                    self._sink_logits(query, past_length, cache_position),
                    dropout,
                    output_attentions,
                )
                output = output.transpose(1, 2)
            elif config._attn_implementation == "eager" or output_attentions:
                scores = query @ key.transpose(-2, -1) * self.scaling + mask
                weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
                weights = torch.nan_to_num(weights)
                weights = F.dropout(weights, p=dropout, training=self.training)
                output = (weights @ value).transpose(1, 2)
            else:
                output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=dropout, scale=self.scaling
                ).transpose(1, 2)
        output = output.reshape(batch, length, self.num_heads * config.v_head_dim)
        if config.attention_output_gate:
            output = (output * self.g_proj(hidden_states).float().sigmoid()).to(output.dtype)
        return self.o_proj(output), weights if output_attentions else None, past_key_value

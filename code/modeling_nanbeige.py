import inspect
import math
import warnings
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, LinearAttentionLayer, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_nanbeige import NanbeigeConfig
from .modeling_mla import NanbeigeMLAAttention


logger = logging.get_logger(__name__)

_flash_attn_import_error = None
_flash_attn_available = False
_flash_supports_window_size = False
flash_attn_func = None
flash_attn_varlen_func = None
index_first_axis = None
pad_input = None
unpad_input = None

if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    except (ImportError, OSError) as exc:
        # Metadata can report an installed package even when its CUDA extension
        # was built against a different PyTorch ABI.
        _flash_attn_import_error = exc
        logger.warning_once(
            "FlashAttention was detected but could not be imported; falling back to SDPA/eager attention. "
            "If flash_attention_2 is explicitly requested, install a build matching the active PyTorch. "
            f"Import error: {exc}"
        )
    else:
        _flash_attn_available = True
        _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
else:
    _flash_supports_window_size = False

_CONFIG_FOR_DOC = "NanbeigeConfig"

DepthAttentionCacheEntry = tuple[int, torch.Tensor, torch.Tensor]


class _DepthAttentionCacheState:
    """Keep independent Depth-Attention caches and their group-local layer ordinals."""

    def __init__(self):
        self.entries_by_group = {}
        self.layer_ids_by_group = {}

    def register_layer(self, layer_idx: int, group: str) -> None:
        layer_ids = self.layer_ids_by_group.setdefault(group, {})
        self.entries_by_group.setdefault(group, [])
        if layer_idx not in layer_ids:
            layer_ids[layer_idx] = len(layer_ids)

    def select(self, layer_idx: int, group: str) -> tuple[list[DepthAttentionCacheEntry], int]:
        try:
            isolated_layer_idx = self.layer_ids_by_group[group][layer_idx]
        except KeyError as exc:
            raise RuntimeError(f"Depth-Attention layer {layer_idx} was not registered in group {group!r}.") from exc
        return self.entries_by_group[group], isolated_layer_idx


DepthAttentionCache = list[DepthAttentionCacheEntry] | _DepthAttentionCacheState


def _get_linear_depth_logit_multiplier(config, key_head_dim, learnable_parameter=None):
    if config.linear_depth_logit_mul_learnable and learnable_parameter is not None:
        return learnable_parameter.float().exp()
    multiplier = config.linear_depth_logit_mul
    if multiplier is None:
        return key_head_dim**-0.5
    if multiplier == "sqrt":
        return math.sqrt(key_head_dim)
    return multiplier


def _resolve_depth_attention_cache(
    cache: DepthAttentionCache,
    layer_idx: int,
    group: str,
) -> tuple[list[DepthAttentionCacheEntry], int]:
    if isinstance(cache, _DepthAttentionCacheState):
        return cache.select(layer_idx, group)
    return cache, layer_idx


def _append_depth_attention_anchor(
    cache: list[DepthAttentionCacheEntry],
    layer_idx: int,
    stride: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Append every execution whose active layer ordinal is an anchor."""
    if layer_idx % stride == 0:
        cache.append((layer_idx, key, value))


def _get_unpad_data(attention_mask):
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("Linear-attention padding requires a 2D [batch, sequence] attention_mask.")
    attention_mask = attention_mask.to(dtype=torch.bool)
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    if torch.any(seqlens_in_batch == 0):
        raise ValueError("Each sequence must contain at least one non-padding token.")
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class NanbeigeLinearAttentionCache(Cache):
    """Cache layout for Nanbeige's mixed full- and linear-attention layers."""

    def __init__(self, config: NanbeigeConfig):
        num_loops = getattr(config, "num_loops", 1)
        loop_weights = getattr(config, "loop_loss_weights", [])
        if loop_weights:
            num_loops = len(loop_weights) + 1
        double_loop_split = getattr(config, "enable_double_loop_split", False) or getattr(
            config, "enable_double_loop_split_loopie", False
        )
        if double_loop_split:
            # Double-loop execution repeats the middle layers. Cache slots are
            # indexed by execution order, not by the original layer number.
            cache_layer_indices = [
                layer_idx
                for layer_idx, _ in _get_double_loop_split_layer_order_with_loop_indices(
                    config.num_hidden_layers,
                    getattr(config, "loop_middle_layers", None),
                    _get_loopie_style(config),
                )
            ]
        else:
            num_layers = config.num_hidden_layers * max(num_loops, 1)
            cache_layer_indices = [cache_idx % config.num_hidden_layers for cache_idx in range(num_layers)]
        layers = []
        for layer_idx in cache_layer_indices:
            pattern = getattr(config, "linear_attention_freq", None)
            use_linear = False
            if getattr(config, "experimental_attention_variant", None) == "kimi_delta_attention":
                if isinstance(pattern, int):
                    use_linear = (layer_idx + 1) % pattern != 0
                elif isinstance(pattern, list):
                    use_linear = bool(pattern[layer_idx])
            layers.append(LinearAttentionLayer() if use_linear else DynamicLayer())
        super().__init__(layers=layers)
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.seen_tokens + query_length, 0


class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NanbeigeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class NanbeigeKimiDeltaAttention(nn.Module):
    """Kimi Delta Attention matching Megatron's separate projection layout."""

    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.gate_dim = self.key_head_dim * self.num_value_heads
        self.gate_low_rank_dim = getattr(config, "linear_gate_low_rank_dim", None)
        if self.gate_low_rank_dim is None:
            self.gate_low_rank_dim = self.value_head_dim
        self.kda_use_full_rank_output_gate = getattr(config, "kda_use_full_rank_output_gate", False)
        self.kda_use_safe_gate = getattr(config, "kda_use_safe_gate", False)
        self.kda_gate_lower_bound = getattr(config, "kda_gate_lower_bound", -5.0)
        projection_bias = getattr(config, "kda_bias", False)
        conv_bias = getattr(config, "kda_conv_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, self.qk_dim, bias=projection_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.qk_dim, bias=projection_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.v_dim, bias=projection_bias)
        self.b_proj = nn.Linear(config.hidden_size, self.num_value_heads, bias=projection_bias)
        self.f_proj_down = nn.Linear(config.hidden_size, self.gate_low_rank_dim, bias=False)
        self.f_proj_up = nn.Linear(self.gate_low_rank_dim, self.gate_dim, bias=projection_bias)
        if self.kda_use_full_rank_output_gate:
            self.g_proj_down = None
            self.g_proj_up = nn.Linear(config.hidden_size, self.v_dim, bias=projection_bias)
        else:
            self.g_proj_down = nn.Linear(config.hidden_size, self.gate_low_rank_dim, bias=False)
            self.g_proj_up = nn.Linear(self.gate_low_rank_dim, self.v_dim, bias=projection_bias)
        self.conv1d = nn.Conv1d(
            self.qk_dim * 2 + self.v_dim,
            self.qk_dim * 2 + self.v_dim,
            config.linear_conv_kernel_dim,
            groups=self.qk_dim * 2 + self.v_dim,
            padding=config.linear_conv_kernel_dim - 1,
            bias=conv_bias,
        )
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.gate_dim))
        if config.enable_depth_attention and config.linear_depth_logit_mul_learnable:
            initial_multiplier = _get_linear_depth_logit_multiplier(config, self.key_head_dim)
            self.depth_logit_log_scale = nn.Parameter(
                torch.tensor(math.log(initial_multiplier)),
                requires_grad=_is_depth_attention_enabled(config, "linear"),
            )
        if config.enable_depth_attention and config.linear_depth_logit_scale:
            self.depth_q_scale = nn.Parameter(
                torch.ones(self.key_head_dim),
                requires_grad=_is_depth_attention_enabled(config, "linear"),
            )
            self.depth_k_scale = nn.Parameter(
                torch.ones(self.key_head_dim),
                requires_grad=_is_depth_attention_enabled(config, "linear"),
            )
        self.out_norm = NanbeigeRMSNorm(self.value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.v_dim, config.hidden_size, bias=projection_bias)

    def forward(self, hidden_states, **kwargs):
        # Keep KDA optional when HF inspects imports for trust_remote_code.
        try:
            from fla.modules.convolution import causal_conv1d
            from fla.modules.l2norm import l2norm
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
        except ImportError as exc:
            raise ImportError("KDA attention requires fla-core==0.5.2.") from exc

        past_key_value = kwargs.pop("past_key_value", None)
        use_cache = kwargs.pop("use_cache", False)
        linear_attention_mask = kwargs.pop("linear_attention_mask", None)
        loop_idx = kwargs.pop("loop_idx", 0)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        cache_layer_idx = _get_loop_cache_layer_idx(
            self.layer_idx, loop_idx, self.config.num_hidden_layers, loop_cache_layer_idx
        )
        original_bsz, original_seq_len, _ = hidden_states.shape
        bsz, seq_len, _ = hidden_states.shape
        has_padding = (
            linear_attention_mask is not None
            and linear_attention_mask.ndim == 2
            and torch.any(linear_attention_mask == 0)
        )
        is_packed = has_padding and (
            past_key_value is None
            or (
                isinstance(past_key_value, NanbeigeLinearAttentionCache)
                and not past_key_value.has_previous_state(cache_layer_idx)
            )
        )
        indices = cu_seqlens = None
        if is_packed:
            indices, cu_seqlens, _ = _get_unpad_data(linear_attention_mask)
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1]).index_select(0, indices).unsqueeze(0)
            bsz, seq_len = hidden_states.shape[:2]

        cache_layer = None
        has_previous_state = False
        if use_cache:
            if not isinstance(past_key_value, Cache) or cache_layer_idx >= len(past_key_value.layers):
                raise ValueError("KDA generation requires NanbeigeLinearAttentionCache.")
            cache_layer = past_key_value.layers[cache_layer_idx]
            if not isinstance(cache_layer, LinearAttentionLayer):
                raise ValueError(f"Cache layer {cache_layer_idx} is not a linear-attention cache layer.")
            has_previous_state = past_key_value.has_previous_state(cache_layer_idx)

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        beta = self.b_proj(hidden_states)
        forget = self.f_proj_up(self.f_proj_down(hidden_states))
        if self.g_proj_down is None:
            output_gate = self.g_proj_up(hidden_states)
        else:
            output_gate = self.g_proj_up(self.g_proj_down(hidden_states))
        qkv_input = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        if is_packed:
            qkv, conv_final_state = causal_conv1d(
                x=qkv_input.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif use_cache and has_previous_state and seq_len == 1:
            qkv, conv_final_state = causal_conv1d(
                x=qkv_input.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                initial_state=cache_layer.conv_states[0],
                output_final_state=True,
            )
        else:
            if use_cache:
                qkv_input = past_key_value.update_conv_state(
                    qkv_input, cache_layer_idx, conv_kernel_size=self.config.linear_conv_kernel_dim
                )
            qkv, conv_final_state = causal_conv1d(
                x=qkv_input.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
            if use_cache:
                qkv = qkv[:, -seq_len:]

        if use_cache and conv_final_state is not None:
            if not cache_layer.is_conv_states_initialized[0]:
                cache_layer.lazy_initialization(
                    conv_states=conv_final_state,
                    state_idx=0,
                    conv_kernel_size=self.config.linear_conv_kernel_dim,
                )
            cache_layer.conv_states[0].copy_(conv_final_state[..., -self.config.linear_conv_kernel_dim :])
            cache_layer.has_previous_state[0] = True
        query, key, value = qkv.split((self.qk_dim, self.qk_dim, self.v_dim), dim=-1)
        query = query.reshape(bsz, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(bsz, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(bsz, seq_len, self.num_value_heads, self.value_head_dim)
        forget = forget.reshape(bsz, seq_len, self.num_value_heads, self.key_head_dim)
        beta = beta.reshape(bsz, seq_len, self.num_value_heads)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        if depth_attention_kv_cache is not None and _is_depth_attention_enabled(self.config, "linear"):
            depth_attention_kv_cache, depth_layer_idx = _resolve_depth_attention_cache(
                depth_attention_kv_cache,
                self.layer_idx,
                _get_depth_attention_cache_group(self.config, self.layer_idx, "linear"),
            )
            query_for_depth = l2norm(query)
            key_for_depth = l2norm(key)
            if self.config.linear_depth_logit_scale:
                query_for_depth = query_for_depth * self.depth_q_scale
                key_for_depth = key_for_depth * self.depth_k_scale
            value_for_depth = value.transpose(1, 2)
            key_for_depth = key_for_depth.transpose(1, 2)
            query_for_depth = query_for_depth.transpose(1, 2)
            grouped_value = value_for_depth.reshape(
                bsz, self.num_key_heads, self.num_value_heads // self.num_key_heads, seq_len, self.value_head_dim
            ).mean(dim=2)
            value_for_depth, grouped_value = _depth_attention_mix_grouped_value(
                query_for_depth,
                key_for_depth,
                value_for_depth,
                ((source_key, source_value) for _, source_key, source_value in depth_attention_kv_cache),
                grouped_value,
                softmax_scale=(
                    _get_linear_depth_logit_multiplier(
                        self.config,
                        self.key_head_dim,
                        getattr(self, "depth_logit_log_scale", None),
                    )
                ),
            )
            value = value_for_depth.transpose(1, 2)
            _append_depth_attention_anchor(
                depth_attention_kv_cache,
                depth_layer_idx,
                self.config.depth_attention_stride,
                key_for_depth,
                grouped_value,
            )
        recurrent_state = cache_layer.recurrent_states[0] if use_cache and has_previous_state else None
        runner = fused_recurrent_kda if use_cache and has_previous_state and seq_len == 1 else chunk_kda
        gate_kwargs = {
            "lower_bound": self.kda_gate_lower_bound if self.kda_use_safe_gate else None,
        }
        if runner is chunk_kda:
            gate_kwargs["safe_gate"] = self.kda_use_safe_gate
        core, recurrent_final_state = runner(
            query,
            key,
            value,
            g=forget,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            cu_seqlens=cu_seqlens,
            **gate_kwargs,
        )
        if use_cache and recurrent_final_state is not None:
            past_key_value.update_recurrent_state(recurrent_final_state, cache_layer_idx)
        core = self.out_norm(core.reshape(-1, self.value_head_dim)).reshape(bsz, seq_len, self.v_dim)
        core = core * torch.sigmoid(output_gate.float()).to(core.dtype)
        output = self.out_proj(core)
        if is_packed:
            restored = output.new_zeros((original_bsz * original_seq_len, output.shape[-1]))
            restored.index_copy_(0, indices, output.squeeze(0))
            output = restored.view(original_bsz, original_seq_len, -1)
        return output, None, past_key_value if use_cache else None


ALL_LAYERNORM_LAYERS.append(NanbeigeRMSNorm)


def rms_norm_without_scale(hidden_states, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return (hidden_states * torch.rsqrt(variance + eps)).to(input_dtype)


def _get_loop_cache_layer_idx(
    layer_idx: int | None,
    loop_idx: int,
    num_hidden_layers: int,
    cache_layer_idx: int | None = None,
) -> int:
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop-aware caching is enabled.")
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers


def _maybe_apply_loop_share_kv(
    loop_share_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    layer_idx: int | None,
    loop_repeat_idx: int | None,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_key: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loop_share_kv_cache is None or loop_repeat_idx is None:
        return key_states, value_states
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop_share_kv is enabled.")
    cache_key = layer_idx if cache_key is None else cache_key
    if loop_repeat_idx == 0:
        loop_share_kv_cache[cache_key] = (key_states, value_states)
        return key_states, value_states
    if cache_key not in loop_share_kv_cache:
        raise RuntimeError(f"loop_share_kv missing first-pass KV for layer {layer_idx}.")
    return loop_share_kv_cache[cache_key]


def _maybe_apply_global_loop_share_kv(
    config: NanbeigeConfig,
    global_loop_share_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    layer_idx: int | None,
    loop_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_key: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    method = getattr(config, "global_loop_share_kv_method", None)
    if (
        global_loop_share_kv_cache is None
        or method in (None, "none")
        or _get_layer_window_size(config, layer_idx) is not None
    ):
        return key_states, value_states
    if layer_idx is None:
        raise ValueError("layer_idx must be set when global_loop_share_kv_method is enabled.")
    cache_key = layer_idx if cache_key is None else cache_key
    if loop_idx == 0:
        global_loop_share_kv_cache[cache_key] = (key_states, value_states)
        return key_states, value_states
    if cache_key not in global_loop_share_kv_cache:
        raise RuntimeError(f"global_loop_share_kv_method missing first-loop KV for layer {layer_idx}.")
    shared_key, shared_value = global_loop_share_kv_cache[cache_key]
    if method in ("kv", "k"):
        key_states = shared_key
    if method in ("kv", "v"):
        value_states = shared_value
    return key_states, value_states


def _reduce_query_to_kv_groups(query: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
    num_query_heads = query.shape[1]
    if num_query_heads == num_kv_groups:
        return query
    if num_query_heads % num_kv_groups != 0:
        raise ValueError(f"query heads ({num_query_heads}) must be divisible by KV groups ({num_kv_groups}).")
    return query.reshape(
        query.shape[0],
        num_kv_groups,
        num_query_heads // num_kv_groups,
        query.shape[2],
        query.shape[3],
    ).mean(dim=2)


def _depth_attention_mix_value(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    source_kv: list[tuple[torch.Tensor, torch.Tensor]],
    softmax_scale: float | None = None,
) -> torch.Tensor:
    source_kv = list(source_kv)
    if not source_kv:
        return current_value

    num_kv_groups = current_key.shape[1]
    query_for_kv = _reduce_query_to_kv_groups(query, num_kv_groups)
    if query_for_kv.shape != current_key.shape:
        raise ValueError(f"query/K shape mismatch after GQA grouping: {query_for_kv.shape} vs {current_key.shape}.")

    keys = [key for key, _ in source_kv] + [current_key]
    values = [value for _, value in source_kv] + [current_value]
    for key in keys:
        if key.shape != current_key.shape:
            raise ValueError(f"source key shape {key.shape} does not match {current_key.shape}.")
    for value in values:
        if value.shape != current_value.shape:
            raise ValueError(f"source value shape {value.shape} does not match {current_value.shape}.")

    key_stack = torch.stack(keys, dim=0)
    value_stack = torch.stack(values, dim=0)
    logits = (query_for_kv.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    depth_probs = torch.softmax(logits * softmax_scale, dim=0).to(value_stack.dtype)
    return (depth_probs.unsqueeze(-1) * value_stack).sum(dim=0).to(current_value.dtype)


def _attach_zero_grad_dependencies(value: torch.Tensor, *dependencies: object) -> torch.Tensor:
    """Keep learnable depth-attention inputs in autograd on identity paths."""
    zero = None
    for dependency in dependencies:
        if isinstance(dependency, torch.Tensor) and dependency.requires_grad:
            dependency_zero = dependency.float().sum() * 0.0
            zero = dependency_zero if zero is None else zero + dependency_zero
    if zero is None:
        return value
    return value + zero.to(dtype=value.dtype)


def _depth_attention_mix_grouped_value(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    source_kv: list[tuple[torch.Tensor, torch.Tensor]],
    current_grouped_value: torch.Tensor,
    softmax_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix distinct KDA V heads and return their post-mix group means."""
    if query.shape != current_key.shape:
        raise ValueError(f"query/K shape mismatch: {query.shape} vs {current_key.shape}.")
    num_groups = current_key.shape[1]
    num_value_heads = current_value.shape[1]
    if num_value_heads % num_groups != 0:
        raise ValueError(f"value heads ({num_value_heads}) must be a multiple of Q/K groups ({num_groups}).")
    grouped_value_shape = (current_value.shape[0], num_groups, current_value.shape[2], current_value.shape[3])
    if current_grouped_value.shape != grouped_value_shape:
        raise ValueError(
            f"current grouped value shape mismatch: {current_grouped_value.shape} vs {grouped_value_shape}."
        )
    source_kv = [
        (key, value)
        for key, value in source_kv
        if key.shape == current_key.shape and value.shape == grouped_value_shape
    ]
    if not source_kv:
        return (
            _attach_zero_grad_dependencies(current_value, query, current_key, softmax_scale),
            _attach_zero_grad_dependencies(current_grouped_value, query, current_key, softmax_scale),
        )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    key_stack = torch.stack([key for key, _ in source_kv] + [current_key], dim=0)
    source_value_stack = torch.stack([value for _, value in source_kv], dim=0)
    logits = (query.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
    depth_probs = torch.softmax(logits * softmax_scale, dim=0).to(current_value.dtype)
    source_mix = (depth_probs[:-1].unsqueeze(-1) * source_value_stack).sum(dim=0)
    current_prob = depth_probs[-1]
    heads_per_group = num_value_heads // num_groups
    grouped_current_value = current_value.reshape(
        current_value.shape[0],
        num_groups,
        heads_per_group,
        current_value.shape[2],
        current_value.shape[3],
    )
    mixed_value = source_mix.unsqueeze(2) + (current_prob.unsqueeze(2).unsqueeze(-1) * grouped_current_value)
    mixed_grouped_value = source_mix + current_prob.unsqueeze(-1) * current_grouped_value
    return (
        mixed_value.flatten(1, 2).to(current_value.dtype),
        mixed_grouped_value.to(current_grouped_value.dtype),
    )


def _maybe_apply_depth_attention(
    config: NanbeigeConfig,
    layer_idx: int | None,
    depth_attention_kv_cache: DepthAttentionCache | None,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    depth_query_states: torch.Tensor | None = None,
    depth_key_states: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth_attention_kv_cache is None or not _is_depth_attention_enabled(config, "attention", layer_idx):
        return key_states, value_states
    if layer_idx is None:
        raise ValueError("layer_idx must be set when enable_depth_attention=True.")

    depth_attention_kv_cache, depth_layer_idx = _resolve_depth_attention_cache(
        depth_attention_kv_cache,
        layer_idx,
        _get_depth_attention_cache_group(config, layer_idx, "attention"),
    )

    source_kv = [(key, value) for _, key, value in depth_attention_kv_cache]
    depth_query_states = query_states if depth_query_states is None else depth_query_states
    depth_key_states = key_states if depth_key_states is None else depth_key_states
    value_states = _depth_attention_mix_value(
        depth_query_states,
        depth_key_states,
        value_states,
        source_kv,
        softmax_scale=softmax_scale,
    )
    _append_depth_attention_anchor(
        depth_attention_kv_cache,
        depth_layer_idx,
        config.depth_attention_stride,
        depth_key_states,
        value_states,
    )
    return key_states, value_states


def _apply_depth_attention_then_update_cache(
    config: NanbeigeConfig,
    layer_idx: int | None,
    depth_attention_kv_cache: DepthAttentionCache | None,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: Cache | None,
    loop_idx: int,
    loop_cache_layer_idx: int | None,
    cache_kwargs: dict[str, Any],
    depth_query_states: torch.Tensor | None = None,
    depth_key_states: torch.Tensor | None = None,
    skip_cache_update: bool = False,
    softmax_scale: float | None = None,
    global_loop_share_kv_method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states, value_states = _maybe_apply_depth_attention(
        config,
        layer_idx,
        depth_attention_kv_cache,
        query_states,
        key_states,
        value_states,
        depth_query_states=depth_query_states,
        depth_key_states=depth_key_states,
        softmax_scale=softmax_scale,
    )
    if skip_cache_update and past_key_value is not None and global_loop_share_kv_method not in (None, "none"):
        cache_layer_idx = _get_loop_cache_layer_idx(
            layer_idx, loop_idx, config.num_hidden_layers, loop_cache_layer_idx
        )
        cache_layer = past_key_value.layers[cache_layer_idx]
        cached_key, cached_value = cache_layer.keys, cache_layer.values
        if cached_key.shape[-2] < key_states.shape[-2]:
            raise RuntimeError("Incremental KV cache is shorter than the current attention input.")
        prefix_length = cached_key.shape[-2] - key_states.shape[-2]
        key_states = torch.cat((cached_key[..., :prefix_length, :], key_states), dim=-2)
        value_states = torch.cat((cached_value[..., :prefix_length, :], value_states), dim=-2)
        return key_states, value_states
    if past_key_value is not None and not skip_cache_update:
        cache_layer_idx = _get_loop_cache_layer_idx(
            layer_idx, loop_idx, config.num_hidden_layers, loop_cache_layer_idx
        )
        key_states, value_states = past_key_value.update(key_states, value_states, cache_layer_idx, cache_kwargs)
    return key_states, value_states


def _get_double_loop_split_layer_order(
    num_hidden_layers: int,
    loop_middle_layers: int | None = None,
    loopie: bool = False,
) -> list[int]:
    return [
        layer_idx
        for layer_idx, _ in _get_double_loop_split_layer_order_with_loop_indices(
            num_hidden_layers, loop_middle_layers, loopie
        )
    ]


def _get_loopie_style(config) -> bool:
    """Return whether double-loop execution should be layer-major."""
    return getattr(config, "loop_style", "model") == "layer" or getattr(
        config, "enable_double_loop_split_loopie", False
    )


def _get_double_loop_split_layer_order_with_loop_indices(
    num_hidden_layers: int,
    loop_middle_layers: int | None = None,
    loopie: bool = False,
) -> list[tuple[int, int | None]]:
    if num_hidden_layers <= 0:
        raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
    if loop_middle_layers is None:
        if num_hidden_layers % 2 != 0:
            raise ValueError(
                "enable_double_loop_split requires num_hidden_layers to be divisible by 2 "
                "when loop_middle_layers is not set."
            )
        loop_middle_layers = num_hidden_layers // 2
    if loop_middle_layers <= 0:
        raise ValueError("loop_middle_layers must be greater than 0.")
    if num_hidden_layers % loop_middle_layers != 0:
        raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")

    first_unlooped_layers = (num_hidden_layers - loop_middle_layers) // 2
    middle_start = first_unlooped_layers
    middle_end = middle_start + loop_middle_layers
    middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
    if loopie:
        middle_order = [
            (idx, repeat_idx) for idx in range(middle_start, middle_end) for repeat_idx in range(middle_repeats)
        ]
    else:
        middle_order = [
            (idx, repeat_idx) for repeat_idx in range(middle_repeats) for idx in range(middle_start, middle_end)
        ]
    return (
        [(idx, None) for idx in range(0, middle_start)]
        + middle_order
        + [(idx, None) for idx in range(middle_end, num_hidden_layers)]
    )


class NanbeigeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        is_identity=False,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_identity = is_identity
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        if self.is_identity:
            inv_freq.zero_()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def _apply(self, fn, recurse=True):
        # Preserve the original FP32 frequencies across model.half()/bfloat16()
        # and .to(dtype=...). Casting back after rounding cannot recover phases.
        # Retain dynamic NTK frequencies as well as the default/identity ones.
        inv_freq = self.inv_freq
        super()._apply(fn, recurse=recurse)
        if inv_freq.device.type != "meta":
            self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        else:
            # Meta materialization is followed by _init_weights during HF load.
            self.inv_freq = self.inv_freq.float()
        return self

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NanbeigeLinearScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    """NanbeigeRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class NanbeigeDynamicNTKScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    """NanbeigeRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    k_embed = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return q_embed, k_embed


class NanbeigeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def _get_layer_window_size(config: NanbeigeConfig, layer_idx: int | None) -> tuple[int, int] | None:
    window_size = getattr(config, "window_size", None)
    if window_size is None:
        return None

    skip_freq = getattr(config, "window_attn_skip_freq", None)
    layer_number = 1 if layer_idx is None else layer_idx + 1
    if isinstance(skip_freq, int) and layer_number % skip_freq == 0:
        return None
    if isinstance(skip_freq, list) and not bool(skip_freq[layer_number - 1]):
        return None
    return tuple(window_size)


def _get_depth_attention_cache_group(config: NanbeigeConfig, layer_idx: int, base_group: str) -> str:
    # Disabled windows must not consume active layers' anchor-stride ordinals.
    if (
        base_group == "attention"
        and getattr(config, "window_no_depth_attention", False)
        and _get_layer_window_size(config, layer_idx) is not None
    ):
        return "disabled_window"
    if getattr(config, "combine_mla_depth_attention", False) and base_group == "attention":
        is_window = _get_layer_window_size(config, layer_idx) is not None
        if getattr(config, "combine_window_mla_depth_attention", False):
            if getattr(config, "window_depth_attention_isolation", False):
                return "latent_window" if is_window else "latent_global"
            return "latent_mla"
        if not is_window:
            return "latent_global"
    group = base_group if getattr(config, "linear_depth_attention_isolation", False) else "all"
    if getattr(config, "window_depth_attention_isolation", False):
        window_or_global = "window" if _get_layer_window_size(config, layer_idx) is not None else "global"
        group = f"{group}:{window_or_global}" if group != "all" else window_or_global
    return group


def _is_depth_attention_enabled(config: NanbeigeConfig, base_group: str, layer_idx: int | None = None) -> bool:
    """Return whether the current attention family should apply Depth-Attention."""
    if base_group == "attention":
        if getattr(config, "window_no_depth_attention", False):
            if layer_idx is None:
                raise ValueError("window_no_depth_attention requires layer_idx.")
            if _get_layer_window_size(config, layer_idx) is not None:
                return False
        # Preserve the legacy global switch, which disables all non-linear attention.
        return not getattr(config, "global_no_depth_attention", False)
    if base_group == "linear":
        return not getattr(config, "linear_no_depth_attention", False)
    return True


def _apply_sliding_window_mask(
    attention_mask: torch.Tensor | None,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cache_position: torch.LongTensor | None,
    window_size: tuple[int, int] | None,
) -> torch.Tensor | None:
    key_length = key_states.shape[-2]
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, :key_length]
    if window_size is None:
        return attention_mask

    query_length = query_states.shape[-2]
    if cache_position is not None and cache_position.numel() >= query_length:
        query_positions = cache_position[-query_length:].to(device=query_states.device)
    else:
        query_positions = torch.arange(key_length - query_length, key_length, device=query_states.device)
    key_positions = torch.arange(key_length, device=query_states.device)

    left_window, right_window = window_size
    local_mask = torch.zeros((query_length, key_length), dtype=torch.bool, device=query_states.device)
    if left_window != -1:
        local_mask |= key_positions.unsqueeze(0) < query_positions.unsqueeze(1) - left_window
    if right_window != -1:
        local_mask |= key_positions.unsqueeze(0) > query_positions.unsqueeze(1) + right_window

    if not local_mask.any():
        return attention_mask
    if attention_mask is None:
        attention_mask = torch.zeros(
            (1, 1, query_length, key_length),
            dtype=query_states.dtype,
            device=query_states.device,
        )
    return attention_mask.masked_fill(local_mask[None, None, :, :], torch.finfo(query_states.dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NanbeigeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: NanbeigeConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.window_size = _get_layer_window_size(config, layer_idx)
        layer_rope_theta = (
            getattr(config, "window_rope_theta", None)
            if self.window_size is not None
            else getattr(config, "global_rope_theta", None)
        )
        self.rope_theta = config.rope_theta if layer_rope_theta is None else layer_rope_theta
        self.global_layer_kvequal = getattr(config, "global_layer_kvequal", False) and self.window_size is None
        use_layer_sink = (getattr(config, "window_layer_sink", False) and self.window_size is not None) or (
            getattr(config, "global_layer_sink", False) and self.window_size is None
        )
        if use_layer_sink:
            self.sinks = nn.Parameter(torch.empty(self.num_heads))
            nn.init.normal_(self.sinks, mean=0.0, std=config.initializer_range)
        else:
            self.register_parameter("sinks", None)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        if config.attention_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.q_proj_loop = nn.ModuleList()
        if self._uses_global_loop_independent_q():
            for _ in range(1, self._get_loop_cycle_count()):
                self.q_proj_loop.append(
                    nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
                )
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        if config.qk_layernorm:
            self.q_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None
        self.q_layernorm_loop = nn.ModuleList()
        self.k_layernorm_loop = nn.ModuleList()
        if getattr(config, "loop_cycles_qknorm", False) and self._is_loop_middle_layer():
            for _ in range(1, self._get_loop_cycle_count()):
                self.q_layernorm_loop.append(NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps))
                self.k_layernorm_loop.append(NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps))

        self._init_rope()

    def _is_loop_middle_layer(self) -> bool:
        if self.layer_idx is None:
            return False
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            loop_middle_layers = self.config.num_hidden_layers // 2
        middle_start = (self.config.num_hidden_layers - loop_middle_layers) // 2
        return middle_start <= self.layer_idx < middle_start + loop_middle_layers

    def _get_loop_cycle_count(self) -> int:
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            loop_middle_layers = self.config.num_hidden_layers // 2
        return (self.config.num_hidden_layers + loop_middle_layers) // loop_middle_layers

    def _uses_global_loop_independent_q(self) -> bool:
        return (
            getattr(self.config, "global_loop_independent_q", False)
            and self._is_loop_middle_layer()
            and self.window_size is None
        )

    def _get_qk_layernorms(self, loop_idx: int):
        if loop_idx > 0 and loop_idx <= len(self.q_layernorm_loop):
            return self.q_layernorm_loop[loop_idx - 1], self.k_layernorm_loop[loop_idx - 1]
        return self.q_layernorm, self.k_layernorm

    def _get_q_proj(self, loop_idx: int | None):
        if loop_idx is not None and loop_idx > 0 and loop_idx <= len(self.q_proj_loop):
            return self.q_proj_loop[loop_idx - 1]
        return self.q_proj

    def _apply_v_norm_without_scale(self, value_states: torch.Tensor) -> torch.Tensor:
        if not getattr(self.config, "v_norm_without_scale", False):
            return value_states
        return rms_norm_without_scale(value_states, self.config.rms_norm_eps)

    def _apply_output_gate(self, attn_output, hidden_states):
        if not self.config.attention_output_gate:
            return attn_output
        gate = self.g_proj(hidden_states)
        return (attn_output * torch.sigmoid(gate.float())).to(attn_output.dtype)

    def _init_rope(self):
        rotary_percent = self.config.rotary_percent
        if self.window_size is None and self.config.global_layer_rotary_percent is not None:
            rotary_percent = self.config.global_layer_rotary_percent
        rotary_dim = int(self.head_dim * rotary_percent)
        # Match Megatron's frequency denominator before arange(0, dim, 2)
        # implicitly rounds the number of rotated coordinates up to even.
        if rotary_percent == 0.0:
            # Preserve the normal RoPE call path with an identity embedding.
            # This keeps global no-RoPE layers compatible with cached and fused
            # attention implementations that expect full-width cos/sin tensors.
            rotary_dim = self.head_dim
            self.rotary_emb = NanbeigeRotaryEmbedding(
                rotary_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                is_identity=True,
            )
        rope_parameters = self.config.rope_parameters
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rotary_percent != 0.0 and rope_type == "default":
            self.rotary_emb = NanbeigeRotaryEmbedding(
                rotary_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        elif rotary_percent != 0.0:
            scaling_factor = rope_parameters["factor"]
            if rope_type == "linear":
                self.rotary_emb = NanbeigeLinearScalingRotaryEmbedding(
                    rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif rope_type == "dynamic":
                self.rotary_emb = NanbeigeDynamicNTKScalingRotaryEmbedding(
                    rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unsupported custom Nanbeige RoPE type {rope_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        loop_idx = kwargs.pop("loop_idx", 0)
        global_loop_idx = kwargs.pop("global_loop_idx", loop_idx)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        global_loop_share_kv_cache = kwargs.pop("global_loop_share_kv_cache", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        loop_q_idx = kwargs.pop("loop_q_idx", loop_idx)
        q_proj = self._get_q_proj(loop_q_idx)
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            if self.global_layer_kvequal:
                value_states = key_states
            else:
                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = key_states if self.global_layer_kvequal else self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q_layernorm, k_layernorm = self._get_qk_layernorms(loop_idx)
        if q_layernorm is not None:
            query_states = q_layernorm(query_states)
        if k_layernorm is not None:
            key_states = k_layernorm(key_states)
        value_states = self._apply_v_norm_without_scale(value_states)

        depth_query_states = depth_key_states = None
        if depth_attention_kv_cache is not None and self.config.depth_attention_qk_nope:
            depth_query_states, depth_key_states = query_states, key_states

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_loop_shared_kv = loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        use_global_loop_shared_kv = (
            global_loop_share_kv_cache is not None and _get_layer_window_size(self.config, self.layer_idx) is None
        )
        skip_cache_update = (use_loop_shared_kv and loop_share_kv_repeat_idx > 0) or (
            use_global_loop_shared_kv and global_loop_idx > 0
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, self.config.num_hidden_layers, loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
            if depth_key_states is not None:
                depth_key_states, _ = _maybe_apply_loop_share_kv(
                    loop_share_kv_cache,
                    self.layer_idx,
                    loop_share_kv_repeat_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
                depth_key_states, _ = _maybe_apply_global_loop_share_kv(
                    self.config,
                    global_loop_share_kv_cache,
                    self.layer_idx,
                    global_loop_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                depth_query_states=depth_query_states,
                depth_key_states=depth_key_states,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
                global_loop_share_kv_method=(
                    self.config.global_loop_share_kv_method if use_global_loop_shared_kv else None
                ),
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        causal_mask = _apply_sliding_window_mask(
            attention_mask, query_states, key_states, cache_position, self.window_size
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        # Add an output-less logit to the softmax denominator for window-layer attention sinks.
        if self.sinks is not None:
            sink_logits = self.sinks.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
            if getattr(self.config, "window_layer_sink_only_when_truncated", False) and self.window_size is not None:
                if cache_position is not None and cache_position.numel() >= q_len:
                    query_positions = cache_position[-q_len:].to(device=attn_weights.device)
                else:
                    query_positions = torch.arange(
                        key_states.shape[-2] - q_len,
                        key_states.shape[-2],
                        device=attn_weights.device,
                    )
                sink_active = query_positions > self.window_size[0]
                sink_logits = sink_logits.masked_fill(~sink_active.reshape(1, 1, q_len, 1), float("-inf"))
            attn_weights = torch.cat([attn_weights, sink_logits.to(attn_weights.dtype)], dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.sinks is not None:
            attn_weights = attn_weights[..., :-1]
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self._apply_output_gate(attn_output, hidden_states)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=1
            )
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NanbeigeFlashAttention2(NanbeigeAttention):
    """
    Nanbeige flash attention module. This module inherits from `NanbeigeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        if not _flash_attn_available:
            raise ImportError(
                "NanbeigeFlashAttention2 requires a working flash-attn CUDA extension."
            ) from _flash_attn_import_error
        super().__init__(*args, **kwargs)

        if self.window_size is not None and not _flash_supports_window_size:
            raise ValueError("Hybrid sliding window attention requires a flash-attn version with window_size support.")

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        loop_idx = kwargs.pop("loop_idx", 0)
        global_loop_idx = kwargs.pop("global_loop_idx", loop_idx)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        global_loop_share_kv_cache = kwargs.pop("global_loop_share_kv_cache", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        loop_q_idx = kwargs.pop("loop_q_idx", loop_idx)
        if self.sinks is not None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                loop_idx=loop_idx,
                loop_q_idx=loop_q_idx,
                global_loop_idx=global_loop_idx,
                loop_cache_layer_idx=loop_cache_layer_idx,
                loop_share_kv_cache=loop_share_kv_cache,
                loop_share_kv_repeat_idx=loop_share_kv_repeat_idx,
                global_loop_share_kv_cache=global_loop_share_kv_cache,
                depth_attention_kv_cache=depth_attention_kv_cache,
                **kwargs,
            )
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self._get_q_proj(loop_q_idx)(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = key_states if self.global_layer_kvequal else self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q_layernorm, k_layernorm = self._get_qk_layernorms(loop_idx)
        if q_layernorm is not None:
            query_states = q_layernorm(query_states)
        if k_layernorm is not None:
            key_states = k_layernorm(key_states)
        value_states = self._apply_v_norm_without_scale(value_states)

        depth_query_states = depth_key_states = None
        if depth_attention_kv_cache is not None and self.config.depth_attention_qk_nope:
            depth_query_states, depth_key_states = query_states, key_states

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_loop_shared_kv = loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        use_global_loop_shared_kv = (
            global_loop_share_kv_cache is not None and _get_layer_window_size(self.config, self.layer_idx) is None
        )
        skip_cache_update = (use_loop_shared_kv and loop_share_kv_repeat_idx > 0) or (
            use_global_loop_shared_kv and global_loop_idx > 0
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, self.config.num_hidden_layers, loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
            if depth_key_states is not None:
                depth_key_states, _ = _maybe_apply_loop_share_kv(
                    loop_share_kv_cache,
                    self.layer_idx,
                    loop_share_kv_repeat_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
                depth_key_states, _ = _maybe_apply_global_loop_share_kv(
                    self.config,
                    global_loop_share_kv_cache,
                    self.layer_idx,
                    global_loop_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                depth_query_states=depth_query_states,
                depth_key_states=depth_key_states,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
                global_loop_share_kv_method=(
                    self.config.global_loop_share_kv_method if use_global_loop_shared_kv else None
                ),
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (NanbeigeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            window_size=self.window_size,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self._apply_output_gate(attn_output, hidden_states)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        window_size=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in NanbeigeFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            flash_window_kwargs = {}
            if _flash_supports_window_size:
                flash_window_kwargs["window_size"] = window_size if window_size is not None else (-1, -1)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                **flash_window_kwargs,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            flash_window_kwargs = {}
            if _flash_supports_window_size:
                flash_window_kwargs["window_size"] = window_size if window_size is not None else (-1, -1)
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                **flash_window_kwargs,
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class NanbeigeSdpaAttention(NanbeigeAttention):
    """
    Nanbeige attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `NanbeigeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    _supports_attention_sinks = False

    # Adapted from NanbeigeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        loop_idx = kwargs.pop("loop_idx", 0)
        global_loop_idx = kwargs.pop("global_loop_idx", loop_idx)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        global_loop_share_kv_cache = kwargs.pop("global_loop_share_kv_cache", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        loop_q_idx = kwargs.pop("loop_q_idx", loop_idx)
        if output_attentions or (self.sinks is not None and not self._supports_attention_sinks):
            if output_attentions:
                logger.warning_once(
                    "NanbeigeModel is using NanbeigeSdpaAttention, but "
                    "`torch.nn.functional.scaled_dot_product_attention` does not support "
                    "`output_attentions=True`. Falling back to the manual attention implementation."
                )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                loop_idx=loop_idx,
                loop_q_idx=loop_q_idx,
                global_loop_idx=global_loop_idx,
                loop_cache_layer_idx=loop_cache_layer_idx,
                loop_share_kv_cache=loop_share_kv_cache,
                loop_share_kv_repeat_idx=loop_share_kv_repeat_idx,
                global_loop_share_kv_cache=global_loop_share_kv_cache,
                depth_attention_kv_cache=depth_attention_kv_cache,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self._get_q_proj(loop_q_idx)(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = key_states if self.global_layer_kvequal else self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q_layernorm, k_layernorm = self._get_qk_layernorms(loop_idx)
        if q_layernorm is not None:
            query_states = q_layernorm(query_states)
        if k_layernorm is not None:
            key_states = k_layernorm(key_states)
        value_states = self._apply_v_norm_without_scale(value_states)

        depth_query_states = depth_key_states = None
        if depth_attention_kv_cache is not None and self.config.depth_attention_qk_nope:
            depth_query_states, depth_key_states = query_states, key_states

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_loop_shared_kv = loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        use_global_loop_shared_kv = (
            global_loop_share_kv_cache is not None and _get_layer_window_size(self.config, self.layer_idx) is None
        )
        skip_cache_update = (use_loop_shared_kv and loop_share_kv_repeat_idx > 0) or (
            use_global_loop_shared_kv and global_loop_idx > 0
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, self.config.num_hidden_layers, loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _maybe_apply_loop_share_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _maybe_apply_global_loop_share_kv(
                self.config,
                global_loop_share_kv_cache,
                self.layer_idx,
                global_loop_idx,
                key_states,
                value_states,
            )
            if depth_key_states is not None:
                depth_key_states, _ = _maybe_apply_loop_share_kv(
                    loop_share_kv_cache,
                    self.layer_idx,
                    loop_share_kv_repeat_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
                depth_key_states, _ = _maybe_apply_global_loop_share_kv(
                    self.config,
                    global_loop_share_kv_cache,
                    self.layer_idx,
                    global_loop_idx,
                    depth_key_states,
                    value_states,
                    cache_key=("depth_nope", self.layer_idx),
                )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                depth_query_states=depth_query_states,
                depth_key_states=depth_key_states,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
                global_loop_share_kv_method=(
                    self.config.global_loop_share_kv_method if use_global_loop_shared_kv else None
                ),
            )

        attn_output = self._compute_attention(query_states, key_states, value_states, attention_mask, cache_position)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self._apply_output_gate(attn_output, hidden_states)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _compute_attention(self, query_states, key_states, value_states, attention_mask, cache_position):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = _apply_sliding_window_mask(
            attention_mask, query_states, key_states, cache_position, self.window_size
        )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = causal_mask is None and query_states.shape[-2] > 1

        return torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )


def _nanbeige_flex_attention_kernel(query_states, key_states, value_states, **kwargs):
    if not torch.compiler.is_compiling():
        raise RuntimeError("Nanbeige FlexAttention must be compiled; refusing the dense uncompiled fallback.")
    from torch.nn.attention.flex_attention import flex_attention

    for states in (query_states, key_states, value_states):
        torch._dynamo.mark_static(states, 0)
        torch._dynamo.mark_static(states, 1)
        torch._dynamo.mark_static(states, 3)
    kernel_options = {"BLOCK_N": 32}
    if query_states.shape[-2] >= 128:
        kernel_options["BLOCK_M"] = 64
    return flex_attention(query_states, key_states, value_states, kernel_options=kernel_options, **kwargs)


@lru_cache(maxsize=1)
def _get_compiled_nanbeige_flex_attention():
    return torch.compile(_nanbeige_flex_attention_kernel, fullgraph=True, dynamic=True)


class NanbeigeFlexAttention(NanbeigeSdpaAttention):
    """Block-sparse causal attention with output-less sinks normalized using FP32 LSE."""

    _supports_attention_sinks = True

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        if hidden_states.device.type != "cuda":
            raise ValueError("Nanbeige flex_attention requires CUDA. Use sdpa or eager on CPU.")
        if output_attentions:
            raise ValueError(
                "Nanbeige flex_attention does not return attention weights. Use eager for output_attentions."
            )
        if self.training and self.attention_dropout > 0:
            raise ValueError("Nanbeige flex_attention requires attention_dropout=0 during training.")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "Nanbeige flex_attention accepts only a 2D padding mask or None; use eager for custom 4D masks."
            )
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            **kwargs,
        )

    def _compute_attention(self, query_states, key_states, value_states, attention_mask, cache_position):
        from torch.nn.attention.flex_attention import create_block_mask

        from transformers.integrations.flex_attention import get_flex_attention_lse_kwargs

        batch_size, num_heads, query_length, _ = query_states.shape
        key_length = key_states.shape[-2]
        if cache_position is None:
            query_positions = torch.arange(key_length - query_length, key_length, device=query_states.device)
        else:
            query_positions = cache_position[-query_length:].to(query_states.device).clone()
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=query_states.device, dtype=torch.bool).clone()
        left_window, right_window = self.window_size if self.window_size is not None else (-1, -1)
        window_bounds = torch.tensor((left_window, right_window), device=query_states.device)

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            query_position = query_positions[query_idx.clamp(max=query_length - 1)]
            allowed = (query_idx < query_length) & (key_idx < key_length) & (key_idx <= query_position)
            allowed = allowed & ((window_bounds[0] < 0) | (key_idx >= query_position - window_bounds[0]))
            allowed = allowed & ((window_bounds[1] < 0) | (key_idx <= query_position + window_bounds[1]))
            if attention_mask is not None:
                mask_length = attention_mask.shape[-1]
                allowed = allowed & (key_idx < mask_length)
                allowed = allowed & attention_mask[batch_idx, key_idx.clamp(max=mask_length - 1)].bool()
            return allowed

        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=query_states.device,
            _compile=True,
        )
        enable_gqa = True
        if num_heads & (num_heads - 1):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            enable_gqa = False

        # Torch 2.6 sorts Q strides with Python comparisons during FlexAttention
        # lowering. A changed batch size can leave symbolic sequence strides
        # whose order it cannot determine. Store Q as [S, B, H, D], then expose
        # [B, H, S, D]: strides depend only on the static batch/head/feature sizes.
        # Keep sequence lengths dynamic, as in latent_flex_attention.
        query_states = query_states.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)

        with torch._dynamo.config.patch(cache_size_limit=max(64, torch._dynamo.config.cache_size_limit)):
            result = _get_compiled_nanbeige_flex_attention()(
                query_states,
                key_states,
                value_states,
                block_mask=block_mask,
                enable_gqa=enable_gqa,
                **get_flex_attention_lse_kwargs(self.sinks is not None),
            )
        if self.sinks is None:
            return result

        attention_output, auxiliary = result
        log_normalizer = (auxiliary.lse if hasattr(auxiliary, "lse") else auxiliary).float()
        sink_logits = self.sinks.float().view(1, -1, 1)
        if self.config.window_layer_sink_only_when_truncated and self.window_size is not None:
            sink_logits = sink_logits.expand(batch_size, -1, query_length).masked_fill(
                ~(query_positions > left_window).view(1, 1, -1), float("-inf")
            )
        combined_normalizer = torch.logaddexp(log_normalizer, sink_logits)
        renorm = torch.exp(log_normalizer - combined_normalizer)
        renorm = torch.where(torch.isneginf(log_normalizer), 0.0, renorm)
        return (attention_output.float() * renorm.unsqueeze(-1)).to(query_states.dtype)


NANBEIGE_ATTENTION_CLASSES = {
    "eager": NanbeigeAttention,
    "flash_attention_2": NanbeigeFlashAttention2,
    "sdpa": NanbeigeSdpaAttention,
    "flex_attention": NanbeigeFlexAttention,
}


class NanbeigeDecoderLayer(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self._loop_middle_layer = (
            self._is_loop_middle_layer()
            if (
                getattr(config, "enable_double_loop_split", False)
                or getattr(config, "enable_double_loop_split_loopie", False)
            )
            else False
        )

        use_default_mla_flex = (
            config.multi_latent_attention
            and getattr(config, "combine_mla_depth_attention", False)
            and getattr(config, "mla_depth_attention_mode", "latent") == "latent"
            and torch.cuda.is_available()
            and not getattr(config, "_attn_implementation_was_explicit", False)
        )
        if use_default_mla_flex and config._attn_implementation in (None, "eager", "sdpa"):
            config._attn_implementation = "flex_attention"
        full_attention_cls = (
            NanbeigeMLAAttention
            if config.multi_latent_attention
            else NANBEIGE_ATTENTION_CLASSES[config._attn_implementation]
        )

        if getattr(config, "experimental_attention_variant", None) == "kimi_delta_attention":
            pattern = config.linear_attention_freq
            use_linear = isinstance(pattern, int) and ((layer_idx + 1) % pattern != 0)
            if isinstance(pattern, list):
                use_linear = bool(pattern[layer_idx])
            linear_cls = NanbeigeKimiDeltaAttention
            self.self_attn = (
                linear_cls(config, layer_idx) if use_linear else full_attention_cls(config=config, layer_idx=layer_idx)
            )
        else:
            self.self_attn = full_attention_cls(config=config, layer_idx=layer_idx)

        self.mlp = NanbeigeMLP(config)
        self.input_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_loop = nn.ModuleList()
        self.post_attention_layernorm_loop = nn.ModuleList()
        if getattr(config, "loop_cycles_layernorm", False) and self._loop_middle_layer:
            for _ in range(1, self._get_loop_cycle_count()):
                self.input_layernorm_loop.append(NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
                self.post_attention_layernorm_loop.append(NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

    def _get_loop_cycle_count(self) -> int:
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            loop_middle_layers = self.config.num_hidden_layers // 2
        return (self.config.num_hidden_layers + loop_middle_layers) // loop_middle_layers

    def _get_loop_middle_bounds(self) -> tuple[int, int]:
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            if self.config.num_hidden_layers <= 0 or self.config.num_hidden_layers % 2 != 0:
                raise ValueError("Loop execution requires loop_middle_layers or even num_hidden_layers.")
            loop_middle_layers = self.config.num_hidden_layers // 2
        first_unlooped_layers = (self.config.num_hidden_layers - loop_middle_layers) // 2
        return first_unlooped_layers, first_unlooped_layers + loop_middle_layers

    def _is_loop_middle_layer(self) -> bool:
        middle_start, middle_end = self._get_loop_middle_bounds()
        return middle_start <= self.layer_idx < middle_end

    def _get_loop_layernorms(self, loop_repeat_idx: int | None):
        if loop_repeat_idx is not None and loop_repeat_idx > 0 and loop_repeat_idx <= len(self.input_layernorm_loop):
            return (
                self.input_layernorm_loop[loop_repeat_idx - 1],
                self.post_attention_layernorm_loop[loop_repeat_idx - 1],
            )
        return self.input_layernorm, self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        loop_idx: int = 0,
        loop_cache_layer_idx: int | None = None,
        loop_repeat_idx: int | None = None,
        loop_share_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        global_loop_share_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        depth_attention_kv_cache: DepthAttentionCache | None = None,
        linear_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        input_layernorm, post_attention_layernorm = self._get_loop_layernorms(loop_repeat_idx)

        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        global_loop_idx = loop_repeat_idx if loop_repeat_idx is not None else loop_idx
        loop_q_idx = loop_repeat_idx if getattr(self.config, "global_loop_independent_q", False) else loop_idx

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            loop_idx=loop_idx,
            loop_q_idx=loop_q_idx,
            loop_cache_layer_idx=loop_cache_layer_idx,
            loop_share_kv_cache=loop_share_kv_cache,
            loop_share_kv_repeat_idx=loop_repeat_idx,
            global_loop_share_kv_cache=global_loop_share_kv_cache,
            global_loop_idx=global_loop_idx,
            depth_attention_kv_cache=depth_attention_kv_cache,
            linear_attention_mask=linear_attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


NANBEIGE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NanbeigeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Nanbeige Model outputting raw hidden-states without any specific head on top.",
    NANBEIGE_START_DOCSTRING,
)
class NanbeigePreTrainedModel(PreTrainedModel):
    config_class = NanbeigeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanbeigeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = _flash_attn_available
    _supports_flash_attn_2 = _flash_attn_available
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def get_correct_attn_implementation(self, requested_attention, is_init_check=False):
        # Select before the generic None -> SDPA default so explicit loading
        # overrides remain authoritative, including with a supplied config.
        if (
            requested_attention is None
            and not self.config.multi_latent_attention
            and self.config.window_layer_sink
            and torch.cuda.is_available()
            and any(
                _get_layer_window_size(self.config, layer_idx) is not None
                for layer_idx in range(self.config.num_hidden_layers)
            )
        ):
            requested_attention = "flex_attention"
        return super().get_correct_attn_implementation(requested_attention, is_init_check)

    def _supports_default_dynamic_cache(self) -> bool:
        return self.config.num_loops == 1 and super()._supports_default_dynamic_cache()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, NanbeigeMLAAttention):
            init.copy_(module.inv_freq, module._compute_inv_freq())
            if module.sinks is not None:
                init.normal_(module.sinks, mean=0.0, std=std)
            if hasattr(module, "depth_query"):
                init.normal_(module.depth_query, mean=0.0, std=std)
        elif isinstance(module, NanbeigeRotaryEmbedding):
            if module.is_identity:
                init.zeros_(module.inv_freq)
                return
            inv_freq = 1.0 / (
                module.base
                ** (torch.arange(0, module.dim, 2, dtype=torch.float32, device=module.inv_freq.device) / module.dim)
            )
            init.copy_(module.inv_freq, inv_freq)


NANBEIGE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Nanbeige Model outputting raw hidden-states without any specific head on top.",
    NANBEIGE_START_DOCSTRING,
)
class NanbeigeModel(NanbeigePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`NanbeigeDecoderLayer`]

    Args:
        config: NanbeigeConfig
    """

    def __init__(self, config: NanbeigeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [NanbeigeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_num_loops(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False) or getattr(
            self.config, "enable_double_loop_split_loopie", False
        ):
            return 1
        loop_weights = getattr(self.config, "loop_loss_weights", [])
        if loop_weights is not None and len(loop_weights) > 0:
            return len(loop_weights) + 1
        return getattr(self.config, "num_loops", 1)

    def _get_layer_order(self) -> list[int]:
        if getattr(self.config, "enable_double_loop_split", False) or getattr(
            self.config, "enable_double_loop_split_loopie", False
        ):
            return _get_double_loop_split_layer_order(
                self.config.num_hidden_layers,
                getattr(self.config, "loop_middle_layers", None),
                _get_loopie_style(self.config),
            )
        return list(range(self.config.num_hidden_layers))

    def _get_layer_execution_order(self) -> list[tuple[int, int | None]]:
        if getattr(self.config, "enable_double_loop_split", False) or getattr(
            self.config, "enable_double_loop_split_loopie", False
        ):
            return _get_double_loop_split_layer_order_with_loop_indices(
                self.config.num_hidden_layers,
                getattr(self.config, "loop_middle_layers", None),
                _get_loopie_style(self.config),
            )
        return [(layer_idx, None) for layer_idx in range(self.config.num_hidden_layers)]

    def _get_cache_seq_length(self, past_key_values: Cache | None) -> int:
        if past_key_values is None:
            return 0
        max_seq_length = 0
        for loop_idx in range(self._get_num_loops()):
            layer_idx = loop_idx * self.config.num_hidden_layers
            max_seq_length = max(max_seq_length, past_key_values.get_seq_length(layer_idx))
        return max_seq_length

    @add_start_docstrings_to_model_forward(NANBEIGE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_generation: bool = False,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        is_linear_attention_model = (
            getattr(self.config, "experimental_attention_variant", None) == "kimi_delta_attention"
        )
        if is_linear_attention_model and not is_generation:
            use_cache = False
            past_key_values = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        num_loops = self._get_num_loops()
        double_loop_split = getattr(self.config, "enable_double_loop_split", False) or getattr(
            self.config, "enable_double_loop_split_loopie", False
        )

        # Transformers 5 cache containers are the only supported cache format.
        if use_cache and past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError(
                "Transformers 5 requires `past_key_values` to be a Cache instance; "
                "legacy tuple caches are no longer supported."
            )

        # Generic generation code can initialize an empty DynamicCache before the
        # model-specific generate() override runs. Linear-attention layers need the
        # mixed cache so their recurrent and convolution states have the right layout.
        if (
            use_cache
            and is_linear_attention_model
            and past_key_values is not None
            and not isinstance(past_key_values, NanbeigeLinearAttentionCache)
        ):
            if past_key_values.get_seq_length() == 0:
                past_key_values = NanbeigeLinearAttentionCache(self.config)
            else:
                raise ValueError("KDA generation requires NanbeigeLinearAttentionCache for a non-empty cache.")

        # Initialize the Transformers 5 cache implementation required by this model.
        if (
            use_cache
            and past_key_values is None
            and getattr(self.config, "experimental_attention_variant", None) == "kimi_delta_attention"
        ):
            past_key_values = NanbeigeLinearAttentionCache(self.config)
        if use_cache and past_key_values is None and (num_loops > 1 or double_loop_split):
            past_key_values = DynamicCache()
        elif use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if use_cache and isinstance(past_key_values, StaticCache) and (num_loops > 1 or double_loop_split):
            raise ValueError(
                "StaticCache is not supported when loop-aware caching is enabled. Please use the default dynamic cache."
            )
        if use_cache and getattr(self.config, "enable_depth_attention", False):
            if getattr(self.config, "loop_share_kv", False):
                raise ValueError(
                    "enable_depth_attention with loop_share_kv does not support use_cache=True/generation."
                )
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "StaticCache is not supported with enable_depth_attention. Please use the default dynamic cache."
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = self._get_cache_seq_length(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
            else:
                position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        linear_attention_mask = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None

        # embed positions
        hidden_states = inputs_embeds

        last_loop_all_hidden_states = None
        last_loop_all_self_attns = None
        last_loop_next_decoder_cache = None

        layer_order = self._get_layer_execution_order()
        layer_lookup = list(self.layers)
        loop_share_kv_cache = {} if getattr(self.config, "loop_share_kv", False) else None
        if loop_share_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("loop_share_kv does not support gradient checkpointing during training.")
        global_loop_share_kv_cache = (
            {}
            if (num_loops > 1 or double_loop_split)
            and getattr(self.config, "global_loop_share_kv_method", None) not in (None, "none")
            else None
        )
        if global_loop_share_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("global_loop_share_kv_method does not support gradient checkpointing during training.")
        global_loop_cache_idx_by_execution = []
        global_loop_cache_idx_by_layer = {}
        next_global_loop_cache_idx = 0
        for layer_idx, _ in layer_order:
            layer_shares_global_loop_kv = (
                global_loop_share_kv_cache is not None and _get_layer_window_size(self.config, layer_idx) is None
            )
            if layer_shares_global_loop_kv and layer_idx in global_loop_cache_idx_by_layer:
                cache_idx = global_loop_cache_idx_by_layer[layer_idx]
            else:
                cache_idx = next_global_loop_cache_idx
                next_global_loop_cache_idx += 1
                if layer_shares_global_loop_kv:
                    global_loop_cache_idx_by_layer[layer_idx] = cache_idx
            global_loop_cache_idx_by_execution.append(cache_idx)
        if getattr(self.config, "enable_depth_attention", False):
            if (
                getattr(self.config, "combine_mla_depth_attention", False)
                or getattr(self.config, "linear_depth_attention_isolation", False)
                or getattr(self.config, "window_depth_attention_isolation", False)
                or getattr(self.config, "window_no_depth_attention", False)
            ):
                depth_attention_kv_cache = _DepthAttentionCacheState()
                for decoder_layer in self.layers:
                    base_group = (
                        "linear" if isinstance(decoder_layer.self_attn, NanbeigeKimiDeltaAttention) else "attention"
                    )
                    group = _get_depth_attention_cache_group(self.config, decoder_layer.layer_idx, base_group)
                    depth_attention_kv_cache.register_layer(decoder_layer.layer_idx, group)
            else:
                depth_attention_kv_cache = []
        else:
            depth_attention_kv_cache = None
        if depth_attention_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("enable_depth_attention does not support gradient checkpointing during training.")

        for loop_idx in range(num_loops):
            current_loop_all_hidden_states = () if output_hidden_states else None
            current_loop_all_self_attns = () if output_attentions else None
            current_loop_next_decoder_cache = None

            for execution_idx, (layer_idx, loop_repeat_idx) in enumerate(layer_order):
                decoder_layer = layer_lookup[layer_idx]
                if output_hidden_states:
                    current_loop_all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    cache_layer_idx = (
                        (
                            layer_idx
                            if loop_share_kv_cache is not None
                            else (
                                global_loop_cache_idx_by_execution[execution_idx]
                                if global_loop_share_kv_cache is not None
                                else execution_idx
                            )
                        )
                        if double_loop_split
                        else None
                    )
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        loop_idx,
                        cache_layer_idx,
                        loop_repeat_idx,
                        loop_share_kv_cache,
                        global_loop_share_kv_cache,
                        depth_attention_kv_cache,
                        linear_attention_mask,
                    )
                else:
                    cache_layer_idx = (
                        (
                            layer_idx
                            if loop_share_kv_cache is not None
                            else (
                                global_loop_cache_idx_by_execution[execution_idx]
                                if global_loop_share_kv_cache is not None
                                else execution_idx
                            )
                        )
                        if double_loop_split
                        else None
                    )
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        loop_idx=loop_idx,
                        loop_cache_layer_idx=cache_layer_idx,
                        loop_repeat_idx=loop_repeat_idx,
                        loop_share_kv_cache=loop_share_kv_cache,
                        global_loop_share_kv_cache=global_loop_share_kv_cache,
                        depth_attention_kv_cache=depth_attention_kv_cache,
                        linear_attention_mask=linear_attention_mask,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    current_loop_next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    current_loop_all_self_attns += (layer_outputs[1],)

            if not getattr(self.config, "skip_loop_final_norm", False):
                hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                current_loop_all_hidden_states += (hidden_states,)

            last_loop_all_hidden_states = current_loop_all_hidden_states
            last_loop_all_self_attns = current_loop_all_self_attns
            last_loop_next_decoder_cache = current_loop_next_decoder_cache

        if getattr(self.config, "skip_loop_final_norm", False):
            hidden_states = self.norm(hidden_states)
            if output_hidden_states and last_loop_all_hidden_states is not None:
                last_loop_all_hidden_states = last_loop_all_hidden_states[:-1] + (hidden_states,)

        if use_cache and isinstance(past_key_values, NanbeigeLinearAttentionCache):
            past_key_values.seen_tokens += inputs_embeds.shape[1]
        next_cache = last_loop_next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, last_loop_all_hidden_states, last_loop_all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=last_loop_all_hidden_states,
            attentions=last_loop_all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flex_attention":
            return attention_mask

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = self._get_cache_seq_length(past_key_values)
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class NanbeigeForCausalLM(NanbeigePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config, use_cache=None):
        # ``from_pretrained(..., use_cache=False)`` forwards this option to the
        # model constructor in Transformers.  Consume it here and keep the
        # canonical value on the config used by NanbeigeModel.forward.
        if use_cache is not None:
            config.use_cache = use_cache
        super().__init__(config)
        self.model = NanbeigeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(NANBEIGE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_generation: bool = False,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer

        >>> model = AutoModelForCausalLM.from_pretrained("path/to/nanbeige-checkpoint", trust_remote_code=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/nanbeige-checkpoint", trust_remote_code=True)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            is_generation=is_generation,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, *args, **kwargs):
        """Select loop-aware and recurrent caches for autoregressive generation."""
        kwargs["is_generation"] = True
        is_linear_attention_model = (
            getattr(self.config, "experimental_attention_variant", None) == "kimi_delta_attention"
        )
        if is_linear_attention_model:
            generation_config = kwargs.get("generation_config", args[1] if len(args) > 1 else None)
            cache_implementation = kwargs.get(
                "cache_implementation", getattr(generation_config, "cache_implementation", None)
            )
            if cache_implementation is not None:
                raise ValueError(
                    "KDA generation only supports the default NanbeigeLinearAttentionCache; "
                    "cache_implementation is not supported."
                )
            kwargs["use_cache"] = True
            if kwargs.get("past_key_values") is None:
                kwargs["past_key_values"] = NanbeigeLinearAttentionCache(self.config)
        if getattr(self.config, "enable_depth_attention", False):
            if getattr(self.config, "loop_share_kv", False):
                raise ValueError("enable_depth_attention with loop_share_kv does not support generation.")
            generation_config = kwargs.get("generation_config", args[1] if len(args) > 1 else None)
            cache_implementation = kwargs.get(
                "cache_implementation",
                getattr(generation_config, "cache_implementation", None),
            )
            if cache_implementation is not None:
                raise ValueError(
                    "enable_depth_attention generation only supports the default DynamicCache; "
                    "cache_implementation is not supported."
                )
            kwargs["use_cache"] = True
        if not is_linear_attention_model and (
            self.config.num_loops > 1
            or getattr(self.config, "enable_double_loop_split", False)
            or getattr(self.config, "enable_double_loop_split_loopie", False)
        ):
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = DynamicCache()
        elif getattr(self.config, "enable_depth_attention", False):
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = DynamicCache(config=self.config)
        return super().generate(*args, **kwargs)

    def _get_cache_seq_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, Cache):
            return past_key_values[0][0].shape[-2] if len(past_key_values) > 0 else 0
        if getattr(self.config, "enable_double_loop_split", False) or getattr(
            self.config, "enable_double_loop_split_loopie", False
        ):
            return past_key_values.get_seq_length(0)
        max_seq_length = 0
        loop_weights = getattr(self.config, "loop_loss_weights", [])
        num_loops = (
            len(loop_weights) + 1 if loop_weights is not None and len(loop_weights) > 0 else self.config.num_loops
        )
        for loop_idx in range(num_loops):
            layer_idx = loop_idx * self.config.num_hidden_layers
            max_seq_length = max(max_seq_length, past_key_values.get_seq_length(layer_idx))
        return max_seq_length

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        is_generation=False,
        **kwargs,
    ):
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            is_generation=is_generation,
            **kwargs,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if not isinstance(past_key_values, Cache):
            raise TypeError("Transformers 5 beam search requires a Cache instance.")
        past_key_values.reorder_cache(beam_idx)
        return past_key_values


__all__ = [
    "NanbeigeForCausalLM",
    "NanbeigeModel",
    "NanbeigePreTrainedModel",
]

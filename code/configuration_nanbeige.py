"""Nanbeige model configuration"""

import warnings
from dataclasses import fields

from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _parse_linear_depth_logit_mul(value):
    if value is None or value == "sqrt":
        return value
    if isinstance(value, bool):
        raise ValueError("linear_depth_logit_mul must be a positive number or 'sqrt'.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("linear_depth_logit_mul must be a positive number or 'sqrt'.") from exc
    if value <= 0:
        raise ValueError("linear_depth_logit_mul must be greater than 0.")
    return value


# NBG-specific options are explicit constructor arguments. Only HF metadata and
# shared runtime options may pass through **kwargs to PretrainedConfig.
_HF_CONFIG_KWARGS = {field.name for field in fields(PretrainedConfig)} | {
    "_auto_class",
    "_commit_hash",
    "_name_or_path",
    "_attn_implementation_internal",
    "_attn_implementation_was_explicit",
    "_experts_implementation_internal",
    "architectures",
    "attn_implementation",
    "auto_map",
    "experts_implementation",
    "finetuning_task",
    "model_type",
    "name_or_path",
    "num_labels",
    "output_attentions",
    "per_layer_config",
    "prefix",
    "quantization_config",
    "task_specific_params",
    "tokenizer_class",
    "torch_dtype",
} | GenerationConfig._get_default_generation_params().keys()


class NanbeigeConfig(PretrainedConfig):
    """Configuration for Nanbeige causal decoders with Loop, GQA/SWA, KDA and MLA depth attention."""

    model_type = "nanbeige"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        attention_bias=False,
        qkv_bias=None,
        attention_dropout=0.0,
        mlp_bias=False,
        qk_layernorm=False,
        qk_pos_emb_norm=False,
        v_norm_without_scale=False,
        rotary_percent=1.0,
        global_layer_rotary_percent=None,
        global_rope_theta=None,
        window_rope_theta=None,
        window_size=None,
        window_attn_skip_freq=None,
        window_layer_sink=False,
        window_layer_sink_only_when_truncated=False,
        global_layer_sink=False,
        global_layer_kvequal=False,
        num_loops=1,
        loop_loss_weights=None,
        skip_loop_final_norm=False,
        enable_double_loop_split=False,
        double_loop_all_layers=False,
        loop_style="model",
        enable_double_loop_split_loopie=False,
        loop_middle_layers=None,
        loop_share_kv=False,
        global_loop_share_kv_method=None,
        loop_cycles_layernorm=False,
        loop_cycles_qknorm=False,
        global_loop_independent_q=False,
        enable_depth_attention=False,
        depth_attention_stride=None,
        depth_attention_recent_window=0,
        depth_attention_static_anchor_once=True,
        linear_depth_attention_isolation=False,
        global_no_depth_attention=False,
        linear_no_depth_attention=False,
        window_no_depth_attention=False,
        window_depth_attention_isolation=False,
        depth_attention_qk_nope=False,
        linear_depth_logit_mul=None,
        linear_depth_logit_mul_learnable=False,
        linear_depth_logit_scale=False,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_gate_low_rank_dim=None,
        kda_use_full_rank_output_gate=False,
        kda_use_safe_gate=False,
        kda_gate_lower_bound=-5.0,
        kda_bias=False,
        kda_conv_bias=False,
        multi_latent_attention=False,
        combine_mla_depth_attention=False,
        combine_window_mla_depth_attention=False,
        kv_share_latent=False,
        mla_kpos_from_mix_ck=False,
        depth_attention_mla_q="recover",
        depth_attention_mla_k="recover",
        depth_attention_mla_q_rank=None,
        depth_attention_mla_k_rank=None,
        depth_attention_mla_group_mode="kv",
        depth_attention_mla_num_kv_group=1,
        depth_attention_mla_num_q_head=None,
        depth_attention_mla_num_v_group=1,
        depth_attention_mla_num_qk_head=None,
        k_lora_rank=None,
        v_lora_rank=None,
        mla_depth_attention_mode="latent",
        attention_output_gate=False,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        mla_rope_type="yarn",
        rotary_scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0,
        **kwargs,
    ):
        # Older NBG checkpoints serialize these optimization flags. The runtime
        # now always uses ordinary tensor operations; do not retain the flags.
        kwargs.pop("depth_attention_kernel", None)
        kwargs.pop("depth_attention_kernel_low_cpu", None)
        unsupported = kwargs.keys() - _HF_CONFIG_KWARGS
        if unsupported:
            raise ValueError(f"Unsupported Nanbeige configuration fields: {', '.join(sorted(unsupported))}")
        if experimental_attention_variant not in (None, "kimi_delta_attention"):
            raise ValueError("Nanbeige supports only kimi_delta_attention as a linear attention variant.")
        if experimental_attention_variant is not None and (
            linear_attention_freq is None or isinstance(linear_attention_freq, int) and linear_attention_freq <= 0
        ):
            raise ValueError("KDA requires a positive linear_attention_freq or a per-layer list.")
        kwargs.pop("model_type", None)
        kwargs["architectures"] = ["NanbeigeForCausalLM"]
        if "tokenizer_class" in kwargs:
            kwargs["tokenizer_class"] = "NanbeigeTokenizer"
        # Preserve whether the caller explicitly selected an attention backend;
        # this lets supported compact MLA models choose FlexAttention by default
        # without overriding an explicit eager/SDPA request.
        self._attn_implementation_was_explicit = "attn_implementation" in kwargs or bool(
            kwargs.get("_attn_implementation_was_explicit", False)
        )
        kwargs.pop("_attn_implementation_was_explicit", None)
        linear_depth_logit_mul = _parse_linear_depth_logit_mul(linear_depth_logit_mul)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        if rope_parameters is not None and rope_scaling is not None:
            raise ValueError("Specify only one of rope_parameters or the legacy rope_scaling field.")
        if rope_parameters is None:
            self._rope_scaling_validation(rope_scaling)
            rope_parameters = dict(rope_scaling) if rope_scaling is not None else {}
        else:
            rope_parameters = dict(rope_parameters)
        rope_type = rope_parameters.pop("type", rope_parameters.get("rope_type", "default"))
        rope_parameters["rope_type"] = rope_type
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope_parameters["rope_theta"]
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.qkv_bias = attention_bias if qkv_bias is None else qkv_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.qk_layernorm = qk_layernorm
        self.qk_pos_emb_norm = qk_pos_emb_norm
        self.v_norm_without_scale = v_norm_without_scale
        if not (0.0 <= rotary_percent <= 1.0 if multi_latent_attention else 0.0 < rotary_percent <= 1.0):
            raise ValueError("rotary_percent must be in the interval (0, 1].")
        if global_layer_rotary_percent is not None and not 0.0 <= global_layer_rotary_percent <= 1.0:
            raise ValueError("global_layer_rotary_percent must be in the interval [0, 1].")
        self.rotary_percent = rotary_percent
        self.global_layer_rotary_percent = global_layer_rotary_percent
        for name, value in (
            ("global_rope_theta", global_rope_theta),
            ("window_rope_theta", window_rope_theta),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        self.global_rope_theta = global_rope_theta
        self.window_rope_theta = window_rope_theta
        self.window_size = window_size
        self.window_attn_skip_freq = window_attn_skip_freq
        self.window_layer_sink = window_layer_sink
        self.window_layer_sink_only_when_truncated = window_layer_sink_only_when_truncated
        if self.window_layer_sink_only_when_truncated:
            if not self.window_layer_sink:
                raise ValueError("window_layer_sink_only_when_truncated requires window_layer_sink=True")
            if self.window_size is None or self.window_size[0] < 0:
                raise ValueError("window_layer_sink_only_when_truncated requires a finite left window_size")
        self.global_layer_sink = global_layer_sink
        self.global_layer_kvequal = global_layer_kvequal
        self.num_loops = num_loops
        self.loop_loss_weights = loop_loss_weights if loop_loss_weights is not None else []
        self.skip_loop_final_norm = skip_loop_final_norm
        self.enable_double_loop_split = enable_double_loop_split
        self.double_loop_all_layers = double_loop_all_layers
        self.loop_style = loop_style
        self.enable_double_loop_split_loopie = enable_double_loop_split_loopie
        self.loop_middle_layers = loop_middle_layers
        self.loop_share_kv = loop_share_kv
        self.global_loop_share_kv_method = global_loop_share_kv_method
        self.loop_cycles_layernorm = loop_cycles_layernorm
        self.loop_cycles_qknorm = loop_cycles_qknorm
        self.global_loop_independent_q = global_loop_independent_q
        self.enable_depth_attention = enable_depth_attention
        self.depth_attention_stride = depth_attention_stride
        self.depth_attention_recent_window = depth_attention_recent_window
        self.depth_attention_static_anchor_once = depth_attention_static_anchor_once
        self.linear_depth_attention_isolation = linear_depth_attention_isolation
        self.global_no_depth_attention = global_no_depth_attention
        self.linear_no_depth_attention = linear_no_depth_attention
        self.window_no_depth_attention = window_no_depth_attention
        self.window_depth_attention_isolation = window_depth_attention_isolation
        self.depth_attention_qk_nope = depth_attention_qk_nope
        self.linear_depth_logit_mul = linear_depth_logit_mul
        self.linear_depth_logit_mul_learnable = linear_depth_logit_mul_learnable
        self.linear_depth_logit_scale = linear_depth_logit_scale
        self.experimental_attention_variant = experimental_attention_variant
        self.linear_attention_freq = linear_attention_freq
        if isinstance(self.linear_attention_freq, list) and len(self.linear_attention_freq) != self.num_hidden_layers:
            raise ValueError(
                "linear_attention_freq list length must equal num_hidden_layers: "
                f"got {len(self.linear_attention_freq)} for {self.num_hidden_layers} layers."
            )
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_gate_low_rank_dim = linear_gate_low_rank_dim
        self.kda_use_full_rank_output_gate = kda_use_full_rank_output_gate
        self.kda_use_safe_gate = kda_use_safe_gate
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_bias = kda_bias
        self.kda_conv_bias = kda_conv_bias
        self.combine_mla_depth_attention = combine_mla_depth_attention
        self.combine_window_mla_depth_attention = combine_window_mla_depth_attention
        self.kv_share_latent = kv_share_latent
        self.mla_kpos_from_mix_ck = mla_kpos_from_mix_ck
        self.k_lora_rank = k_lora_rank
        self.v_lora_rank = v_lora_rank
        self.mla_depth_attention_mode = mla_depth_attention_mode
        self.depth_attention_mla_q = depth_attention_mla_q
        self.depth_attention_mla_k = depth_attention_mla_k
        self.depth_attention_mla_q_rank = depth_attention_mla_q_rank
        self.depth_attention_mla_k_rank = depth_attention_mla_k_rank
        self.depth_attention_mla_group_mode = depth_attention_mla_group_mode
        self.depth_attention_mla_num_kv_group = depth_attention_mla_num_kv_group
        self.depth_attention_mla_num_q_head = depth_attention_mla_num_q_head
        self.depth_attention_mla_num_v_group = depth_attention_mla_num_v_group
        self.depth_attention_mla_num_qk_head = depth_attention_mla_num_qk_head
        if combine_mla_depth_attention and mla_depth_attention_mode not in ("latent", "expanded"):
            raise ValueError("mla_depth_attention_mode must be latent or expanded.")
        if mla_kpos_from_mix_ck and not combine_mla_depth_attention:
            raise ValueError("mla_kpos_from_mix_ck requires combine_mla_depth_attention.")
        if mla_kpos_from_mix_ck and not kv_share_latent:
            raise ValueError("mla_kpos_from_mix_ck requires kv_share_latent.")
        self.multi_latent_attention = multi_latent_attention
        self.attention_output_gate = attention_output_gate
        self.q_lora_rank = q_lora_rank
        if kv_lora_rank is None and not kv_share_latent:
            kv_lora_rank = 512
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim
        from .configuration_mla_depth import resolve_mla_depth_config

        resolve_mla_depth_config(self)
        self.mla_rope_type = mla_rope_type
        self.rotary_scaling_factor = rotary_scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        if multi_latent_attention:
            if mla_rope_type not in ("rope", "yarn"):
                raise ValueError("MLA supports mla_rope_type='rope' or 'yarn'.")
            if min(kv_lora_rank, qk_head_dim, v_head_dim) <= 0 or qk_pos_emb_head_dim < 0:
                raise ValueError("MLA content dimensions must be positive; positional dimension may be zero.")
            if qk_pos_emb_head_dim == 0 and mla_rope_type != "rope":
                raise ValueError("MLA without a K positional branch requires mla_rope_type='rope'.")
            if q_lora_rank is not None and q_lora_rank <= 0:
                raise ValueError("q_lora_rank must be positive or None.")
            if qk_pos_emb_head_dim % 2 or rotary_scaling_factor <= 0:
                raise ValueError("MLA needs an even positional head dimension and positive scaling factor.")
            if enable_depth_attention and not combine_mla_depth_attention and qk_pos_emb_head_dim:
                raise ValueError("MLA depth attention with RoPE requires combine_mla_depth_attention.")
            if (
                loop_share_kv
                or global_loop_share_kv_method not in (None, "none")
                or global_loop_independent_q
                or loop_cycles_qknorm
            ):
                raise ValueError("HF MLA does not support loop-specific Q/KV or QK norms.")
            if global_layer_kvequal:
                raise ValueError("HF MLA does not support global_layer_kvequal.")
        self._hybrid_swa_validation()
        self._loop_depth_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self, rope_scaling):
        """
        Validate the `rope_scaling` configuration.
        """
        if rope_scaling is None:
            return

        if not isinstance(rope_scaling, dict) or len(rope_scaling) != 2:
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {rope_scaling}"
            )
        rope_scaling_type = rope_scaling.get("type", None)
        rope_scaling_factor = rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _hybrid_swa_validation(self):
        if self.window_size is None:
            if self.window_attn_skip_freq is not None:
                raise ValueError("window_attn_skip_freq requires window_size to be set.")
            return

        if not isinstance(self.window_size, (list, tuple)) or len(self.window_size) != 2:
            raise ValueError("window_size must contain exactly two integers: (left, right).")
        if any(not isinstance(size, int) or size < -1 for size in self.window_size):
            raise ValueError("window_size values must be integers greater than or equal to -1.")

        if self.window_attn_skip_freq is None:
            return
        if isinstance(self.window_attn_skip_freq, int):
            if self.window_attn_skip_freq <= 0:
                raise ValueError("window_attn_skip_freq must be greater than 0.")
            return
        if not isinstance(self.window_attn_skip_freq, list):
            raise ValueError("window_attn_skip_freq must be an integer or a list.")
        if len(self.window_attn_skip_freq) != self.num_hidden_layers:
            raise ValueError("A window_attn_skip_freq list must have one entry per hidden layer.")

    def _loop_depth_validation(self):
        if self.loop_style not in ("model", "layer"):
            raise ValueError("loop_style must be one of: model, layer.")
        if self.enable_double_loop_split_loopie:
            warnings.warn(
                "enable_double_loop_split_loopie is deprecated; use "
                "enable_double_loop_split with loop_style='layer' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.loop_style = "layer"
        double_loop_split = self.enable_double_loop_split or self.enable_double_loop_split_loopie
        if self.double_loop_all_layers and not self.enable_double_loop_split:
            raise ValueError("double_loop_all_layers requires enable_double_loop_split=True.")
        if self.loop_share_kv and not double_loop_split:
            raise ValueError("loop_share_kv requires enable_double_loop_split=True.")
        if self.global_loop_share_kv_method not in (None, "none", "kv", "k", "v"):
            raise ValueError("global_loop_share_kv_method must be one of: none, kv, k, v.")
        has_loop = double_loop_split or self.num_loops > 1 or bool(self.loop_loss_weights)
        if self.global_loop_share_kv_method not in (None, "none") and not has_loop:
            raise ValueError("global_loop_share_kv_method requires a model loop or enable_double_loop_split=True.")
        if self.loop_cycles_layernorm and not double_loop_split:
            raise ValueError("loop_cycles_layernorm requires enable_double_loop_split=True.")
        if self.loop_cycles_qknorm and not double_loop_split:
            raise ValueError("loop_cycles_qknorm requires enable_double_loop_split=True.")
        if self.loop_cycles_qknorm and not self.qk_layernorm:
            raise ValueError("loop_cycles_qknorm requires qk_layernorm=True.")
        if self.global_loop_independent_q and not self.enable_double_loop_split:
            raise ValueError("global_loop_independent_q requires enable_double_loop_split=True.")
        if self.linear_depth_logit_mul_learnable and not self.enable_depth_attention:
            raise ValueError("linear_depth_logit_mul_learnable requires enable_depth_attention.")
        if self.linear_depth_logit_mul_learnable and self.linear_depth_logit_mul is None:
            raise ValueError("linear_depth_logit_mul_learnable requires linear_depth_logit_mul.")
        if self.linear_depth_logit_scale and not self.enable_depth_attention:
            raise ValueError("linear_depth_logit_scale requires enable_depth_attention.")
        if self.depth_attention_qk_nope and not self.enable_depth_attention:
            raise ValueError("depth_attention_qk_nope requires enable_depth_attention.")
        if self.window_no_depth_attention and not self.enable_depth_attention:
            raise ValueError("window_no_depth_attention requires enable_depth_attention.")
        if (self.global_no_depth_attention or self.linear_no_depth_attention) and not self.enable_depth_attention:
            raise ValueError("global_no_depth_attention and linear_no_depth_attention require enable_depth_attention.")
        if (
            self.global_no_depth_attention or self.linear_no_depth_attention
        ) and not self.linear_depth_attention_isolation:
            raise ValueError(
                "global_no_depth_attention and linear_no_depth_attention require linear_depth_attention_isolation."
            )
        if self.enable_depth_attention:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_depth_attention requires num_hidden_layers to be greater than 0.")
            if self.depth_attention_stride is None:
                self.depth_attention_stride = self.num_hidden_layers // 2
            if self.depth_attention_stride <= 0:
                raise ValueError("depth_attention_stride must be greater than 0.")
            if self.depth_attention_recent_window < 0:
                raise ValueError("depth_attention_recent_window must be >= 0.")
            if self.depth_attention_recent_window != 0:
                raise ValueError("anchor-only Depth-Attention requires depth_attention_recent_window == 0.")
            if double_loop_split and not self.depth_attention_static_anchor_once:
                raise ValueError(
                    "enable_depth_attention with double-loop split requires depth_attention_static_anchor_once=True."
                )
            if self.experimental_attention_variant == "kimi_delta_attention":
                pattern = self.linear_attention_freq
                has_linear = (isinstance(pattern, int) and pattern > 1) or (isinstance(pattern, list) and any(pattern))
                if (
                    has_linear
                    and not self.linear_depth_attention_isolation
                    and self.linear_num_key_heads != self.num_key_value_heads
                ):
                    raise ValueError(
                        "GQA + linear Depth-Attention requires linear_num_key_heads "
                        f"({self.linear_num_key_heads}) to equal num_key_value_heads "
                        f"({self.num_key_value_heads})."
                    )
                if (
                    has_linear
                    and not self.linear_depth_attention_isolation
                    and self.linear_key_head_dim != self.head_dim
                ):
                    raise ValueError(
                        "GQA + linear Depth-Attention requires linear_key_head_dim "
                        f"({self.linear_key_head_dim}) to equal head_dim ({self.head_dim})."
                    )
                if (
                    has_linear
                    and not self.linear_depth_attention_isolation
                    and self.linear_value_head_dim != self.head_dim
                ):
                    raise ValueError(
                        "GQA + linear Depth-Attention requires linear_value_head_dim "
                        f"({self.linear_value_head_dim}) to equal head_dim ({self.head_dim})."
                    )
        if not double_loop_split and self.loop_loss_weights is not None and sum(self.loop_loss_weights) > 1.0:
            raise ValueError("sum(loop_loss_weights) must be <= 1.0.")
        if double_loop_split and self.double_loop_all_layers:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
            self.loop_middle_layers = self.num_hidden_layers
        elif double_loop_split and self.loop_middle_layers is None:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
            if self.num_hidden_layers % 2 != 0:
                raise ValueError(
                    "enable_double_loop_split requires num_hidden_layers to be divisible by 2 "
                    "when loop_middle_layers is not set."
                )
            self.loop_middle_layers = self.num_hidden_layers // 2
        if self.loop_middle_layers is not None:
            if self.num_hidden_layers <= 0:
                raise ValueError("loop_middle_layers requires num_hidden_layers to be greater than 0.")
            if self.loop_middle_layers <= 0:
                raise ValueError("loop_middle_layers must be greater than 0.")
            if self.num_hidden_layers % self.loop_middle_layers != 0:
                raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")


__all__ = ["NanbeigeConfig"]

"""Validation shared in meaning by Megatron and exported HF depth MLA configs."""


def get_mla_depth_head_counts(config):
    """Return resolved Q heads, K heads and V groups for the active grouping mode."""
    if getattr(config, "depth_attention_mla_group_mode", "kv") == "qk":
        heads = config.depth_attention_mla_num_qk_head
        return heads, heads, config.depth_attention_mla_num_v_group
    return (
        config.depth_attention_mla_num_q_head,
        config.depth_attention_mla_num_kv_group,
        config.depth_attention_mla_num_kv_group,
    )


def resolve_mla_depth_config(config):
    """Resolve dimensions in place, without changing non-combined MLA defaults."""
    if getattr(config, "combine_window_mla_depth_attention", False) and not config.combine_mla_depth_attention:
        raise ValueError("combine_window_mla_depth_attention requires combine_mla_depth_attention.")
    shared = getattr(config, "kv_share_latent", False)
    if shared and not config.combine_mla_depth_attention:
        raise ValueError("kv_share_latent requires combine_mla_depth_attention.")
    if not config.combine_mla_depth_attention:
        return
    if not config.multi_latent_attention or not config.enable_depth_attention:
        raise ValueError("combine_mla_depth_attention requires MLA and enable_depth_attention.")

    def positive(name):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be explicitly specified as a positive integer.")
        return value

    if shared:
        rank = positive("kv_lora_rank")
        for name in ("k_lora_rank", "v_lora_rank"):
            value = getattr(config, name)
            # Resolved checkpoint/HF configs contain the derived ranks. Accept
            # those on reload; raw CLI rejects independently supplied ranks.
            if value is not None and (type(value) is not int or value != rank):
                raise ValueError(f"{name} must equal kv_lora_rank with kv_share_latent.")
            setattr(config, name, rank)
    else:
        positive("k_lora_rank")
        positive("v_lora_rank")
    if config.q_lora_rank is not None:
        positive("q_lora_rank")
    group_mode = config.depth_attention_mla_group_mode
    if group_mode not in ("kv", "qk"):
        raise ValueError("depth_attention_mla_group_mode must be kv or qk.")
    group_name, head_name = (
        ("depth_attention_mla_num_v_group", "depth_attention_mla_num_qk_head")
        if group_mode == "qk"
        else ("depth_attention_mla_num_kv_group", "depth_attention_mla_num_q_head")
    )
    groups = positive(group_name)
    if getattr(config, head_name) is None:
        setattr(config, head_name, groups)
    heads = positive(head_name)
    for kind, choices in (
        ("q", ("recover", "fix", "independent", "latent")),
        ("k", ("recover", "independent", "latent")),
    ):
        mode = getattr(config, f"depth_attention_mla_{kind}")
        if mode not in choices:
            raise ValueError(f"depth_attention_mla_{kind} must be one of {choices}.")
        name = f"depth_attention_mla_{kind}_rank"
        rank = getattr(config, name)
        if kind == "q" and mode in ("latent", "recover"):
            positive("q_lora_rank")
        if mode in ("latent", "recover"):
            expected = (
                getattr(config, f"{kind}_lora_rank")
                if mode == "latent"
                else config.num_attention_heads * config.qk_head_dim
            )
            if rank is not None and (type(rank) is not int or rank != expected):
                raise ValueError(f"{name} must equal derived rank {expected} for {mode}.")
            setattr(config, name, expected)
        positive(name)
    rq, rk = config.depth_attention_mla_q_rank, config.depth_attention_mla_k_rank
    if group_mode == "qk":
        if rq != rk:
            raise ValueError("depth MLA qk requires depth_attention_mla_q_rank == depth_attention_mla_k_rank.")
        if heads % groups or rq % heads or rk % heads or config.v_lora_rank % groups:
            raise ValueError("depth MLA qk requires H % G = Rq % H = Rk % H = v_lora_rank % G = 0.")
    else:
        if heads % groups or rq % heads or rk % groups or config.v_lora_rank % groups:
            raise ValueError("depth MLA requires Hq % G = Rq % Hq = Rk % G = v_lora_rank % G = 0.")
        if rq // heads != rk // groups:
            raise ValueError("depth MLA requires equal head dimensions: Rq / Hq == Rk / G.")
    if config.qk_layernorm and config.q_lora_rank is None and not shared:
        raise ValueError("combine_mla_depth_attention with qk_layernorm requires q_lora_rank.")

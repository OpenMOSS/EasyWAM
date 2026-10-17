from ...component.action_dit import standard_initialization_payload


def _cosmos_source_key(target_key: str) -> str | None:
    parts = target_key.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None
    layer = parts[1]
    suffix = ".".join(parts[2:])
    mapping = {
        "self_attn.q.weight": "self_attn.q_proj.weight",
        "self_attn.k.weight": "self_attn.k_proj.weight",
        "self_attn.v.weight": "self_attn.v_proj.weight",
        "self_attn.o.weight": "self_attn.output_proj.weight",
        "self_attn.norm_q.weight": "self_attn.q_norm.weight",
        "self_attn.norm_k.weight": "self_attn.k_norm.weight",
        "cross_attn.q.weight": "cross_attn.q_proj.weight",
        "cross_attn.k.weight": "cross_attn.k_proj.weight",
        "cross_attn.v.weight": "cross_attn.v_proj.weight",
        "cross_attn.o.weight": "cross_attn.output_proj.weight",
        "cross_attn.norm_q.weight": "cross_attn.q_norm.weight",
        "cross_attn.norm_k.weight": "cross_attn.k_norm.weight",
        "ffn.0.weight": "mlp.layer1.weight",
        "ffn.2.weight": "mlp.layer2.weight",
    }
    source_suffix = mapping.get(suffix)
    return None if source_suffix is None else f"blocks.{layer}.{source_suffix}"


def initialize_action(action, visual, *, alpha=True):
    if action.preset != "standard":
        raise ValueError(
            "Cosmos visual initialization requires ActionDiT preset=standard"
        )
    return standard_initialization_payload(
        action,
        visual,
        source_key=_cosmos_source_key,
        missing_value=lambda key, target: target,
        source_backbone="cosmos25",
        mapping_version=2,
        alpha=alpha,
        repeat_head_norm=True,
    )

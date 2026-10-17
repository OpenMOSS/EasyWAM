"""Map double-stream and single-stream visual weights into staged ActionDiT."""

import torch

from ...component.action_dit import resize_weight


def initialize_from_flux(action, visual, alpha=True):
    if action.preset != "staged":
        raise ValueError(
            "FLUX visual initialization requires ActionDiT preset=staged; standard cross-topology mapping is not defined"
        )
    source = visual.transformer.state_dict()
    output = {}
    report = {category: [] for category in ("copied", "resized", "initialized")}
    records = {}
    source_hidden = visual.hidden_dim
    source_attention = visual.num_heads * visual.attn_head_dim

    def transform(value, target, key, source_key, sections=None, axis=0):
        if sections:
            source_sizes, target_sizes = sections
            values = value.split(source_sizes, dim=axis)
            shapes = []
            for size in target_sizes:
                shape = list(target.shape)
                shape[axis] = size
                shapes.append(tuple(shape))
            result = torch.cat(
                [resize_weight(v, shape, alpha) for v, shape in zip(values, shapes)],
                dim=axis,
            )
        elif value.ndim == 1 and target.numel() % value.numel() == 0 and "norm_" in key:
            result = value.repeat(target.numel() // value.numel())
        else:
            result = resize_weight(value, tuple(target.shape), alpha)
        category = "copied" if tuple(value.shape) == tuple(target.shape) else "resized"
        report[category].append(key)
        records[key] = dict(
            source=source_key,
            source_shape=list(value.shape),
            target_shape=list(target.shape),
            method=category,
            numel=target.numel(),
        )
        return result.to(dtype=target.dtype, device="cpu")

    for key, target in action.state_dict().items():
        if key.startswith(action.ACTION_BACKBONE_SKIP_PREFIXES):
            continue
        source_key = key
        sections = None
        axis = 0
        value = None
        # Newly introduced condition paths have no corresponding visual layer.
        if key.startswith("text_embedding.") or ".cross_attn." in key:
            source_key = None
        if action.options["time_embedding_type"] != "scaled" and key.startswith(
            ("time_in.", "double_stream_modulation_img.", "single_stream_modulation.")
        ):
            source_key = None
        if ".img_mlp." in key and action.options["ffn_activation"] != "swiglu":
            source_key = None
        if key.startswith("double_blocks.") and source_key:
            if ".img_attn." in key:
                for part, index in [("q", 0), ("k", 1), ("v", 2)]:
                    marker = f".img_attn.{part}."
                    if marker in key:
                        source_key = key.replace(marker, ".img_attn.qkv.")
                        if source_key in source:
                            value = source[source_key].chunk(3, dim=0)[index]
                if ".img_attn.qkv." in key:
                    sections = (
                        [source_attention] * 3,
                        [action.num_heads * action.attn_head_dim] * 3,
                    )
                if ".norm_q." in key or ".norm_k." in key:
                    source_key = key.replace("norm_q.", "norm.query_norm.").replace(
                        "norm_k.", "norm.key_norm."
                    )
            if ".img_mlp.0." in key:
                if source_key in source:
                    sections = (
                        [source[source_key].shape[0] // 2] * 2,
                        [target.shape[0] // 2] * 2,
                    )
        if key.startswith("single_blocks.") and source_key:
            base = ".".join(key.split(".")[:2])
            source_width = (
                source[f"{base}.linear2.weight"].shape[1] - source_attention
                if f"{base}.linear2.weight" in source
                else None
            )
            if source_width is not None:
                action_attention = action.num_heads * action.attn_head_dim
                target_width = action.blocks[
                    action.double_layers + int(key.split(".")[1])
                ].mlp_hidden_dim
                if ".linear1." in key:
                    if action.options["ffn_activation"] == "swiglu":
                        sections = (
                            [source_attention] * 3 + [source_width] * 2,
                            [action_attention] * 3 + [target_width] * 2,
                        )
                    else:
                        source_key = None
                elif ".linear2." in key and key.endswith("weight"):
                    if action.options["ffn_activation"] == "swiglu":
                        sections = (
                            [source_attention, source_width],
                            [action_attention, target_width],
                        )
                        axis = 1
                    else:
                        source_key = None
                else:
                    for part, index in [("q", 0), ("k", 1), ("v", 2), ("mlp_in", 3)]:
                        if key.startswith(base + "." + part + "."):
                            source_key = key.replace("." + part + ".", ".linear1.")
                            if source_key in source:
                                pieces = source[source_key].split(
                                    [source_attention] * 3 + [source_width * 2], dim=0
                                )
                                value = pieces[index]
                                if part == "mlp_in":
                                    if action.options["ffn_activation"] == "swiglu":
                                        sections = (
                                            [source_width] * 2,
                                            [target_width] * 2,
                                        )
                                    else:
                                        source_key = None
                                        value = None
                if ".norm_q." in key or ".norm_k." in key:
                    source_key = key.replace("norm_q.", "norm.query_norm.").replace(
                        "norm_k.", "norm.key_norm."
                    )
        if source_key:
            source_key = source_key.replace(
                "norm.query_norm.weight", "norm.query_norm.scale"
            ).replace("norm.key_norm.weight", "norm.key_norm.scale")
        if source_key and value is None:
            value = source.get(source_key)
        if source_key and value is not None:
            if key.endswith("modulation_img.lin.weight") or key.endswith(
                "modulation_img.lin.bias"
            ):
                sections = ([source_hidden] * 6, [action.hidden_dim] * 6)
            elif key.endswith("single_stream_modulation.lin.weight") or key.endswith(
                "single_stream_modulation.lin.bias"
            ):
                sections = ([source_hidden] * 3, [action.hidden_dim] * 3)
            output[key] = transform(value, target, key, source_key, sections, axis)
        else:
            output[key] = target.detach().cpu().clone()
            report["initialized"].append(key)
            records[key] = dict(
                source=None,
                target_shape=list(target.shape),
                method="initialized",
                numel=target.numel(),
            )
    total = sum(t.numel() for t in output.values())
    initialized = sum(records[k]["numel"] for k in report["initialized"])
    return dict(
        backbone_state_dict=output,
        action_config=action.resolved_config,
        policy=dict(source_backbone="flux2", mapping_version=1, alpha_scaling=alpha),
        initialization=dict(
            **report,
            tensors=records,
            mapped_numel=total - initialized,
            total_numel=total,
            coverage=(total - initialized) / max(total, 1),
        ),
    )

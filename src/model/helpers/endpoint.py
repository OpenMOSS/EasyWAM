"""Image endpoint objectives behind the common policy training/inference API.

Architecture masks are independent of the visual backbone's block implementation.
No simulator, dataset conversion, or pretrained policy compatibility lives here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from ..component.attention import run_attention
from .batching import randn_per_sample, validate_batch_inputs
from .gradient import gradient_checkpoint_forward


def enabled(model):
    return (
        getattr(getattr(model, "video_dit", None), "visual_target", None) == "endpoint"
    )


def variant(model):
    return model.model_variant


def _map_vae(model, images, operation):
    size = getattr(model.vae, "micro_batch_size", None)
    if size is None:
        return getattr(model.vae, operation)(images)
    return torch.cat(
        [getattr(model.vae, operation)(chunk) for chunk in images.split(size)], dim=0
    )


def encode_video(model, video):
    if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1:
        raise ValueError("Expected normalized video [B,3,T,H,W]")
    if video.shape[-2] % 16 or video.shape[-1] % 16:
        raise ValueError("Endpoint images must have height/width divisible by 16")
    frames = video if video.shape[2] == 1 else video[:, :, [0, -1]]
    batch, channels, steps, height, width = frames.shape
    images = frames.transpose(1, 2).reshape(batch * steps, channels, height, width)
    latent = _map_vae(model, images.to(model.device, model.torch_dtype), "encode")
    return latent.reshape(batch, steps, *latent.shape[1:]).transpose(1, 2)


def decode_latents(model, latent):
    batch, channels, steps, height, width = latent.shape
    decoded = _map_vae(
        model,
        latent.transpose(1, 2).reshape(batch * steps, channels, height, width),
        "decode",
    )
    decoded = decoded.reshape(batch, steps, *decoded.shape[1:])
    pixels = ((decoded.detach().float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
    return [
        [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in video]
        for video in pixels
    ]


def schedulers(model, training=True):
    if variant(model) == "unified":
        return model.scheduler, model.scheduler
    prefix = "train" if training else "infer"
    return getattr(model, prefix + "_video_scheduler"), getattr(
        model, prefix + "_action_scheduler"
    )


def _state(model, state, batch):
    if getattr(model, "state_dim", None) is None:
        if state is not None:
            raise ValueError("State supplied to a policy with state_dim disabled")
        return None
    if state is None:
        raise ValueError("Endpoint policy requires proprioception")
    if state.ndim == 3:
        state = state[:, 0]
    if state.shape != (batch, model.state_dim):
        raise ValueError(
            f"Expected proprio [B,{model.state_dim}], got {tuple(state.shape)}"
        )
    return state.to(model.device, model.torch_dtype)


def _context(model, context, mask, prompt=None):
    if context is None:
        if prompt is None:
            raise ValueError("Provide context/context_mask or prompt")
        context, mask = model.encode_prompt(prompt)
    if mask is None or context.ndim != 3 or mask.shape != context.shape[:2]:
        raise ValueError("Context must be [B,L,D] with a matching [B,L] mask")
    return context.to(model.device, model.torch_dtype), mask.to(
        model.device, torch.bool
    )


def _scene(model, target, reference, context, mask, timestep, grid):
    expert = model.video_dit
    batch = target.shape[0]
    ids = expert.build_img_ids(
        batch, *grid, time_value=0, device=target.device, dtype=target.dtype
    )
    ref_ids = expert.build_img_ids(
        batch, *grid, time_value=10, device=target.device, dtype=target.dtype
    )
    if target.shape[1] == 0:
        ids = ids[:, :0]
    return expert.pre_dit(
        x=target,
        timestep=timestep,
        context=context,
        context_mask=mask,
        ref_image_hidden_states=reference,
        target_img_ids=ids,
        ref_img_ids=ref_ids,
    )


def architecture_mask(pres, action_len, kind):
    """No text-mediated leakage: each teacher/prediction scene has its own text."""
    lengths = [p["txt_len"] + p["tokens"]["img"].shape[1] for p in pres]
    batch, device = pres[0]["text_mask"].shape[0], pres[0]["text_mask"].device
    total = sum(lengths) + action_len
    mask = torch.zeros(batch, total, total, device=device, dtype=torch.bool)
    offset = 0
    for pre, length in zip(pres, lengths):
        prefix = pre["txt_len"] + pre["cond_len"]
        mask[:, offset : offset + prefix, offset : offset + prefix] = True
        mask[:, offset + prefix : offset + length, offset : offset + length] = True
        offset += length
    if action_len:
        action_start = sum(lengths)
        mask[:, action_start:, action_start:] = True
        if kind in {"mot_joint", "unified"}:
            mask[:, action_start:, :action_start] = True
            prefix = pres[0]["txt_len"] + pres[0]["cond_len"]
            mask[:, prefix:action_start, action_start:] = True
        elif kind == "mot_idm":
            start = sum(lengths[:-1])
            mask[:, action_start:, start:action_start] = True
        else:
            prefix = pres[0]["txt_len"] + pres[0]["cond_len"]
            mask[:, action_start:, :prefix] = True
    offset = 0
    for pre, length in zip(pres, lengths):
        mask[:, :, offset : offset + pre["txt_len"]] &= pre["text_mask"][:, None, :]
        if pre.get("target_valid") is not None:
            start = offset + pre["txt_len"] + pre["cond_len"]
            mask[:, :, start : start + pre["target_len"]] &= pre["target_valid"][
                :, None, None
            ]
        offset += length
    return mask


def _mixed(model, pres, action_pre, kind):
    return model.mot.forward_prepared(
        pres, action_pre, architecture_mask(pres, action_pre["tokens"].shape[1], kind)
    )


def _visual(model, pre, capture=None, return_prediction=True):
    tokens = pre["tokens"]
    mask = architecture_mask([pre], 0, "mot")
    captured = None
    for index in range(len(model.video_dit.blocks)):
        tokens = gradient_checkpoint_forward(
            lambda values, i=index: model.video_dit.forward_block(i, values, pre, mask),
            model.video_dit.use_gradient_checkpointing and torch.is_grad_enabled(),
            tokens,
        )
        if index == capture:
            captured = model.video_dit.visual_features(tokens)
            if not return_prediction:
                break
    return (
        model.video_dit.post_dit(tokens, pre) if return_prediction else None
    ), captured


def predict(
    model,
    target,
    reference,
    context,
    mask,
    state,
    action,
    tv,
    ta,
    grid,
    condition=None,
    tc=None,
    hidden=None,
    visual_valid=None,
    return_visual=True,
    hidden_cache=None,
):
    kind = variant(model)
    vs, acs = schedulers(model)
    tv = tv / float(vs.num_train_timesteps)
    ta = ta / float(acs.num_train_timesteps)
    if kind.startswith("mot") and state is not None:
        context, mask = model._append_state_to_context(context, mask, state)
    pre = _scene(model, target, reference, context, mask, tv, grid)
    pre["target_valid"] = visual_valid
    if action is None:
        return _visual(model, pre)[0], None, None
    if kind == "hidden":
        if hidden is None:
            pred, features = _visual(model, pre, model.video_hidden_layer)
        else:
            pred = _visual(model, pre)[0] if return_visual else None
            features = hidden
        if model.detach_video_hidden:
            features = features.detach()
        condition = None if hidden_cache is None else hidden_cache.get("condition")
        if condition is None:
            projected = model.action_dit.project_context(
                model.dit["video_context_projector"](features)
            )
            feature_mask = torch.ones(
                projected.shape[:2], device=projected.device, dtype=torch.bool
            )
            if visual_valid is not None:
                feature_mask[:, pre["cond_len"] :] &= visual_valid[:, None]
            cross_kv = (
                model.action_dit.build_cross_attention_kv_cache(projected)
                if hidden_cache is not None
                else None
            )
            condition = model._prepare_action_condition(
                state[:, None], projected, feature_mask, cross_kv
            )
            if hidden_cache is not None:
                hidden_cache["condition"] = condition
        act = model.action_dit.pre_dit(
            action,
            ta,
            condition["context"],
            condition["context_mask"],
            context_is_projected=True,
            cross_kv_cache=condition["cross_kv_cache"],
        )
        if model.state_position == "sequence":
            act = model.action_dit.append_state_tokens(act, condition["state_tokens"])
        return (
            pred,
            model.action_dit.forward_prepared(act),
            features,
        )
    if kind == "unified":
        act = model.dit["action_encoder"](action)
        st = model.dit["state_encoder"](state[:, None])
        pre = model.video_dit.pre_unified_dit(pre, act, st, ta, model.state_position)
        tokens = pre["tokens"]
        # Unified action tokens live in the shared image stream; construct its
        # mask with the same prefix/future split as the joint MoT architecture.
        shared_mask = architecture_mask([pre], 0, "unified")
        for index in range(len(model.video_dit.blocks)):
            tokens = gradient_checkpoint_forward(
                lambda x, i=index: model.video_dit.forward_block(
                    i, x, pre, shared_mask
                ),
                model.video_dit.use_gradient_checkpointing and torch.is_grad_enabled(),
                tokens,
            )
        outputs = model.video_dit.post_unified_dit(tokens, pre)
        return (
            outputs["video"],
            model.dit["action_decoder"](outputs["action_tokens"]),
            None,
        )
    act = model.action_expert.pre_dit(action, ta, context, mask)
    act = model._append_state_to_action_sequence(act, state)
    pres = [pre]
    if condition is not None:
        cond = _scene(
            model,
            condition,
            reference,
            context,
            mask,
            tc / float(vs.num_train_timesteps),
            grid,
        )
        cond["target_valid"] = visual_valid
        pres.append(cond)
    pred, pred_action = _mixed(model, pres, act, kind)
    return pred, pred_action, None


def _loss(error, valid=None):
    if valid is None:
        return error.flatten(1).mean(1)
    valid = valid.to(error.dtype)
    return (error * valid).flatten(1).sum(1) / valid.expand_as(error).flatten(1).sum(
        1
    ).clamp(min=1)


def training_loss(model, sample):
    video = sample["video"].to(model.device, model.torch_dtype)
    with torch.no_grad():
        latent = encode_video(model, video)
    if latent.shape[2] != 2:
        raise ValueError("Endpoint training needs current and future frames")
    reference = latent[:, :, 0].flatten(2).transpose(1, 2)
    target = latent[:, :, 1].flatten(2).transpose(1, 2)
    grid = latent.shape[-2:]
    action = sample["action"].to(model.device, model.torch_dtype)
    state = _state(model, sample.get("proprio"), action.shape[0])
    context, mask = _context(
        model, sample.get("context"), sample.get("context_mask"), sample.get("prompt")
    )
    vs, acs = schedulers(model)
    tv = vs.sample_training_t(action.shape[0], model.device, action.dtype)
    ta = (
        tv
        if variant(model) == "unified"
        else acs.sample_training_t(action.shape[0], model.device, action.dtype)
    )
    nv, na = torch.randn_like(target), torch.randn_like(action)
    noisy_v, noisy_a = vs.add_noise(target, nv, tv), acs.add_noise(action, na, ta)
    condition = tc = None
    if variant(model) == "mot_idm":
        use_noise = (
            torch.rand(action.shape[0], device=model.device)
            < model.video_cond_noise_prob
        )
        tc = (
            vs.sample_training_t(action.shape[0], model.device, action.dtype)
            * use_noise
        )
        condition = vs.add_noise(target, torch.randn_like(target), tc)
    visual_valid = (
        None
        if sample.get("image_is_pad") is None
        else (~sample["image_is_pad"][:, -1]).to(model.device)
    )
    pv, pa, _ = model._forward_endpoint(
        noisy_v,
        reference,
        context,
        mask,
        state,
        noisy_a,
        tv,
        ta,
        grid,
        condition,
        tc,
        visual_valid=visual_valid,
    )
    image_valid = None if visual_valid is None else visual_valid[:, None, None]
    av = torch.ones_like(action, dtype=torch.bool)
    if sample.get("action_is_pad") is not None:
        av &= ~sample["action_is_pad"].to(model.device)[:, :, None]
    if sample.get("action_dim_is_pad") is not None:
        av &= ~sample["action_dim_is_pad"].to(model.device)[:, None, :]
    lv = (
        _loss((pv.float() - (nv - target).float()).square(), image_valid)
        * vs.training_weight(tv)
    ).mean()
    la = (
        _loss((pa.float() - (na - action).float()).square(), av)
        * acs.training_weight(ta)
    ).mean()
    return model.loss_lambda_video * lv + model.loss_lambda_action * la, dict(
        loss_video=lv.detach(),
        loss_action=la.detach(),
        weighted_loss_video=lv.detach() * model.loss_lambda_video,
        weighted_loss_action=la.detach() * model.loss_lambda_action,
    )


def infer(
    model,
    *,
    input_image,
    action_horizon,
    prompt=None,
    context=None,
    context_mask=None,
    proprio=None,
    num_inference_steps=20,
    seed=None,
    rand_device="cpu",
    sigma_shift=None,
    action_only=False,
    decode_video=True,
    text_cfg_scale=1.0,
    negative_prompt=None,
    **kwargs,
):
    validate_batch_inputs(input_image, prompt, context, context_mask)
    if text_cfg_scale != 1.0 or negative_prompt not in (None, ""):
        raise ValueError(
            "Endpoint policies currently support text_cfg_scale=1 without negative_prompt"
        )
    if action_horizon is None or action_horizon <= 0:
        raise ValueError("Positive action_horizon is required")
    model.eval()
    batch = input_image.shape[0]
    state = _state(model, proprio, batch)
    context, context_mask = _context(model, context, context_mask, prompt)
    latent = encode_video(model, input_image[:, :, None])
    reference = latent[:, :, 0].flatten(2).transpose(1, 2)
    grid = latent.shape[-2:]
    target = randn_per_sample(
        tuple(reference.shape[1:]),
        seeds=seed,
        batch_size=batch,
        rand_device=rand_device,
    ).to(model.device, model.torch_dtype)
    action_dim = (
        model.action_dim
        if hasattr(model, "action_dim")
        else model.action_expert.action_dim
    )
    action = randn_per_sample(
        (action_horizon, action_dim),
        seeds=seed,
        batch_size=batch,
        rand_device=rand_device,
    ).to(model.device, model.torch_dtype)
    vs, acs = schedulers(model, False)
    vt, vd = vs.build_inference_schedule(
        num_inference_steps, model.device, model.torch_dtype, shift_override=sigma_shift
    )
    at, ad = acs.build_inference_schedule(
        num_inference_steps, model.device, model.torch_dtype, shift_override=sigma_shift
    )
    kind = variant(model)
    hidden = None
    if kind == "mot_idm":
        for t, d in zip(vt, vd):
            pv, _, _ = model._forward_endpoint(
                target,
                reference,
                context,
                context_mask,
                state,
                None,
                t.expand(batch),
                t.expand(batch),
                grid,
            )
            target = vs.step(pv, d, target, stream_id="video")
    if action_only and kind == "hidden":
        pre = _scene(
            model,
            target,
            reference,
            context,
            context_mask,
            vt[0].expand(batch) / vs.num_train_timesteps,
            grid,
        )
        _, hidden = _visual(
            model, pre, model.video_hidden_layer, return_prediction=False
        )
    # A causal MoT prefix is invariant across denoising steps. Cache only when
    # that dependency is guaranteed; joint/teacher/hidden paths remain explicit.
    cache = None
    if kind == "mot" and action_only and model.inference_cross_kv_reuse:
        cache = _prefix_cache(model, reference, context, context_mask, state, grid)
    hidden_cache = {} if kind == "hidden" and model.inference_cross_kv_reuse else None
    for tv, dv, ta, da in zip(vt, vd, at, ad):
        if cache is not None:
            pa = model._forward_endpoint_cached_action(
                action, state, context, context_mask, ta.expand(batch), cache
            )
            pv = None
        elif kind == "mot_idm":
            # Teacher-conditioning at inference uses the generated endpoint,
            # never ground-truth future observations.
            empty = target[:, :0]
            _, pa, _ = model._forward_endpoint(
                empty,
                reference,
                context,
                context_mask,
                state,
                action,
                torch.zeros(batch, device=model.device),
                ta.expand(batch),
                grid,
                target,
                torch.zeros(batch, device=model.device),
            )
            pv = None
        else:
            pv, pa, features = model._forward_endpoint(
                target,
                reference,
                context,
                context_mask,
                state,
                action,
                tv.expand(batch),
                ta.expand(batch),
                grid,
                hidden=hidden,
                return_visual=not action_only,
                hidden_cache=hidden_cache,
            )
            if kind == "hidden" and hidden is None:
                hidden = features
        if pv is not None and not (action_only and kind in {"mot", "hidden"}):
            target = vs.step(pv, dv, target, stream_id="video")
        action = acs.step(pa, da, action, stream_id="action")
    result = {"action": action.detach().float().cpu()}
    if not action_only and decode_video:
        last = target.transpose(1, 2).reshape(batch, latent.shape[1], *grid)
        result["video"] = decode_latents(model, torch.stack([latent[:, :, 0], last], 2))
    return result


def _prefix_cache(model, reference, context, mask, state, grid):
    if state is not None:
        context, mask = model._append_state_to_context(context, mask, state)
    pre = _scene(
        model,
        reference[:, :0],
        reference,
        context,
        mask,
        reference.new_zeros(reference.shape[0]),
        grid,
    )
    tokens = pre["tokens"]
    attention_mask = architecture_mask([pre], 0, "mot")
    cache = []
    for i in range(len(model.video_dit.blocks)):
        q, k, v, s = model.video_dit.prepare_attention(i, tokens, pre)
        cache.append((k, v))
        mixed = run_attention(
            q,
            k,
            v,
            num_heads=model.video_dit.num_heads,
            attention_mask=attention_mask,
            backend=model.video_dit.attention_backend,
        )
        tokens = model.video_dit.finish_attention(i, mixed, s)
    return cache, pre["text_mask"]


def _cached_action(model, values, state, context, mask, timestep, cache):
    action = model.action_expert
    vs, acs = schedulers(model, False)
    if state is not None:
        context, mask = model._append_state_to_context(context, mask, state)
    pre = action.pre_dit(values, timestep / acs.num_train_timesteps, context, mask)
    pre = model._append_state_to_action_sequence(pre, state)
    pe = action.prepare_position(pre, model.video_dit.transformer.pe_embedder)
    values = pre["tokens"]
    layers, textmask = cache
    for i, block in enumerate(action.blocks):
        mod = (
            pre["t_mod"]["double_img"]
            if i < action.double_layers
            else pre["t_mod"]["single"]
        )
        s = block.prepare_qkv(values, pe, mod)
        pk, pv = layers[i]
        keys = torch.cat([pk, s["k"]], 1)
        vals = torch.cat([pv, s["v"]], 1)
        valid = torch.ones(keys.shape[:2], device=keys.device, dtype=torch.bool)
        valid[:, : textmask.shape[1]] = textmask
        mixed = run_attention(
            s["q"],
            keys,
            vals,
            num_heads=action.num_heads,
            attention_mask=valid[:, None, :].expand(-1, values.shape[1], -1),
            backend=action.attention_backend,
        )
        values = block.apply_post(mixed, s)
        if action.cross_attention:
            values = values + block.cross_attn(
                values, pre["context"], ctx_mask=pre["context_mask"]
            )
    return action.post_dit(values, pre)


def structure(model):
    action = getattr(model, "action_expert", getattr(model, "action_dit", None))
    visual = model.video_dit
    return dict(
        version=1,
        variant=variant(model),
        state_position=model.state_position,
        state_dim=model.state_dim,
        visual_dim=visual.hidden_dim,
        heads=visual.num_heads,
        double_layers=visual.double_layers,
        single_layers=visual.single_layers,
        action=None if action is None else action.resolved_config,
        hidden_layer=getattr(model, "video_hidden_layer", None),
        detach_hidden=getattr(model, "detach_video_hidden", None),
    )


def validate_structure(model, payload):
    if payload.get("structure") != structure(model):
        raise ValueError(
            "Endpoint checkpoint structure differs from the configured policy"
        )

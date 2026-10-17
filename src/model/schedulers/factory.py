from .scheduler_continuous import ContinuousFlowMatchScheduler
from .scheduler_flow_unipc import FlowUniPCScheduler


def build_video_schedulers(
    *,
    family,
    config=None,
    num_train_timesteps=1000,
    train_shift=5.0,
    infer_shift=5.0,
    shared=False,
):
    config = dict(config or {})
    family = str(config.get("family", family))
    if family == "continuous_flow":
        train = ContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=train_shift
        )
        infer = (
            train
            if shared
            else ContinuousFlowMatchScheduler(
                num_train_timesteps=num_train_timesteps, shift=infer_shift
            )
        )
    elif family == "flow_unipc":
        train = FlowUniPCScheduler(
            num_train_timesteps,
            float(config.get("train_shift", train_shift)),
            use_karras_sigmas=(
                bool(config.get("use_karras_sigmas", True)) if shared else False
            ),
            time_distribution=str(config.get("time_distribution", "logitnormal")),
            training_weight_method=str(config.get("training_weight", "uniform")),
        )
        infer = (
            train
            if shared
            else FlowUniPCScheduler(
                num_train_timesteps,
                float(config.get("infer_shift", infer_shift)),
                use_karras_sigmas=bool(config.get("use_karras_sigmas", True)),
            )
        )
    else:
        raise ValueError(f"Unsupported scheduler family: {family!r}")
    return train, infer

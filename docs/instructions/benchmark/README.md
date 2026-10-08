# Benchmark Guides

[中文](README_zh.md) | [Documentation index](../../README.md) | [Back to project README](../../../README.md)

This section covers simulator preparation, checkpoint evaluation, dynamic inference batching, result output, and resume behavior. Training-data preparation is kept in the [data guides](../data/README.md).

Published metrics across backbones are collected in the [benchmark results](../../results/result.md) ([中文](../../results/result_zh.md)).

| Benchmark | Evaluation scope | Guide |
| --- | --- | --- |
| LIBERO | Standard four-suite evaluation | [Setup and evaluation](libero.md) |
| LIBERO-Pro | Official five-dimension generalization evaluation | [Setup and evaluation](libero_pro.md) |
| LIBERO-Plus | Robustness evaluation with LIBERO checkpoints | [Setup and evaluation](libero_plus.md) |
| RoboTwin | Clean and randomized evaluation | [Setup and evaluation](robotwin.md) |
| RoboDojo | Official 42-task simulation evaluation | [Setup and evaluation](robodojo.md) |
| RoboCasa365 | Official 50-task Human300 evaluation | [Setup and evaluation](robocasa.md) |

## Inference tensor contracts

`infer_action`, `infer_joint`, and `infer` accept one normalized floating-point RGB image `[3,H,W]` or `[1,3,H,W]`. These single-sample entrypoints normalize input dimensions once and return CPU float32 actions `[A,Da]`; decoded video is `list[PIL.Image]`. Cached text may be `[L,Dc]`/`[L]` or include a batch dimension of size one.

`infer_action_batch` and `infer_joint_batch` require image `[B,3,H,W]`, context `[B,L,Dc]`, bool context_mask `[B,L]`, and proprio `[B,Ds]` when enabled. Online prompts must be a sequence of `B` strings. Hidden and Unified also accept explicit state sequences `[B,S,Ds]` and retain their existing use of the first state for inference. Batch size one still requires the batch dimension; a two-dimensional state sequence is not inferred from `B=1`.

Batch actions are CPU float32 `[B,A,Da]`; decoded videos are `list[list[PIL.Image]]`, including when `B=1`. For APIs supporting `decode_video=false`, disabling video decoding omits the video key. Single-request queues store images `[3,H,W]` and states `[Ds]`, stack them for the model, then split the fixed batch result. Video VAE calls inside the models use five-dimensional batched tensors; standalone VAE list-input compatibility remains available.

Internal Wan head time embeddings use `[B,Q,D]`, with `Q=1` for a shared time condition. Cosmos time features receive `[B,T]` after entrypoint normalization. Video VAE scale parameters are two per-channel tensors `[z_dim]`; self-attention prepares RoPE factors once for both Q and K. `ContinuousFlowMatchScheduler.training_weight` preserves the input timestep shape, including `[1]` for a batch of size one.

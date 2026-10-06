# Training Data Guides

[中文](README_zh.md) | [Documentation index](../../README.md) | [Back to project README](../../../README.md)

This section covers dataset downloads, the expected LeRobot layout, text-embedding preprocessing, normalization statistics, and benchmark-specific training entrypoints.

| Dataset | Contents | Guide |
| --- | --- | --- |
| LIBERO | Hub download, dataset layout, statistics, text cache, and training | [Data and training](libero.md) |
| RoboTwin | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robotwin.md) |
| RoboDojo | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robodojo.md) |
| RoboCasa365 | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robocasa.md) |

Simulator setup and rollout evaluation are intentionally kept in the [benchmark guides](../benchmark/README.md).

## Data processing

Dataset construction reuses metadata, Arrow storage and normalization statistics across train/validation wrappers. Each sample reads its numeric windows once; shared camera timestamps and repeated padded frames are reused. Training keeps decoded images as uint8 until the processor converts them to float. RoboTwin and RoboDojo resize each camera once to its final tile size (top: 256×320, wrists: 128×160), then assemble the 384×320 video. This changes pixels relative to the former two-stage interpolation, while preserving the layout, output shape and augmentation parameters.

The processor defaults remain `stack_images=true` and `include_gt_action=true`. The camera-tile recipes set `stack_images=false` for differently sized cameras, and RobotVideoDataset omits unused GT action copies. Decoder callers retain float output by default; the training path requests `output_dtype=torch.uint8`. Existing text cache files and payloads remain compatible.

Text preprocessing defaults to skipping existing files and initializes the encoder only if there are missing prompts. Use `+overwrite=true` to regenerate caches. Normalization-statistics generation uses at most four workers and eight pending tasks, returning compact per-episode reductions instead of retaining all episode tensors.

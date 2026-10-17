# Backbone Guides

[中文](README_zh.md) | [Documentation index](../../README.md) | [Back to project README](../../../README.md)

Choose a backbone guide for model downloads, ActionDiT initialization, text features, training, and evaluation. Each guide follows the same workflow and describes its backbone's configuration and prediction behavior.

| Backbone | Supported architectures | English | 中文 |
| --- | --- | --- | --- |
| Wan2.2-TI2V-5B | Unified, MoT, MoT-Joint, MoT-IDM, Hidden | [Guide](wan22.md) | [指南](wan22_zh.md) |
| Cosmos-Predict2.5-2B | Unified, MoT, MoT-Joint, MoT-IDM, Hidden | [Guide](cosmos25.md) | [指南](cosmos25_zh.md) |
| FLUX.2 Klein Base 4B | Unified, MoT, MoT-Joint, MoT-IDM, Hidden | [Guide](flux2.md) | [指南](flux2_zh.md) |

The [model configuration guide](../config/models.md) explains architecture selection and shared settings. Dataset preparation is covered by the [data guides](../data/README.md); simulator setup and rollout options are in the [benchmark guides](../benchmark/README.md).

## Source layout

Each backbone package uses the same file roles; network implementations remain local to that package.

| File | Role |
| --- | --- |
| `__init__.py` | Public exports |
| `loader.py` | Component loading, text encoding, and ActionDiT initialization entry points |
| `action_initialization.py` | Visual-to-action weight conversion |
| `<backbone>_core.py` | Standalone visual training, inference, and checkpoints |
| `<prefix>_video_dit.py` | DiT network and policy adapter |
| `<prefix>_video_text_encoder.py` | Text encoder and tokenizer |
| `<prefix>_video_vae.py` | VAE wrapper and weight loading |
| `<prefix>_vae_arch.py` | VAE network |

Core modules expose Python APIs for visual-only training. The policy training commands above continue to train the selected EasyWAM architecture.

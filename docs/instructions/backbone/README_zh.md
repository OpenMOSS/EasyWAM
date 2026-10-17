# Backbone 文档索引

[English](README.md) | [文档总索引](../../README_zh.md) | [返回项目 README](../../../README_zh.md)

按所选 backbone 查阅模型下载、ActionDiT 初始化、文本特征、训练和评测方法。各指南采用相同的使用流程，并说明对应 backbone 的配置与预测行为。

| Backbone | 支持的架构 | English | 中文 |
| --- | --- | --- | --- |
| Wan2.2-TI2V-5B | Unified、MoT、MoT-Joint、MoT-IDM、Hidden | [Guide](wan22.md) | [指南](wan22_zh.md) |
| Cosmos-Predict2.5-2B | Unified、MoT、MoT-Joint、MoT-IDM、Hidden | [Guide](cosmos25.md) | [指南](cosmos25_zh.md) |
| FLUX.2 Klein Base 4B | Unified、MoT、MoT-Joint、MoT-IDM、Hidden | [Guide](flux2.md) | [指南](flux2_zh.md) |

架构选择和通用设置见[模型配置指南](../config/models_zh.md)。数据准备见[数据指南](../data/README_zh.md)，仿真环境与 rollout 选项见 [Benchmark 指南](../benchmark/README_zh.md)。

## 源码结构

每个 backbone 包采用相同的文件职责，网络实现在各自包内维护。

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 公开接口 |
| `loader.py` | 组件加载、文本编码和 ActionDiT 初始化入口 |
| `action_initialization.py` | 视觉到动作的权重转换 |
| `<backbone>_core.py` | 独立视觉训练、推理和检查点 |
| `<prefix>_video_dit.py` | DiT 网络和策略适配接口 |
| `<prefix>_video_text_encoder.py` | 文本编码器和 tokenizer |
| `<prefix>_video_vae.py` | VAE 封装和权重加载 |
| `<prefix>_vae_arch.py` | VAE 网络 |

Core 模块通过 Python API 支持独立视觉训练；各指南中的策略训练命令仍用于训练所选 EasyWAM 架构。

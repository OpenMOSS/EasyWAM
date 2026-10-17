# FLUX.2 Klein Base 4B Backbone

[English](flux2.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

FLUX.2 支持 Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。Backbone 由双流/单流 DiT、图像 VAE 和 Qwen3 文本编码器组成。默认路径和运行设置位于 `configs/model/backbone/flux2_klein_4b.yaml`。

## 准备模型权重

在项目根目录执行。下载受限文件前，需在 Hugging Face 上接受自编码器仓库的访问条件，并执行 `huggingface-cli login`：

```bash
mkdir -p checkpoints/flux2

huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
  --include "flux-2-klein-base-4b.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-klein-base-4B

huggingface-cli download black-forest-labs/FLUX.2-dev \
  --include "ae.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-dev

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir checkpoints/Qwen3-4B
```

文本编码器默认使用 `Qwen/Qwen3-4B`。若使用本地下载，在文本预计算和评测命令中加入 `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B`。Transformer 和 VAE 路径分别由 `model.backbone.model_path` 和 `model.backbone.ae_model_path` 配置。

## 初始化 ActionDiT

动作专家使用 `staged` 预设，宽度为 `1024`，包含 5 个顺序 block 和 20 个并行 block。Hidden 启用视觉特征交叉注意力，也支持独立配置动作层数。通用组件选项见[模型配置](../config/models_zh.md#配置-actiondit)。

`model.action_dit_pretrained_path=null` 表示随机初始化。使用视觉权重初始化 staged ActionDiT：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --backbone flux2 \
  --model-config configs/model/easywam_mot_flux2_klein_4b.yaml \
  --output checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt
```

训练时加入 `model.action_dit_pretrained_path=./checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt`。Hidden 使用 `configs/model/easywam_hidden_flux2_klein_4b.yaml` 并输出到独立文件。MoT 系列的默认动作结构共用初始化文件；Unified 没有独立 ActionDiT。

默认启用 alpha scaling（`--apply-alpha-scaling true`）：沿输入维度缩放的权重乘以 `sqrt(source_dim / target_dim)`。转换器迁移兼容的视觉参数，分别缩放 Q/K/V 与门控 FFN 的各部分。输出包含参数来源和覆盖率，新条件模块保留其初始化。转换支持 staged 预设，生成的文件必须与所选动作结构匹配。

## 预计算文本特征

[准备数据集](../data/README_zh.md)后，生成 Qwen3 特征缓存：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_flux2_klein_4b
```

其他数据集选择对应 task，再覆盖模型组。例如：

```bash
python scripts/precompute_text_embeds.py \
  task=vlabench_easywam_mot_wan22 \
  model=easywam_mot_flux2_klein_4b
```

Task 决定数据集和缓存位置。数据集、编码器设置与 `context_len` 一致时，五种架构可复用特征。缓存采用通用的 `<sha256>.text_len<context_len>.<encoder_id>[dtype].pt` 命名，包含 `context`、布尔类型的 `mask` 和缓存元数据。预计算和训练时保持缓存路径及文本设置一致。使用旧专用格式的缓存需通过上述命令重新生成。

## 训练

选择数据集 task 和模型配置：

```bash
# 全参数 MoT 训练
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_flux2_klein_4b

# Hidden 视觉 backbone LoRA 训练
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_flux2_klein_4b \
  model=easywam_hidden_flux2_klein_4b_lora
```

模型命名为 `easywam_<architecture>_flux2_klein_4b`，架构可选 `unified`、`mot`、`mot_joint`、`mot_idm` 或 `hidden`；LoRA 使用 `_lora` 后缀。模型配置默认设置 `lambda_video=0.5`、`lambda_action=1.0`。

### 预测目标与条件

训练使用通用的 `video`、`action`、`proprio`、`context` 和 `context_mask` 字段。视频的首帧与末帧分别作为当前观测和未来端点。视觉损失和评估使用未来端点；可视化输出包含当前图像与预测端点。

| 架构 | 动作条件与推理 |
| --- | --- |
| MoT | 动作读取文本、当前图像和状态；动作推理可缓存因果视觉前缀。 |
| MoT-Joint | 联合去噪端点和动作。 |
| MoT-IDM | 训练使用独立的 teacher 端点；推理先生成端点，再生成动作。 |
| Hidden | 读取视觉中间层特征。动作推理从当前图像与未来噪声中提取一次特征；联合推理继续去噪端点，动作条件复用初始特征。 |
| Unified | 在共享 Transformer 中联合去噪端点与动作/状态 token。 |

`video_hidden_layer` 按视觉两个阶段的连续 block 索引取值。Hidden 需要交叉注意力，可选择两种动作预设；从视觉权重转换动作权重需要 staged block。状态布局支持 `context` 和 `sequence`。推理使用 `text_cfg_scale=1`，不设置 negative prompt。策略加载会检查保存的架构和动作结构，不支持旧 ImageWAM 策略 checkpoint。

## 评测

使用与训练一致的 task、模型选择及数据集统计：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_flux2_klein_4b \
  model=easywam_hidden_flux2_klein_4b_lora \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json>
```

评测加载 Qwen3 文本编码器处理任务指令。仿真环境、rollout 选项和结果目录见 [Benchmark 指南](../benchmark/README_zh.md)。

## 独立视觉训练

`Flux2Core` 可独立训练视觉 DiT，无需动作专家。先完成上述权重、数据和文本缓存准备，再从项目根目录运行这个单步示例。它使用现有数据协议，动作和状态字段可省略，传入时会被忽略。VAE 和文本编码器保持冻结及评估模式。文本缓存优先于 prompt；没有缓存时，可在 Python 配置中启用 `backbone.load_text_encoder` 来编码 prompt。

```bash
PYTHONPATH=src:. python - <<'PYTHON'
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from model.backbone.flux2 import Flux2Core
from runtime import build_datasets
from utils import misc
from utils.config_resolvers import register_default_resolvers

register_default_resolvers()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=libero_easywam_mot_flux2_klein_4b"])
misc.register_work_dir("./runs/flux2_core")
dataset, _ = build_datasets(cfg.data)
sample = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

model = Flux2Core.from_flux2_pretrained(
    cfg.model.backbone, device="cuda", torch_dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.dit.parameters(), lr=1e-4)
model.train()
optimizer.zero_grad(set_to_none=True)
loss, metrics = model(sample)
loss.backward()
optimizer.step()
model.save_checkpoint("./runs/flux2_core/checkpoint.pt", optimizer=optimizer, step=1)
model.load_checkpoint("./runs/flux2_core/checkpoint.pt", optimizer=optimizer)

result = model.infer(
    sample["video"][0, :, 0],
    context=sample["context"][:1],
    context_mask=sample["context_mask"][:1],
    num_frames=2, num_inference_steps=20, seed=42,
)
result["video"][-1].save("./runs/flux2_core/endpoint.png")
PYTHON
```

训练输入 `video` 为 `[B,3,T,H,W]`，取值范围 `[-1,1]`，`T >= 2`，图像高宽能被 16 整除。训练使用首尾帧、`context`/`context_mask`（或 `prompt`），以及可选的 `[B,T]` `image_is_pad`；被标记为 padding 的目标帧不贡献损失。推理接收单张图像，返回当前图像和未来端点（`num_frames=2`），要求 `guidance_scale=1`，不使用负向条件。

Core 检查点保存 DiT 权重、backbone/variant 信息、步数和可选的优化器状态。恢复前使用相同的预训练 VAE 和文本配置创建 Core；这是视觉检查点，仿真评测使用 EasyWAM 策略检查点。

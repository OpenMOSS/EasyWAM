<p align="center">
  <img src="assets/icon.png" alt="EasyWAM Logo" width="20%">
</p>

<h1 align="center">🚀 EasyWAM: An Efficient and Easy-to-Use Codebase for World Action Model</h1>

<p align="center">用于训练与评测 World Action Model 的统一研究代码库。</p>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-111111.svg" alt="English README"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg" alt="中文 README"></a>
  <a href="https://openmoss.github.io/EasyWAM/"><img src="https://img.shields.io/badge/Website-EasyWAM-0969da.svg" alt="EasyWAM Website"></a>
  <a href="https://huggingface.co/collections/OpenMOSS-Team/easywam"><img src="https://img.shields.io/badge/HF%20Model-Checkpoints-FFD21E.svg?logo=huggingface&logoColor=000000" alt="Hugging Face Models"></a>
  <a href="#-联系我们"><img src="https://img.shields.io/badge/WeChat-Join%20Discussion%20Group-brightgreen?logo=wechat" alt="WeChat"></a>
</p>

**从这里开始：** [快速开始](#-快速开始) · [支持的模型](#-支持的模型与-benchmark) · [评测结果](#-benchmark-结果) · [模型下载](https://huggingface.co/collections/OpenMOSS-Team/easywam) · [文档](#-文档)

## ✨ 概览与主要特性

EasyWAM 是一个统一的 World Action Model 研究代码库，旨在让模型开发更加高效、可复现且易于扩展。项目通过一致的工作流连接模型实现、数据处理、分布式训练、参数高效微调和大规模评测。

<p align="center">
  <img src="assets/overview.png" alt="EasyWAM 框架概览" width="100%">
</p>

- ⚡ **高效计算：** 计算效率是 EasyWAM 的核心设计目标。项目原生集成 **FlashAttention 2/3/4** 以加速 Attention 计算，并提供覆盖参数高效训练、checkpoint 保存到合并推理的完整 **LoRA** 支持；BF16、gradient checkpointing、DeepSpeed ZeRO 和 PyTorch SDPA 回退进一步兼顾速度、显存占用与兼容性。
- 🚄 **端到端管线优化：** EasyWAM 对训练与推理的每个阶段进行系统优化。稀疏视频解码、索引式文本缓存、常驻 worker、prompt cache 和断点续评可以消除重复开销，从而显著提升 **训练和推理** 速度。
- 🧩 **统一且易扩展的设计：** 模块化模型组件共用一致的数据、训练、checkpoint 和评测接口，便于接入新的 WAM 设计。
- 🛠️ **简单易用的工作流：** Hydra 配置、标准化训练与评测方案、分布式 launcher、自动 GPU 分片和结果汇总让常用流程更加直接。

> 🌟 **愿景：** 我们希望 EasyWAM 成为一个高效且易用的 WAM 研究代码库，帮助研究者更快、更轻松地开展 World Action Model 研究。未来我们将持续更新并支持更多模型与 benchmark，也诚挚欢迎社区贡献者加入，共同让 EasyWAM 变得更好。如果在使用过程中遇到任何问题，或对 EasyWAM 有任何改进建议，欢迎提交 Issue；我们会持续完善 EasyWAM。

## 📰 最新动态

- **[2026-09-18]** EasyWAM 新增 RoboCasa365 和 RoboDojo 的训练与评测支持，将 RoboTwin 适配到新版上游并加入独立的动态批量推理；评测现支持单 GPU 多模型 worker 和 GPU 并发调优工具。训练新增可配置的 checkpoint 保留策略、动作之后的 state token 布局，以及通用的 FLUX 文本 embedding 预计算。本次还更新了 Unified、MoT、Hidden 在 Wan2.2 和 Cosmos2.5 下的 LIBERO、LIBERO-Plus 结果及 Wan2.2 LoRA 的 LIBERO 结果，并修订架构分析和评测默认配置。
- **[2026-09-12]** EasyWAM 新增动态批量评测，仅需少量 worker 即可执行评测，大幅降低测评显存占用并提升 GPU 利用率。本次更新还加入任务/trial 进度展示、LeRobot v3 支持、优化了文本 padding mask 语义的 FlashAttention 2/3/4、可配置的 state token 位置与因果注意力、执行与缓存优化、自动 run 日志、更完善的文档以及新的 Benchmark 结果。
- **[2026-09-03]** EasyWAM 新增 FLUX.2/ImageWAM backbone 集成。
- **[2026-09-02]** EasyWAM 正式发布，为 World Action Model 提供统一的训练与评测工作流。

## 🤖 支持的模型与 Benchmark

### 🧠 模型

- **EasyWAM-Unified：** 采用单主干架构，将视频和动作 token 输入同一个 Video DiT，联合预测未来视频与动作。该架构参考 [DreamZero](https://github.com/dreamzero0/dreamzero)。
- **EasyWAM-MoT：** 采用双主干架构，使用独立的 Video DiT 和 Action DiT 专家，并通过共享的混合自注意力实现两类 token 的交互。该模型仅进行动作预测，架构参考 [FastWAM](https://github.com/yuantianyuan01/FastWAM)。
- **EasyWAM-MoT-Joint：** 采用双主干架构，通过共享的混合自注意力联合去噪视频和动作 token。
- **EasyWAM-MoT-IDM：** 采用双主干架构，使用 teacher-forcing 条件视频进行动作预测。
- **EasyWAM-Hidden：** 采用双主干架构，将 Video DiT 的中间特征作为独立 Action DiT 的条件输入。该架构参考 [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT)。

| 模型 | 全参数训练 | LoRA 微调 |
| --- | :---: | :---: |
| EasyWAM-Unified | ✅ | ✅ |
| EasyWAM-MoT | ✅ | ✅ |
| EasyWAM-MoT-Joint | ✅ | ✅ |
| EasyWAM-MoT-IDM | ✅ | ✅ |
| EasyWAM-Hidden | ✅ | ✅ |

### 🏗️ Backbone

| Backbone | 支持情况 |
| --- | :---: |
| Wan2.2-TI2V-5B | ✅ |
| Cosmos-Predict2.5-2B | ✅ |
| FLUX.2 Klein-4B | ✅ |

### 🧪 Benchmark

| Benchmark | 支持情况 | 训练 | 评测 |
| --- | :---: | --- | --- |
| LIBERO | ✅ | 全参数训练和 LoRA | 标准四 suite 评测 |
| LIBERO-Plus | ✅ | 使用 LIBERO checkpoint | 鲁棒性评测 |
| RoboTwin | ✅ | 全参数训练和 LoRA | Clean 和 randomized 评测 |
| RoboDojo | ✅ | 全参数训练和 LoRA | 官方 42-task 仿真评测 |
| RoboCasa365 | ✅ | 全参数训练和 LoRA | 官方 50-task 评测 |

## 🏆 Benchmark 结果

> 以下结果均在 `state_position: sequence` 设置下报告，即将 state/proprio token 放在 action token 之后并加入模型序列。

<details open>
<summary><b>LIBERO</b></summary>

**全参数训练**

| Backbone | 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | EasyWAM-Unified | 98.4 | 98.8 | 99.2 | 98.0 | 98.6 |
| Wan2.2-TI2V-5B | EasyWAM-MoT | 97.0 | 99.2 | 96.6 | 94.0 | 96.7 |
| Wan2.2-TI2V-5B | EasyWAM-MoT-Joint | 98.2 | 98.0 | 97.6 | 96.8 | 97.7 |
| Wan2.2-TI2V-5B | EasyWAM-MoT-IDM | 99.0 | 99.2 | 98.8 | 97.4 | 98.6 |
| Wan2.2-TI2V-5B | EasyWAM-Hidden | 99.2 | 100.0 | 97.8 | 98.2 | 98.8 |
| Cosmos-Predict2.5-2B | EasyWAM-Unified | 97.8 | 99.4 | 97.0 | 93.0 | 96.8 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT | 98.0 | 98.4 | 98.4 | 95.6 | 97.6 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT-Joint | 98.6 | 99.8 | 98.4 | 96.0 | 98.2 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT-IDM | 99.4 | 99.4 | 99.8 | 98.4 | 99.3 |
| Cosmos-Predict2.5-2B | EasyWAM-Hidden | 97.2 | 98.8 | 96.0 | 95.0 | 96.8 |

**LoRA（Rank 128）**

| Backbone | 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | EasyWAM-Unified | 91.2 | 98.8 | 91.8 | 66.2 | 87.0 |
| Wan2.2-TI2V-5B | EasyWAM-MoT | 96.8 | 99.6 | 97.4 | 90.0 | 95.9 |
| Wan2.2-TI2V-5B | EasyWAM-Hidden | 98.0 | 99.8 | 89.4 | 86.6 | 93.5 |

</details>

<details open>
<summary><b>LIBERO-Plus</b></summary>

| Backbone | 模型 | Background | Camera | Language | Layout | Light | Noise | Robot | 平均 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | EasyWAM-Unified | 72.3 | 54.8 | 93.7 | 83.4 | 97.0 | 72.0 | 83.4 | 79.0 |
| Wan2.2-TI2V-5B | EasyWAM-MoT | 64.5 | 45.5 | 71.4 | 80.1 | 94.7 | 78.5 | 71.5 | 71.7 |
| Wan2.2-TI2V-5B | EasyWAM-MoT-Joint | 60.9 | 47.3 | 89.7 | 80.5 | 92.4 | 68.9 | 75.9 | 73.3 |
| Wan2.2-TI2V-5B | EasyWAM-MoT-IDM | 62.2 | 52.0 | 94.1 | 82.0 | 93.2 | 67.8 | 78.0 | 75.3 |
| Wan2.2-TI2V-5B | EasyWAM-Hidden | 59.3 | 57.0 | 93.2 | 84.3 | 95.0 | 70.8 | 83.7 | 77.6 |
| Cosmos-Predict2.5-2B | EasyWAM-Unified | 72.7 | 63.7 | 90.1 | 82.6 | 89.2 | 72.1 | 79.2 | 78.2 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT | 60.7 | 75.1 | 92.3 | 82.6 | 92.6 | 81.3 | 58.5 | 77.7 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT-Joint | 62.1 | 60.8 | 96.6 | 84.5 | 94.4 | 69.4 | 76.9 | 77.7 |
| Cosmos-Predict2.5-2B | EasyWAM-MoT-IDM | 66.4 | 59.8 | 93.8 | 85.7 | 89.0 | 72.3 | 81.9 | 78.4 |
| Cosmos-Predict2.5-2B | EasyWAM-Hidden | 59.5 | 65.9 | 92.6 | 85.0 | 89.1 | 68.3 | 82.5 | 77.8 |

</details>

更多结果请查阅[完整 Benchmark 结果](docs/results/result_zh.md)，结果分析请参阅[什么样的 WAM 架构是我们需要的？](docs/blogs/blog01_arch_zh.md)（[English](docs/blogs/blog01_arch.md)）。

## ⚡ 效率结果

> 以下所有效率结果均使用 **Wan2.2-TI2V-5B** 作为 backbone。**实际训练时长会随 CPU 和 GPU 配置而变化，以上数据仅供参考。** 请参阅[效率配置指南](docs/instructions/config/efficiency_zh.md)，根据自己的机器选择合适的配置，以提升训练与推理效率。

EasyWAM-MoT 与 FastWAM 使用相同的模型架构，因此可以在架构一致的条件下对比 EasyWAM 训练框架与 FastWAM 原始代码框架。在 **8 × NVIDIA H100 GPUs**、**per-device batch size 为 16** 的配置下，EasyWAM-MoT 的吞吐量达到 **138.2 samples/s**，是 FastWAM（51.5 samples/s）的 **2.68 倍**；每 step 的数据读取、前向传播和反向传播耗时分别降低 **85.8%**、**70.5%** 和 **49.5%**。

<p align="center">
  <img src="assets/efficiency_comparison.svg" alt="EasyWAM 与 FastWAM 训练效率对比" width="100%">
</p>

| 代码框架 | 吞吐量（samples/s）↑ | 数据读取耗时（s）↓ | 前向传播耗时（s）↓ | 反向传播耗时（s）↓ |
| --- | :---: | :---: | :---: | :---: |
| EasyWAM-MoT | **138.2** | **0.0108** | **0.3663** | **0.5392** |
| FastWAM | 51.5 | 0.0759 | 1.2412 | 1.0680 |

在 LIBERO 数据集上训练 **20,000 steps**，EasyWAM 约需 **5 小时**，而使用 FastWAM 原始代码框架约需 **14 小时**，整体训练时长缩短约 **64.3%**。

> 🌟 **愿景：** World Action Model 的训练与评测通常需要大量计算资源。EasyWAM 希望通过高效且轻量的代码框架降低这一研究门槛，让研究者即使在计算资源有限的情况下，也能完成模型训练与评测、快速验证想法并持续迭代。通过不断优化效率，我们希望研究者能够将更多资源投入模型与算法创新，也让更多社区成员参与到 WAM 研究中。

## 🚀 快速开始

### 🛠️ 安装环境

```bash
conda create -n easywam python=3.10 -y
conda activate easywam
pip install -U pip
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

[FlashAttention](https://github.com/Dao-AILab/flash-attention) 是可选依赖。安装后，EasyWAM 会使用当前环境中最快的兼容实现；无法使用时则回退到 PyTorch SDPA。

请根据 GPU 选择支持的实现：FA2 支持 Ampere、Ada 和 Hopper GPU；FA3 面向 Hopper GPU；FA4 面向 Hopper 和 Blackwell GPU。

```bash
# FlashAttention 2
pip install flash-attn --no-build-isolation

# FlashAttention 3
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install

# FlashAttention 4
pip install flash-attn-4

# FlashAttention 4 + CUDA 13
# pip install "flash-attn-4[cu13]"
```

### 📦 准备模型

已发布的 EasyWAM checkpoint 可从 Hugging Face 的 [OpenMOSS-Team/EasyWAM 模型集合](https://huggingface.co/collections/OpenMOSS-Team/easywam) 下载。

请按所选 backbone 的独立文档完成准备：

- [Wan2.2-TI2V-5B](docs/instructions/backbone/wan22.md)
- [Cosmos-Predict2.5-2B](docs/instructions/backbone/cosmos25.md)
- [FLUX.2 Klein Base 4B](docs/instructions/backbone/flux2.md)

### 📝 预计算文本特征

准备好 benchmark 数据后执行一次：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
# 或：python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
# 或：python scripts/precompute_text_embeds.py task=robocasa_easywam_mot_wan22
# 或：python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=libero_easywam_mot_cosmos25
```

### 🏋️ 训练

训练脚本直接接收 Hydra overrides，通过 `NPROC_PER_NODE` 设置本机进程数：

```bash
# 8 张本机 GPU，DeepSpeed ZeRO-1
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22

# MoT、MoT-Joint 和 MoT-IDM 均有对应的 Cosmos25 task 配置
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_cosmos25

# 4 张本机 GPU，DeepSpeed ZeRO-2 LoRA 训练
NPROC_PER_NODE=4 bash scripts/train_zero2.sh task=libero_easywam_unified_wan22_lora

# RoboDojo joint-only LeRobot v3 训练
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=robodojo_easywam_mot_wan22
```

`scripts/train_zero2_offload.sh` 启用 ZeRO-2 CPU Offload。多机训练还需设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。

### 📊 评测

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>

# LIBERO-Plus（使用 LIBERO checkpoint）
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>

# RoboDojo
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>

# RoboCasa365
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json>
```

评测并发由 `MULTIRUN.num_gpus`、`MULTIRUN.gpu_ids`、`MULTIRUN.workers_per_gpu` 和 `MULTIRUN.env_num_per_worker` 控制。`gpu_ids=null` 时使用前 `num_gpus` 张卡；否则 `gpu_ids` 是有序候选池，实际使用其中前 `num_gpus` 张。例如 `MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[2,3,5,6]'` 会使用 GPU 2、3、5。每个模型 worker 都会独立加载一份模型并持有自己的动态推理 batcher，因此提高 `workers_per_gpu` 也会增加显存占用；`MULTIRUN.inference_batch_size` 和 `MULTIRUN.inference_batch_wait_ms` 按 worker 生效，也可以使用 `scripts/tune_eval_concurrency.py` 在目标机器上扫描候选配置。数据目录、仿真环境安装、已发布权重、任务筛选和断点恢复等内容见对应的数据与 benchmark 指南。

## 📚 文档

| 分区 | English | 中文 |
| --- | --- | --- |
| Backbone 准备与使用 | [Index](docs/instructions/backbone/README.md) | [索引](docs/instructions/backbone/README_zh.md) |
| 训练数据准备 | [Index](docs/instructions/data/README.md) | [索引](docs/instructions/data/README_zh.md) |
| Benchmark 准备与评测 | [Index](docs/instructions/benchmark/README.md) | [索引](docs/instructions/benchmark/README_zh.md) |
| 配置参考 | [Index](docs/instructions/config/README.md) | [索引](docs/instructions/config/README_zh.md) |

## 🗂️ 项目结构

```text
EasyWAM/
├── configs/          # 数据、模型、task、训练及评测配置
├── docs/             # 项目文档
│   ├── README.md      # 文档索引
│   ├── blogs/        # 架构博客
│   ├── results/      # Benchmark 结果汇总
│   └── instructions/   # 数据、评测、Backbone 与配置指南
├── experiments/      # Benchmark 评测器
├── scripts/          # 训练、预处理和缓存入口
├── src/              # 模型、数据管线、runtime 和 trainer
├── checkpoints/      # 外部及训练得到的 checkpoint
├── data/             # 本地数据集和文本缓存
└── runs/             # 训练输出
```

## 🤝 参与贡献

EasyWAM 由社区共同建设，我们欢迎各种规模的贡献。你可以帮助修复 bug、完善文档与示例、支持新的模型或 Benchmark、优化训练与评测流程，或分享可复现的实验结果。

对于 bug 修复和较小的改进，可以直接提交 Issue 或 Pull Request。对于较大的功能，或可能影响公共接口、配置和兼容性的改动，请先创建 Issue 讨论范围与设计。Pull Request 应保持改动聚焦，提供相关验证，并在面向用户的行为发生变化时同步更新中英文文档。

问题反馈要求、开发约定和 Pull Request 检查项请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 🙏 致谢

本项目基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 构建，并参考了 [DreamZero](https://github.com/dreamzero0/dreamzero)、[DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) 和 [ImageWAM](https://github.com/yuyangalin/ImageWAM) 的模型设计与实现。我们由衷感谢以上团队对开源社区的宝贵贡献。

## 📝 Citation

我们欢迎在研究中引用 **EasyWAM** 的实验结果与代码库。如果 **EasyWAM** 对你的研究有所帮助，请引用：

```bibtex
@misc{easywam2026,
  title  = {EasyWAM: A Unified and Efficient Framework for Training and Evaluating World Action Models},
  author = {EasyWAM-Team},
  year   = {2026},
  url    = {https://github.com/OpenMOSS/EasyWAM}
}
```

## 📮 联系我们

如果你有任何问题或建议，欢迎加入我们的微信交流群。

<p align="center">
  <img src="assets/EasyWAM-community.png" alt="EasyWAM 微信交流群" width="320">
</p>

## Star History

<a href="https://www.star-history.com/?repos=OpenMOSS%2FEasyWAM&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/EasyWAM&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/EasyWAM&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=OpenMOSS/EasyWAM&type=date&legend=top-left" />
 </picture>
</a>

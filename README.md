<p align="center">
  <img src="assets/icon.png" alt="EasyWAM Logo" width="20%">
</p>

<h1 align="center">🚀 EasyWAM: An Efficient and Easy-to-Use Codebase for World Action Model</h1>

<p align="center">A unified research codebase for training and evaluating World Action Models.</p>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-111111.svg" alt="English README"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg" alt="中文 README"></a>
</p>

## ✨ Overview and Key Features

EasyWAM is a unified research codebase designed to make World Action Model development efficient, reproducible, and easy to extend. It connects model implementation, data processing, distributed training, parameter-efficient fine-tuning, and large-scale evaluation through a consistent workflow.

<p align="center">
  <img src="assets/overview.png" alt="EasyWAM Overview" width="100%">
</p>

- ⚡ **Efficient computation.** Efficiency is a first-class design goal in EasyWAM. It natively integrates **FlashAttention 2/3/4** to accelerate attention workloads and provides complete **LoRA** support from parameter-efficient training and checkpointing to merged inference. BF16, gradient checkpointing, DeepSpeed ZeRO, and PyTorch SDPA fallback further improve speed, memory usage, and compatibility.
- 🚄 **Optimized end-to-end pipeline.** EasyWAM streamlines every stage of training and inference. Sparse video decoding, indexed text caches, persistent workers, prompt caching, and resumable evaluation eliminate repeated work and deliver substantial speedups across both **training and inference**.
- 🧩 **Unified and extensible design.** Modular model components share consistent data, training, checkpoint, and evaluation interfaces, making new WAM designs easier to integrate.
- 🛠️ **Easy-to-use workflows.** Hydra-based configuration, standardized training and evaluation recipes, distributed launchers, automatic GPU sharding, and result summaries keep common workflows straightforward.

> 🌟 **Hope:** We hope EasyWAM will become an efficient and easy-to-use codebase for World Action Model research, enabling researchers to explore WAMs more quickly and easily. More models and benchmarks will be continuously added and supported. We warmly welcome contributions from the community to help make EasyWAM better. If you encounter any problems or have suggestions for improving EasyWAM, please open an issue. We will continue to refine and improve EasyWAM.

## 📰 News

- **[2026-08]** EasyWAM is released with unified training and evaluation workflows for World Action Models.

## 🤖 Supported Models and Benchmarks

### 🧠 Models

- **EasyWAM-Unified.** A single-backbone architecture that places video, action, and robot-state tokens in one Video DiT to jointly predict future video and actions. The architecture is based on [DreamZero](https://github.com/dreamzero0/dreamzero).
- **EasyWAM-MoT.** A dual-backbone architecture with separate Video DiT and Action DiT experts whose tokens interact through shared mixed self-attention. The architecture is based on [FastWAM](https://github.com/yuantianyuan01/FastWAM).
- **EasyWAM-Hidden.** A dual-backbone architecture that uses intermediate Video DiT features as conditional input to a separate Action DiT. The architecture is based on [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT).

| Model | Full-Parameter Training | LoRA Fine-Tuning |
| --- | :---: | :---: |
| EasyWAM-Unified | ✅ | ✅ |
| EasyWAM-MoT | ✅ | ✅ |
| EasyWAM-Hidden | ✅ | ✅ |

### 🏗️ Backbone

| Backbone | Supported |
| --- | :---: |
| Wan2.2-TI2V-5B | ✅ |

### 🧪 Benchmarks

| Benchmark | Training | Evaluation |
| --- | --- | --- |
| LIBERO | Full-parameter and LoRA | Standard evaluation |
| LIBERO-Plus | Uses LIBERO checkpoints | Robustness evaluation |
| RoboTwin | Full-parameter and LoRA | Clean and randomized evaluation |

## 🏆 Benchmark Results

<details open>
<summary><b>LIBERO</b></summary>

**Full-Parameter**

| Model | Spatial | Object | Goal | Long | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 99.0 | 99.4 | 99.2 | 98.2 | 99.0 |
| EasyWAM-MoT | 97.8 | 98.4 | 97.6 | 95.6 | 97.4 |
| EasyWAM-Hidden | 99.4 | 100.0 | 97.0 | 97.8 | 98.6 |

**LoRA (Rank 128)**

| Model | Spatial | Object | Goal | Long | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 84.0 | 97.8 | 92.0 | 81.2 | 88.8 |
| EasyWAM-MoT | 96.8 | 98.8 | 94.4 | 90.4 | 95.1 |
| EasyWAM-Hidden | 96.8 | 99.4 | 92.6 | 86.8 | 93.9 |

</details>

<details open>
<summary><b>LIBERO-Plus</b></summary>

| Model | Success Rate (%) |
| --- | :---: |
| EasyWAM-Unified | 67.5 |
| EasyWAM-MoT | 56.8 |
| EasyWAM-Hidden | 72.4 |

</details>

## ⚡ Efficiency Results

EasyWAM-MoT and FastWAM use the same model architecture, enabling an architecture-matched comparison between the EasyWAM training framework and the original FastWAM codebase. Measured on **8 × NVIDIA H100 GPUs** with a **per-device batch size of 16**, EasyWAM-MoT achieves **121.9 samples/s**, a **2.37×** throughput improvement over FastWAM's 51.5 samples/s. It also reduces data, forward, and backward time per step by **66.0%**, **58.4%**, and **54.7%**, respectively.

<p align="center">
  <img src="assets/efficiency_comparison.svg" alt="EasyWAM and FastWAM training efficiency comparison" width="100%">
</p>

| Framework | Throughput (samples/s) ↑ | Data Time (s) ↓ | Forward Time (s) ↓ | Backward Time (s) ↓ |
| --- | :---: | :---: | :---: | :---: |
| EasyWAM-MoT | **121.9** | **0.0258** | **0.5162** | **0.4842** |
| FastWAM | 51.5 | 0.0759 | 1.2412 | 1.0680 |

On LIBERO, training for **20,000 steps** takes approximately **6 hours** with EasyWAM, compared with approximately **14 hours** using the original FastWAM codebase—a **57.1% reduction** in overall training time. Actual training time may vary with the CPU and GPU configuration; these figures are provided for reference only.

> 🌟 **Hope:** Training and evaluating World Action Models often requires substantial computational resources. EasyWAM aims to lower this barrier with an efficient and lightweight codebase, enabling researchers to train models, run evaluations, and iterate quickly even with limited compute. Through continuous efficiency improvements, we hope researchers can devote more of their resources to model and algorithm innovation and that more members of the community can participate in WAM research.

## 🚀 Quick Start

### 🛠️ Installation

```bash
conda create -n easywam python=3.10 -y
conda activate easywam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

[FlashAttention](https://github.com/Dao-AILab/flash-attention) is optional. When installed, EasyWAM uses the fastest compatible implementation available and otherwise falls back to PyTorch SDPA.

### 📦 Prepare Models

Place the models at the paths used by the default configs:

```text
checkpoints/
├── Wan2.2-TI2V-5B/
└── umt5-xxl/
```

Download models from Hugging Face by running the following commands from the project root:

```bash
mkdir -p checkpoints
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan2.2-TI2V-5B
huggingface-cli download google/umt5-xxl --local-dir checkpoints/umt5-xxl
```

EasyWAM-MoT and EasyWAM-Hidden also use an interpolated ActionDiT initialization:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified does not require this ActionDiT checkpoint.

### 📝 Precompute Text Embeddings

Run this once after preparing a benchmark dataset:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot
# Or: python scripts/precompute_text_embeds.py task=robotwin_easywam_mot
```

The cache depends on the benchmark data rather than the selected WAM architecture, so it can be shared by the corresponding EasyWAM-MoT, EasyWAM-Unified, EasyWAM-Hidden, and LoRA tasks.

### 🏋️ Train

Set `PYTHON_PATH` to the Python executable in your local EasyWAM environment before launching, or modify its default value in the training scripts:

```bash
export PYTHON_PATH=~/miniforge3/envs/easywam/bin/python
# Or:
export PYTHON_PATH=~/miniconda3/envs/easywam/bin/python
```

The launchers accept Hydra overrides directly. Set the number of local processes through `NPROC_PER_NODE`:

```bash
# DeepSpeed ZeRO-1 on 8 local GPUs
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot

# DeepSpeed ZeRO-2 LoRA training on 4 local GPUs
NPROC_PER_NODE=4 bash scripts/train_zero2.sh task=robotwin_easywam_unified_lora
```

`scripts/train_zero2_offload.sh` enables ZeRO-2 CPU offload. Multi-node runs additionally use `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

### 📊 Evaluate

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot \
  ckpt=<path/to/checkpoint.pt>

# LIBERO-Plus (uses a LIBERO checkpoint)
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot \
  ckpt=<path/to/checkpoint.pt>

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot \
  ckpt=<path/to/checkpoint.pt>
```

The managers default to 8 GPUs. Override `MULTIRUN.num_gpus` and `MULTIRUN.max_tasks_per_gpu` to match your machine. See the benchmark guides for installation, data layout, checkpoint examples, filtering, and resume behavior.

## 📚 Documentation

| Benchmark | English | 中文 |
| --- | --- | --- |
| LIBERO | [Guide](docs/libero.md) | [使用指南](docs/libero_zh.md) |
| LIBERO-Plus | [Guide](docs/libero_plus.md) | [使用指南](docs/libero_plus_zh.md) |
| RoboTwin | [Guide](docs/robotwin.md) | [使用指南](docs/robotwin_zh.md) |

## 🗂️ Repository Layout

```text
EasyWAM/
├── configs/          # Data, model, task, training, and evaluation configs
├── docs/             # Repository documents
├── experiments/      # Benchmark evaluators
├── scripts/          # Training, preprocessing, and caching entrypoints
├── src/              # Models, data pipeline, runtime, and trainer
├── checkpoints/      # External and trained checkpoints
├── data/             # Local datasets and text caches
└── runs/             # Training outputs
```

## 🙏 Acknowledgements

We are especially grateful to the [FastWAM](https://github.com/yuantianyuan01/FastWAM) team for its substantial code contributions, which provided an important foundation for EasyWAM. We also thank the [DreamZero](https://github.com/dreamzero0/dreamzero) and [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) teams for their ideas and open-source resources, as well as the teams behind [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) for releasing their benchmarks, code, and assets.

## 📝 Citation

We welcome you to cite **EasyWAM**'s experimental results and codebase in your research. If **EasyWAM** is useful in your research, please cite:

```bibtex
@misc{easywam2026,
  title  = {EasyWAM: An Efficient and Easy-to-Use Codebase for World Action Model},
  author = {EasyWAM Contributors},
  year   = {2026},
  url    = {https://github.com/KevinConqueror/EasyWAM}
}
```

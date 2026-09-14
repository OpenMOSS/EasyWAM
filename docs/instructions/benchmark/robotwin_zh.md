# RoboTwin 评测指南

[English](robotwin.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/robotwin_zh.md) | [返回项目 README](../../../README_zh.md)

## 安装

从 GitHub 克隆当前 RoboTwin 仓库及其 XPolicyLab 子模块：

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
```

使用 Python 3.10，并安装 EasyWAM 尚未提供的评测依赖：

```bash
pip install scipy==1.10.1 transforms3d==0.4.2 sapien==3.0.0b1 \
  mplib==0.2.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 \
  "pydantic>=2.5" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8"

git clone https://github.com/NVlabs/curobo.git third_party/RoboTwin/envs/curobo
pip install --no-build-isolation -e third_party/RoboTwin/envs/curobo
```

RoboTwin 使用 Vulkan 渲染，并通过系统 `ffmpeg` 命令保存评测视频。在 Ubuntu 上安装：

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg unzip
```

## 资源

下载并解压官方资源，然后更新资源路径：

```bash
huggingface-cli download TianxingChen/RoboTwin2.0 \
  background_texture.zip embodiments.zip objects.zip \
  --repo-type dataset \
  --local-dir third_party/RoboTwin/assets

(
  cd third_party/RoboTwin/assets
  unzip -q -o background_texture.zip
  unzip -q -o embodiments.zip
  unzip -q -o objects.zip
)
(cd third_party/RoboTwin && python scripts/update_embodiment_config_path.py)
```

clean、randomized、相机、embodiment 和任务步数配置已经包含在克隆仓库的 `env_cfg/task_config/` 中。

## 评测

评测 RoboTwin `_eval_step_limit.yml` 中列出的全部任务：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0-lerobot-v3.0/dataset_stats.json \
  MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[0,2,4,6]' \
  MULTIRUN.workers_per_gpu=2 MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

只评测一个任务或覆盖语言指令划分：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.instruction_type=seen
```

manager 会依次评测每个任务的 `demo_clean` 和 `demo_randomized`。没有传入 instruction override 时，使用当前 RoboTwin 各任务配置声明的划分。`EVALUATION.eval_num_episodes` 控制每个阶段的 episode 数量，`EVALUATION.replan_steps` 控制重新规划前连续执行的预测动作数。

`MULTIRUN.gpu_ids=null` 时使用前 `num_gpus` 张卡；否则从有序的 `gpu_ids` 候选池中选择前 `num_gpus` 项。模型 worker 在选中的 GPU 间轮询分配。每个 worker 独立加载一份 EasyWAM，其 rollout 客户端使用相互隔离的 XPolicyLab 会话并共享该 worker 的动态推理 batcher；提高 `workers_per_gpu` 会增加模型显存占用。结果仍保存在 `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`，其中包括上游产物、各阶段结果、worker 日志、`summary.json` 和 `summary.csv`。复用相同的 `EVALUATION.output_dir` 时，只续评尚无有效结果的阶段。

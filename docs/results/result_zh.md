# EasyWAM Benchmark 结果

[English](result.md)

本页面展示 EasyWAM 的 Benchmark 结果。报告指标为任务成功率（%），数值越高越好。所有结果均在 `state_position: sequence` 设置下报告，即将 state/proprio token 放在 action token 之后并加入模型序列。

## LIBERO

LIBERO 从空间关系、物体交互、目标条件和长序列任务四个方面评测机器人操作能力。

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

**全参数训练**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 98.4 | 98.8 | 99.2 | 98.0 | 98.6 |
| EasyWAM-MoT | 97.0 | 99.2 | 96.6 | 94.0 | 96.7 |
| EasyWAM-MoT-Joint | 98.2 | 98.0 | 97.6 | 96.8 | 97.7 |
| EasyWAM-MoT-IDM | 99.0 | 99.2 | 98.8 | 97.4 | 98.6 |
| EasyWAM-Hidden | 99.2 | 100.0 | 97.8 | 98.2 | 98.8 |

**LoRA（Rank 128）**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 91.2 | 98.8 | 91.8 | 66.2 | 87.0 |
| EasyWAM-MoT | 96.8 | 99.6 | 97.4 | 90.0 | 95.9 |
| EasyWAM-Hidden | 98.0 | 99.8 | 89.4 | 86.6 | 93.5 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

**全参数训练**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 97.8 | 99.4 | 97.0 | 93.0 | 96.8 |
| EasyWAM-MoT | 98.0 | 98.4 | 98.4 | 95.6 | 97.6 |
| EasyWAM-MoT-Joint | 98.6 | 99.8 | 98.4 | 96.0 | 98.2 |
| EasyWAM-MoT-IDM | 99.4 | 99.4 | 99.8 | 98.4 | 99.3 |
| EasyWAM-Hidden | 97.2 | 98.8 | 96.0 | 95.0 | 96.8 |

</details>

## LIBERO-Plus

LIBERO-Plus 通过改变背景、相机、语言、布局、光照、观测噪声和机器人外观，评测 LIBERO 策略的鲁棒性。

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

| 模型 | Background | Camera | Language | Layout | Light | Noise | Robot | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 72.3 | 54.8 | 93.7 | 83.4 | 97.0 | 72.0 | 83.4 | 79.0 |
| EasyWAM-MoT | 64.5 | 45.5 | 71.4 | 80.1 | 94.7 | 78.5 | 71.5 | 71.7 |
| EasyWAM-MoT-Joint | 60.9 | 47.3 | 89.7 | 80.5 | 92.4 | 68.9 | 75.9 | 73.3 |
| EasyWAM-MoT-IDM | 62.2 | 52.0 | 94.1 | 82.0 | 93.2 | 67.8 | 78.0 | 75.3 |
| EasyWAM-Hidden | 59.3 | 57.0 | 93.2 | 84.3 | 95.0 | 70.8 | 83.7 | 77.6 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

| 模型 | Background | Camera | Language | Layout | Light | Noise | Robot | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 72.7 | 63.7 | 90.1 | 82.6 | 89.2 | 72.1 | 79.2 | 78.2 |
| EasyWAM-MoT | 60.7 | 75.1 | 92.3 | 82.6 | 92.6 | 81.3 | 58.5 | 77.7 |
| EasyWAM-MoT-Joint | 62.1 | 60.8 | 96.6 | 84.5 | 94.4 | 69.4 | 76.9 | 77.7 |
| EasyWAM-MoT-IDM | 66.4 | 59.8 | 93.8 | 85.7 | 89.0 | 72.3 | 81.9 | 78.4 |
| EasyWAM-Hidden | 59.5 | 65.9 | 92.6 | 85.0 | 89.1 | 68.3 | 82.5 | 77.8 |

</details>

## 说明

- 结果仅供参考，实际数值可能因硬件、软件版本、随机种子、Checkpoint 和评测配置而变化。

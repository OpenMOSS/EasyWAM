# EasyWAM Benchmark 结果

[English](result.md)

本页面展示 EasyWAM 的 Benchmark 结果。报告指标为任务成功率（%），数值越高越好。

## LIBERO

LIBERO 从空间关系、物体交互、目标条件和长序列任务四个方面评测机器人操作能力。

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

**全参数训练**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 99.0 | 99.4 | 99.2 | 98.2 | 99.0 |
| EasyWAM-MoT | 97.8 | 98.4 | 97.6 | 95.6 | 97.4 |
| EasyWAM-Hidden | 99.4 | 100.0 | 97.0 | 97.8 | 98.6 |

**LoRA（Rank 128）**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 84.0 | 97.8 | 92.0 | 81.2 | 88.8 |
| EasyWAM-MoT | 96.8 | 98.8 | 94.4 | 90.4 | 95.1 |
| EasyWAM-Hidden | 96.8 | 99.4 | 92.6 | 86.8 | 93.9 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

**全参数训练**

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 95.2 | 98.4 | 95.6 | 84.8 | 93.5 |
| EasyWAM-MoT | 99.0 | 99.8 | 98.4 | 95.2 | 98.1 |
| EasyWAM-Hidden | 99.6 | 99.8 | 98.6 | 96.8 | 98.7 |

</details>

## LIBERO-Plus

LIBERO-Plus 通过改变背景、相机、语言、布局、光照、观测噪声和机器人外观，评测 LIBERO 策略的鲁棒性。

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

| 模型 | 原始任务 | Background | Camera | Language | Layout | Light | Noise | Robot | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 99.0 | 55.8 | 33.7 | 93.7 | 80.6 | 92.2 | 50.2 | 71.4 | 67.5 |
| EasyWAM-MoT | 97.4 | 52.8 | 20.6 | 80.4 | 65.2 | 85.1 | 51.5 | 49.7 | 56.8 |
| EasyWAM-Hidden | 98.6 | 56.8 | 49.2 | 95.3 | 81.0 | 90.4 | 58.2 | 77.4 | 72.4 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

| 模型 | 原始任务 | Background | Camera | Language | Layout | Light | Noise | Robot | 平均 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 93.5 | 63.8 | 56.5 | 83.7 | 79.8 | 86.1 | 72.3 | 76.9 | 74.1 |
| EasyWAM-MoT | 98.1 | 51.9 | 57.7 | 93.4 | 82.2 | 94.4 | 58.7 | 55.2 | 70.2 |
| EasyWAM-Hidden | 98.7 | 58.2 | 65.2 | 94.9 | 85.1 | 88.7 | 74.2 | 86.2 | 79.4 |

</details>

## 说明

- 结果仅供参考，实际数值可能因硬件、软件版本、随机种子、Checkpoint 和评测配置而变化。

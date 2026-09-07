# EasyWAM Benchmark Results

[中文](result_zh.md)

This page presents EasyWAM's benchmark results. The reported metric is task success rate in percent, and higher values are better.

## LIBERO

LIBERO evaluates robotic manipulation across spatial-relation, object-interaction, goal-conditioned, and long-horizon task suites.

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

**Full-Parameter**

| Model | Spatial | Object | Goal | LIBERO-10 | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 99.0 | 99.4 | 99.2 | 98.2 | 99.0 |
| EasyWAM-MoT | 97.8 | 98.4 | 97.6 | 95.6 | 97.4 |
| EasyWAM-Hidden | 99.4 | 100.0 | 97.0 | 97.8 | 98.6 |

**LoRA (Rank 128)**

| Model | Spatial | Object | Goal | LIBERO-10 | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 84.0 | 97.8 | 92.0 | 81.2 | 88.8 |
| EasyWAM-MoT | 96.8 | 98.8 | 94.4 | 90.4 | 95.1 |
| EasyWAM-Hidden | 96.8 | 99.4 | 92.6 | 86.8 | 93.9 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

**Full-Parameter**

| Model | Spatial | Object | Goal | LIBERO-10 | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 95.2 | 98.4 | 95.6 | 84.8 | 93.5 |
| EasyWAM-MoT | 99.0 | 99.8 | 98.4 | 95.2 | 98.1 |
| EasyWAM-Hidden | 99.6 | 99.8 | 98.6 | 96.8 | 98.7 |

</details>

## LIBERO-Plus

LIBERO-Plus evaluates the robustness of LIBERO policies under changes to backgrounds, cameras, language, layouts, lighting, observation noise, and robot appearance.

<details open>
<summary><b>Backbone</b>: Wan2.2-TI2V-5B</summary>

| Model | Orig | Background | Camera | Language | Layout | Light | Noise | Robot | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 99.0 | 55.8 | 33.7 | 93.7 | 80.6 | 92.2 | 50.2 | 71.4 | 67.5 |
| EasyWAM-MoT | 97.4 | 52.8 | 20.6 | 80.4 | 65.2 | 85.1 | 51.5 | 49.7 | 56.8 |
| EasyWAM-Hidden | 98.6 | 56.8 | 49.2 | 95.3 | 81.0 | 90.4 | 58.2 | 77.4 | 72.4 |

</details>

<details open>
<summary><b>Backbone</b>: Cosmos-Predict2.5-2B</summary>

| Model | Orig | Background | Camera | Language | Layout | Light | Noise | Robot | Avg. |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | 93.5 | 63.8 | 56.5 | 83.7 | 79.8 | 86.1 | 72.3 | 76.9 | 74.1 |
| EasyWAM-MoT | 98.1 | 51.9 | 57.7 | 93.4 | 82.2 | 94.4 | 58.7 | 55.2 | 70.2 |
| EasyWAM-Hidden | 98.7 | 58.2 | 65.2 | 94.9 | 85.1 | 88.7 | 74.2 | 86.2 | 79.4 |

</details>

## Notes

- Results are provided for reference and may vary with hardware, software versions, random seeds, checkpoints, and evaluation settings.

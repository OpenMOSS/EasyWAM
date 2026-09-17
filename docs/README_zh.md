# EasyWAM 文档

[English](README.md) | [返回项目 README](../README_zh.md)

本目录将操作指南、研究文章和结果汇总分开组织。请先进入对应的指南分区，再选择具体文档。

## 操作指南

| 分区 | 内容 | English | 中文 |
| --- | --- | --- | --- |
| Backbone | 模型下载、预处理、训练与评测用法 | [Index](instructions/backbone/README.md) | [索引](instructions/backbone/README_zh.md) |
| 数据 | LIBERO、RoboTwin、RoboDojo 与 RoboCasa 训练数据准备 | [Index](instructions/data/README.md) | [索引](instructions/data/README_zh.md) |
| Benchmark | 仿真环境准备与评测 | [Index](instructions/benchmark/README.md) | [索引](instructions/benchmark/README_zh.md) |
| 配置 | 模型、训练、效率与 Hydra 配置 | [Index](instructions/config/README.md) | [索引](instructions/config/README_zh.md) |

## 研究与结果

| 文章 | English | 中文 |
| --- | --- | --- |
| 什么样的 WAM 架构是我们需要的？ | [Read](blogs/blog01_arch.md) | [阅读](blogs/blog01_arch_zh.md) |
| Benchmark 结果 | [View](results/result.md) | [查看](results/result_zh.md) |

## 目录结构

```text
docs/
├── README.md              # 英文文档索引
├── README_zh.md           # 中文文档索引
├── blogs/                 # 架构分析和项目文章
├── results/               # Benchmark 结果汇总
└── instructions/
    ├── backbone/          # Backbone 集成指南
    ├── benchmark/         # 评测准备与使用
    ├── config/            # 项目配置参考
    └── data/              # 训练数据准备
```

添加文档时应同步更新两个索引，并在可行时同时提供中英文版本。请使用相对链接，确保文档在本地 checkout 和代码托管页面中均可访问。

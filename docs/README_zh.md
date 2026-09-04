# EasyWAM 文档

[English](README.md) | [返回项目 README](../README_zh.md)

本目录包含项目配置参考、Benchmark 与集成指南，以及架构分析文章。

## 配置说明

| 专题 | English | 中文 |
| --- | --- | --- |
| 配置概览 | [Guide](configurations/README.md) | [配置指南](configurations/README_zh.md) |
| 模型与 Backbone | [Guide](configurations/models.md) | [配置指南](configurations/models_zh.md) |
| 数据、训练与评测 | [Guide](configurations/training.md) | [配置指南](configurations/training_zh.md) |
| 效率与显存 | [Guide](configurations/efficiency.md) | [配置指南](configurations/efficiency_zh.md) |

## Benchmark 与集成指南

| 专题 | English | 中文 |
| --- | --- | --- |
| LIBERO | [Guide](instructions/libero.md) | [使用指南](instructions/libero_zh.md) |
| LIBERO-Plus | [Guide](instructions/libero_plus.md) | [使用指南](instructions/libero_plus_zh.md) |
| RoboTwin | [Guide](instructions/robotwin.md) | [使用指南](instructions/robotwin_zh.md) |
| FLUX.2 / ImageWAM 集成 | [Guide](instructions/flux2_imagewam_integration.md) | — |

## 博客

| 文章 | English | 中文 |
| --- | --- | --- |
| 什么样的 WAM 架构是我们需要的？ | [Read](blogs/blog01_arch.md) | [阅读](blogs/blog01_arch_zh.md) |

## 目录结构

```text
docs/
├── README.md              # 英文文档索引
├── README_zh.md           # 中文文档索引
├── blogs/                 # 架构分析和项目文章
├── configurations/        # 项目配置参考
└── instructions/          # Benchmark 与集成指南
```

添加文档时应同步更新两个索引，并在可行时同时提供中英文版本。请使用相对链接，确保文档在本地 checkout 和代码托管页面中均可访问。

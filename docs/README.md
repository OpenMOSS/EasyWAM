# EasyWAM Documentation

[中文](README_zh.md) | [Back to project README](../README.md)

This directory contains the project configuration reference, benchmark and integration guides, and architecture articles.

## Configuration

| Topic | English | 中文 |
| --- | --- | --- |
| Configuration overview | [Guide](configurations/README.md) | [配置指南](configurations/README_zh.md) |
| Models and backbones | [Guide](configurations/models.md) | [配置指南](configurations/models_zh.md) |
| Data, training, and evaluation | [Guide](configurations/training.md) | [配置指南](configurations/training_zh.md) |
| Efficiency and memory | [Guide](configurations/efficiency.md) | [配置指南](configurations/efficiency_zh.md) |

## Benchmark and integration guides

| Topic | English | 中文 |
| --- | --- | --- |
| LIBERO | [Guide](instructions/libero.md) | [使用指南](instructions/libero_zh.md) |
| LIBERO-Plus | [Guide](instructions/libero_plus.md) | [使用指南](instructions/libero_plus_zh.md) |
| RoboTwin | [Guide](instructions/robotwin.md) | [使用指南](instructions/robotwin_zh.md) |
| FLUX.2 / ImageWAM integration | [Guide](instructions/flux2_imagewam_integration.md) | — |

## Blogs

| Article | English | 中文 |
| --- | --- | --- |
| What WAM Architecture Do We Need? | [Read](blogs/blog01_arch.md) | [阅读](blogs/blog01_arch_zh.md) |

## Directory layout

```text
docs/
├── README.md              # English documentation index
├── README_zh.md           # Chinese documentation index
├── blogs/                 # Architecture analysis and project articles
├── configurations/        # Project configuration reference
└── instructions/          # Benchmark and integration guides
```

When adding documentation, update both indexes and provide an English/Chinese pair where practical. Use relative links so the documentation works in local checkouts and repository browsers.

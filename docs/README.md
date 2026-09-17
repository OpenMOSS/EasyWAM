# EasyWAM Documentation

[中文](README_zh.md) | [Back to project README](../README.md)

This directory separates operational instructions, research articles, and result summaries. Enter an instruction section first, then select its specific guide.

## Instructions

| Section | Contents | English | 中文 |
| --- | --- | --- | --- |
| Backbone | Model downloads, preprocessing, training, and evaluation usage | [Index](instructions/backbone/README.md) | [索引](instructions/backbone/README_zh.md) |
| Data | LIBERO, RoboTwin, RoboDojo, and RoboCasa training-data preparation | [Index](instructions/data/README.md) | [索引](instructions/data/README_zh.md) |
| Benchmark | Simulator preparation and evaluation | [Index](instructions/benchmark/README.md) | [索引](instructions/benchmark/README_zh.md) |
| Config | Model, training, efficiency, and Hydra configuration | [Index](instructions/config/README.md) | [索引](instructions/config/README_zh.md) |

## Research and results

| Article | English | 中文 |
| --- | --- | --- |
| What WAM Architecture Do We Need? | [Read](blogs/blog01_arch.md) | [阅读](blogs/blog01_arch_zh.md) |
| Benchmark results | [View](results/result.md) | [查看](results/result_zh.md) |

## Directory layout

```text
docs/
├── README.md              # English documentation index
├── README_zh.md           # Chinese documentation index
├── blogs/                 # Architecture analysis and project articles
├── results/               # Benchmark result summaries
└── instructions/
    ├── backbone/          # Backbone integration guides
    ├── benchmark/         # Evaluation setup and usage
    ├── config/            # Project configuration reference
    └── data/              # Training-data preparation
```

When adding documentation, update both indexes and provide an English/Chinese pair where practical. Use relative links so the documentation works in local checkouts and repository browsers.

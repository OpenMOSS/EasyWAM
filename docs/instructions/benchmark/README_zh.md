# Benchmark 文档索引

[English](README.md) | [文档总索引](../../README_zh.md) | [返回项目 README](../../../README_zh.md)

本分区说明仿真环境准备、checkpoint 评测、动态推理 batch、结果输出和断点续评。训练数据准备统一放在[数据文档](../data/README_zh.md)中。

不同 Backbone 的已发布指标统一汇总在 [Benchmark 结果](../../results/result_zh.md)中（[English](../../results/result.md)）。

| Benchmark | 评测范围 | 文档 |
| --- | --- | --- |
| LIBERO | 标准四 suite 评测 | [准备与评测](libero_zh.md) |
| LIBERO-Pro | 官方五维泛化评测 | [准备与评测](libero_pro_zh.md) |
| LIBERO-Plus | 使用 LIBERO checkpoint 的鲁棒性评测 | [准备与评测](libero_plus_zh.md) |
| RoboTwin | Clean 与 randomized 评测 | [准备与评测](robotwin_zh.md) |
| RoboDojo | 官方 42-task 仿真评测 | [准备与评测](robodojo_zh.md) |
| RoboCasa365 | 官方 50-task Human300 评测 | [准备与评测](robocasa_zh.md) |

## 推理 Tensor 类型与维度约定

`infer_action`、`infer_joint`、`infer` 接收一张已归一化的浮点 RGB 图像 `[3,H,W]` 或 `[1,3,H,W]`。单样本入口只进行一次维度转换，返回 CPU float32 动作 `[A,Da]`，解码视频为 `list[PIL.Image]`。缓存文本可使用 `[L,Dc]`/`[L]`，也可带大小为 1 的 batch 维。

`infer_action_batch`、`infer_joint_batch` 要求 image `[B,3,H,W]`、context `[B,L,Dc]`、bool context_mask `[B,L]`；启用状态时，proprio 为 `[B,Ds]`。在线 prompt 必须是包含 B 个字符串的序列。Hidden、Unified 还接收显式状态序列 `[B,S,Ds]`，维持原有推理时使用首个状态的行为。B=1 也必须带 batch 维，不再将二维状态隐式猜成状态历史。

批量动作固定为 CPU float32 `[B,A,Da]`，解码视频固定为 `list[list[PIL.Image]]`，B=1 时同样如此。支持 `decode_video` 的接口在关闭视频解码时省略 video 字段。单请求队列保存 `[3,H,W]` 图像和 `[Ds]` 状态，统一堆叠后交给模型，再按固定输出格式拆分。模型内部的视频 VAE 使用五维批量 Tensor，独立 VAE 的列表输入兼容仍保留。

Wan head 内部时间 embedding 固定为 `[B,Q,D]`，共享时间条件使用 `Q=1`；Cosmos 时间特征接收入口规范化后的 `[B,T]`。视频 VAE 的 scale 为两个逐通道 `[z_dim]` Tensor；self-attention 只准备一次 RoPE，供 Q/K 共用。`ContinuousFlowMatchScheduler.training_weight` 保持输入 timestep 的形状，B=1 的批量输入也返回 `[1]`。

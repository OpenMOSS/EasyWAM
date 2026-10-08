# 训练数据文档索引

[English](README.md) | [文档总索引](../../README_zh.md) | [返回项目 README](../../../README_zh.md)

本分区说明数据集下载、预期的 LeRobot 目录结构、文本特征预处理、归一化统计和对应的训练入口。

| 数据集 | 内容 | 文档 |
| --- | --- | --- |
| LIBERO | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](libero_zh.md) |
| RoboTwin | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robotwin_zh.md) |
| RoboDojo | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robodojo_zh.md) |
| RoboCasa365 | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robocasa_zh.md) |

仿真环境安装和 rollout 评测统一放在 [Benchmark 文档](../benchmark/README_zh.md)中。

## 数据处理

构造数据集时，训练与验证复用元数据、底层 Arrow 数据和归一化统计，各自保留独立的 episode 映射与处理状态。每个样本的数值窗口合并为一次读取；相同相机窗口复用时间戳，padding 重复帧仅解码一次。训练内部保留 uint8 图像，到 processor 才转为 float。RoboTwin 和 RoboDojo 每路相机只缩放一次：顶部为 256×320，腕部各为 128×160，再组装为 384×320。相机布局、输出尺寸和增强参数保持一致，像素插值结果会与旧的两次缩放不同。

processor 默认 `stack_images=true`、`include_gt_action=true`。不同尺寸相机的配方设置 `stack_images=false`；RobotVideoDataset 关闭未使用的 GT 动作副本。解码函数默认仍输出 float，训练路径指定 `output_dtype=torch.uint8`。文本缓存的文件名和 payload 格式保持兼容，运行时按需读取，不增加 embedding 内存缓存。

文本预计算默认跳过已有文件；只有存在缺失 prompt 才初始化编码器，重新生成可传入 `+overwrite=true`。归一化统计计算最多使用 4 个 worker、8 个在途任务，任务只返回紧凑统计量，避免 futures 保留整集 episode tensor。

文本编码器计算和 embedding 缓存精度均跟随 `mixed_precision`：`bf16` 对应 bfloat16，`fp16` 对应 float16，`no` 对应 float32。预处理与训练应指定相同的 `mixed_precision`。BF16 沿用原有缓存文件名；FP16、FP32 分别使用 `.fp16.pt`、`.fp32.pt` 后缀，不同精度的缓存可以共存。训练读取时会校验缓存 dtype 是否与配置一致。

## Tensor 类型与维度约定

RobotVideoDataset 返回单样本：video `[3,T,H,W]`、action `[A,Da]`、proprio `[S,Ds]`、字符串 prompt、context `[L,Dc]` 和 bool 类型 context_mask `[L]`。视频为 `[-1,1]` 浮点 Tensor，缓存 context 的 dtype 跟随 `mixed_precision`。默认 DataLoader collate 增加 batch 维，并将 prompt 收集为列表；验证从 dataset 取单样本后只增加一次 batch 维。

时间 padding mask 对应各自的采样窗口，维度 padding mask 在 collate 前为 `[Da]` 或 `[Ds]`。视频、动作和原始观察窗口的长度可以不同。WAMProcessor 的相机输出由 `stack_images` 明确决定：开启时为 `[K,T,3,H,W]` Tensor，关闭时为多个 `[T,3,Hk,Wk]` Tensor 组成的列表。

Processor 通过 `camera_views` 按其声明的格式提供相机 Tensor。改变 `pixel_values` 布局的自定义 processor 需要覆盖该方法；BaseProcessor 保留原有单相机格式。

HF 数值列的转换与取行方式在初始化时根据数据格式确定，继续支持原生 Tensor 列和图像 transform 的列表格式；每个窗口只提取请求的行，并保留 padding 产生的重复索引。

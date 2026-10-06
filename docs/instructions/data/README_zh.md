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

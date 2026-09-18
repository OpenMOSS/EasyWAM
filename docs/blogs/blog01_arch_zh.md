# 🧭 什么样的 WAM 架构是我们需要的？

我们在设计 World Action Model 时始终绕不开一个问题：**到底应该让视频和动作共享一个骨干，还是分开建模？在推理时间是否一定要生成未来预测？** 本文基于 EasyWAM 在 LIBERO / LIBERO-Plus 上对三种代表性架构的完整实验数据，把训练方式（全参数 / LoRA）和泛化能力（LIBERO-Plus）两条主线的结果放在一起对照着看，逐一给出数据和背后的架构原因。以下结果涵盖 Wan2.2-TI2V-5B 和 Cosmos-Predict2.5-2B；LIBERO 的 LoRA 结果来自 Wan2.2。

## 🧱 实验设置

- **EasyWAM-Unified** ([DreamZero](https://arxiv.org/pdf/2602.15922)-like)：单 DiT 架构，把视频和动作 token 放进同一个 Video DiT 联合去噪，动作与视频在同一组自注意力中双向耦合。
- **EasyWAM-Hidden** ([DiT4DiT](https://arxiv.org/pdf/2603.10448)-like)：双 DiT 架构，用 Video DiT 的中间特征单向条件化一个独立的 Action DiT，推理时仍需要对未来视频进行预测。
- **EasyWAM-MoT** ([FastWAM](https://arxiv.org/pdf/2603.16666)-like)：双 DiT 架构，独立的 Video DiT 和 Action DiT 通过共享混合自注意力交互，推理时只做动作预测，不生成/预测未来视频。

---

## RQ1：全参数 vs. LoRA 训练下，不同架构的性能表现如何？

**背景**：我们把全参数训练和 LoRA（Rank 128）训练的结果放在同一张表里，看训练方式的切换是否会改变架构之间的相对排名。

**结果（LIBERO，Avg.）**

| Backbone | 模型 | 结构 | 训练方式 | Spatial | Object | Goal | Long | **Avg.** |
| --- | --- | --- | --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2 | EasyWAM-Unified | 单 DiT | 全参数 | 98.4 | 98.8 | 99.2 | 98.0 | **98.6** |
| Wan2.2 | EasyWAM-Hidden | 双 DiT | 全参数 | 99.2 | 100.0 | 97.8 | 98.2 | **98.8** |
| Wan2.2 | EasyWAM-MoT | 双 DiT | 全参数 | 97.0 | 99.2 | 96.6 | 94.0 | **96.7** |
| Wan2.2 | EasyWAM-Unified | 单 DiT | LoRA | 91.2 | 98.8 | 91.8 | 66.2 | **87.0** |
| Wan2.2 | EasyWAM-Hidden | 双 DiT | LoRA | 98.0 | 99.8 | 89.4 | 86.6 | **93.5** |
| Wan2.2 | EasyWAM-MoT | 双 DiT | LoRA | 96.8 | 99.6 | 97.4 | 90.0 | **95.9** |
| Cosmos2.5 | EasyWAM-Unified | 单 DiT | 全参数 | 97.8 | 99.4 | 97.0 | 93.0 | **96.8** |
| Cosmos2.5 | EasyWAM-Hidden | 双 DiT | 全参数 | 97.2 | 98.8 | 96.0 | 95.0 | **96.8** |
| Cosmos2.5 | EasyWAM-MoT | 双 DiT | 全参数 | 98.0 | 98.4 | 98.4 | 95.6 | **97.6** |

**实验发现**

- **全参数训练的排名因 backbone 而异。** Wan2.2 下，Hidden（98.8）略高于 Unified（98.6），MoT 为 96.7；Cosmos2.5 下，MoT（97.6）领先，Unified 与 Hidden 均为 96.8。领先模型的分数接近，且排名随 backbone 改变，不能据此得出某一架构跨 backbone 始终最优的结论。
- **Wan2.2 LoRA 微调：MoT（95.9）> Hidden（93.5）> Unified（87.0）。** 与全参数训练相比，Unified 下降 11.6 个百分点，尤其是 LIBERO-10 从 98.0 降至 66.2；Hidden 下降 5.3 个百分点，MoT 下降 0.8 个百分点。这表明当前模型对 LoRA 的敏感程度不同，但仅凭成功率还无法确定差距由哪部分架构造成。

---

## RQ2：不同架构的泛化表现如何？

**背景**：[LIBERO-Plus](https://arxiv.org/pdf/2510.13626) 在 LIBERO 训练好的检查点基础上，系统性地扰动背景、相机视角、语言指令、物体布局、光照、噪声、机器人本体，用来检验模型是否学到了真正可泛化的视觉-动作映射。三种架构在测试时的推理范式并不相同：MoT 推理时不生成/预测未来视频；Hidden 和 Unified 在推理时都需要对未来视频进行预测。

**结果（LIBERO-Plus，Avg.）**

| Backbone | 模型 | 测试时是否预测视频 | Background | Camera | Language | Layout | Light | Noise | Robot | **Avg.** |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wan2.2 | EasyWAM-Unified | ✅ | 72.3 | 54.8 | 93.7 | 83.4 | 97.0 | 72.0 | 83.4 | **79.0** |
| Wan2.2 | EasyWAM-Hidden | ✅ | 59.3 | 57.0 | 93.2 | 84.3 | 95.0 | 70.8 | 83.7 | **77.6** |
| Wan2.2 | EasyWAM-MoT | ❌ | 64.5 | 45.5 | 71.4 | 80.1 | 94.7 | 78.5 | 71.5 | **71.7** |
| Cosmos2.5 | EasyWAM-Unified | ✅ | 72.7 | 63.7 | 90.1 | 82.6 | 89.2 | 72.1 | 79.2 | **78.2** |
| Cosmos2.5 | EasyWAM-Hidden | ✅ | 59.5 | 65.9 | 92.6 | 85.0 | 89.1 | 68.3 | 82.5 | **77.8** |
| Cosmos2.5 | EasyWAM-MoT | ❌ | 60.7 | 75.1 | 92.3 | 82.6 | 92.6 | 81.3 | 58.5 | **77.7** |

**实验发现**

- **泛化排名会随 backbone 改变。** Wan2.2 下，Unified（79.0）和 Hidden（77.6）领先 MoT（71.7）；Cosmos2.5 下，Unified（78.2）、Hidden（77.8）与 MoT（77.7）非常接近。因此，Wan2.2 上的差距不足以证明预测未来视频一定能提升 LIBERO-Plus 表现；三种架构还存在其他差异。
- **不同扰动类别各有优势模型。** Cosmos2.5 MoT 的 Camera（75.1）和 Noise（81.3）最高，但 Robot 只有 58.5，低于 Unified（79.2）和 Hidden（82.5）。Wan2.2 MoT 的 Noise（78.5）同样领先。两个 backbone 的总平均分均由 Unified 领先，Hidden 不再领先 Unified。要把差距归因于“视频-动作对齐税”，仍需控制其他变量的消融实验。

---

EasyWAM 是一个由社区共同开发、持续演进的 WAM 训练基建，我们欢迎任何形式的 contribution，也期待更多人成为 contributors。如果你对 WAM 的训练设置还有其他疑问，欢迎在 [Issues](https://github.com/OpenMOSS/EasyWAM/issues)  中提出——我们会针对性地开展实验，提供可复现的分析结果；已有的 issue 也可以直接 +1，关注度较高的问题我们会优先加速处理。如果你在自己的场景中做过类似的对比实验，同样欢迎在 Issues 中分享你的发现，我们会持续把更多模型和基准加入这套系统性评测中。

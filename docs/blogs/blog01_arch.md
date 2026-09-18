# 🧭 What WAM Architecture Do We Need?

When designing a World Action Model, there's a question we can never quite avoid: **should video and action share a single backbone, or be modeled separately? And does inference really need to generate a future prediction at all?** This post draws on EasyWAM's full set of results on LIBERO / LIBERO-Plus for three representative architectures, laying the training regime (full-parameter / LoRA) and generalization ability (LIBERO-Plus) side by side, and walking through what the numbers say and the architectural reasons behind them. The results below cover Wan2.2-TI2V-5B and Cosmos-Predict2.5-2B; LoRA results are available for Wan2.2 on LIBERO.

## 🧱 Experimental Setup

- **EasyWAM-Unified** ([DreamZero](https://arxiv.org/pdf/2602.15922)-like): a single-DiT architecture that places video and action tokens into one Video DiT for joint denoising, with action and video bidirectionally coupled within the same self-attention.
- **EasyWAM-Hidden** ([DiT4DiT](https://arxiv.org/pdf/2603.10448)-like): a dual-DiT architecture that conditions a separate Action DiT on the Video DiT's intermediate features in one direction; inference still requires predicting future video.
- **EasyWAM-MoT** ([FastWAM](https://arxiv.org/pdf/2603.16666)-like): a dual-DiT architecture with an independent Video DiT and Action DiT interacting through shared mixed self-attention; at inference time it only predicts actions and does not generate or predict future video.

---

## RQ1: How do the architectures perform under full-parameter vs. LoRA training?

**Setup**: We put full-parameter training and LoRA (Rank 128) training results into the same table to see whether switching the training regime changes the relative ranking of the architectures.

**Results (LIBERO, Avg.)**

| Backbone | Model | Structure | Training | Spatial | Object | Goal | Long | **Avg.** |
| --- | --- | --- | --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2 | EasyWAM-Unified | Single-DiT | Full-parameter | 98.4 | 98.8 | 99.2 | 98.0 | **98.6** |
| Wan2.2 | EasyWAM-Hidden | Dual-DiT | Full-parameter | 99.2 | 100.0 | 97.8 | 98.2 | **98.8** |
| Wan2.2 | EasyWAM-MoT | Dual-DiT | Full-parameter | 97.0 | 99.2 | 96.6 | 94.0 | **96.7** |
| Wan2.2 | EasyWAM-Unified | Single-DiT | LoRA | 91.2 | 98.8 | 91.8 | 66.2 | **87.0** |
| Wan2.2 | EasyWAM-Hidden | Dual-DiT | LoRA | 98.0 | 99.8 | 89.4 | 86.6 | **93.5** |
| Wan2.2 | EasyWAM-MoT | Dual-DiT | LoRA | 96.8 | 99.6 | 97.4 | 90.0 | **95.9** |
| Cosmos2.5 | EasyWAM-Unified | Single-DiT | Full-parameter | 97.8 | 99.4 | 97.0 | 93.0 | **96.8** |
| Cosmos2.5 | EasyWAM-Hidden | Dual-DiT | Full-parameter | 97.2 | 98.8 | 96.0 | 95.0 | **96.8** |
| Cosmos2.5 | EasyWAM-MoT | Dual-DiT | Full-parameter | 98.0 | 98.4 | 98.4 | 95.6 | **97.6** |

**Findings**

- **Full-parameter rankings vary with the backbone.** On Wan2.2, Hidden (98.8) narrowly leads Unified (98.6), followed by MoT (96.7). On Cosmos2.5, MoT (97.6) leads Unified and Hidden (both 96.8). The close scores at the top and the change in ranking mean that these experiments do not identify one architecture as best across backbones.
- **Under Wan2.2 LoRA fine-tuning: MoT (95.9) > Hidden (93.5) > Unified (87.0).** Compared with full-parameter training, Unified drops 11.6 points, especially on LIBERO-10 (98.0 to 66.2); Hidden drops 5.3 points and MoT 0.8 points. This suggests different sensitivity to LoRA in the evaluated models, but the success rates alone do not show which part of each architecture causes the gap.

---

## RQ2: How well do the different architectures generalize?

**Setup**: [LIBERO-Plus](https://arxiv.org/pdf/2510.13626) takes checkpoints trained on LIBERO and systematically perturbs background, camera viewpoint, language instructions, object layout, lighting, noise, and robot embodiment, testing whether a model has actually learned a generalizable vision-action mapping. The three architectures follow different inference paradigms: MoT does not generate or predict future video at inference time, while both Hidden and Unified do.

**Results (LIBERO-Plus, Avg.)**

| Backbone | Model | Predicts video at inference? | Background | Camera | Language | Layout | Light | Noise | Robot | **Avg.** |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wan2.2 | EasyWAM-Unified | ✅ | 72.3 | 54.8 | 93.7 | 83.4 | 97.0 | 72.0 | 83.4 | **79.0** |
| Wan2.2 | EasyWAM-Hidden | ✅ | 59.3 | 57.0 | 93.2 | 84.3 | 95.0 | 70.8 | 83.7 | **77.6** |
| Wan2.2 | EasyWAM-MoT | ❌ | 64.5 | 45.5 | 71.4 | 80.1 | 94.7 | 78.5 | 71.5 | **71.7** |
| Cosmos2.5 | EasyWAM-Unified | ✅ | 72.7 | 63.7 | 90.1 | 82.6 | 89.2 | 72.1 | 79.2 | **78.2** |
| Cosmos2.5 | EasyWAM-Hidden | ✅ | 59.5 | 65.9 | 92.6 | 85.0 | 89.1 | 68.3 | 82.5 | **77.8** |
| Cosmos2.5 | EasyWAM-MoT | ❌ | 60.7 | 75.1 | 92.3 | 82.6 | 92.6 | 81.3 | 58.5 | **77.7** |

**Findings**

- **The generalization ranking changes with the backbone.** On Wan2.2, Unified (79.0) and Hidden (77.6) lead MoT (71.7). On Cosmos2.5, Unified (78.2), Hidden (77.8), and MoT (77.7) are close. Thus the Wan2.2 gap does not establish that predicting future video consistently improves LIBERO-Plus performance; these architectures differ in other ways as well.
- **Different perturbations favor different models.** On Cosmos2.5, MoT leads on Camera (75.1) and Noise (81.3), but scores 58.5 on Robot, behind Unified (79.2) and Hidden (82.5). Wan2.2 MoT also leads on Noise (78.5). Unified leads the overall score on both backbones, while Hidden no longer leads Unified overall. These results do not support attributing their differences to a video-action alignment tax without a controlled ablation.

---

EasyWAM is a WAM training infrastructure built and continuously evolved together with the community, and we welcome contributions of any kind — we'd love for more people to become contributors. 

If you have other questions about WAM training setups, feel free to open an [Issue](https://github.com/OpenMOSS/EasyWAM/issues) — we'll run targeted experiments and share reproducible analysis. You're also welcome to +1 existing issues; the ones that get more attention will be prioritized. And if you've run similar comparative experiments in your own setting, we'd love for you to share your findings in Issues too — we'll keep adding more models and benchmarks to this systematic evaluation.

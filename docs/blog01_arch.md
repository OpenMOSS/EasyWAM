# 🧭 What WAM Architecture Do We Need?

When designing a World Action Model, there's a question we can never quite avoid: **should video and action share a single backbone, or be modeled separately? And does inference really need to generate a future prediction at all?** This post draws on EasyWAM's full set of results on LIBERO / LIBERO-Plus for three representative architectures, laying the training regime (full-parameter / LoRA) and generalization ability (LIBERO-Plus) side by side, and walking through what the numbers say and the architectural reasons behind them. All figures below are from experiments on the Wan2.2-TI2V-5B backbone.

## 🧱 Experimental Setup

- **EasyWAM-Unified** ([DreamZero](https://arxiv.org/pdf/2602.15922)-like): a single-DiT architecture that places video, action, and robot-state tokens into one Video DiT for joint denoising, with action and video bidirectionally coupled within the same self-attention.
- **EasyWAM-Hidden** ([DiT4DiT](https://arxiv.org/pdf/2603.10448)-like): a dual-DiT architecture that conditions a separate Action DiT on the Video DiT's intermediate features in one direction; inference still requires predicting future video.
- **EasyWAM-MoT** ([FastWAM](https://arxiv.org/pdf/2603.16666)-like): a dual-DiT architecture with an independent Video DiT and Action DiT interacting through shared mixed self-attention; at inference time it only predicts actions and does not generate or predict future video.

---

## RQ1: How do the architectures perform under full-parameter vs. LoRA training?

**Setup**: We put full-parameter training and LoRA (Rank 128) training results into the same table to see whether switching the training regime changes the relative ranking of the architectures.

**Results (LIBERO, Avg.)**

| Model | Structure | Training | Spatial | Object | Goal | Long | **Avg.** |
| --- | --- | --- | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | Single-DiT | Full-parameter | 99.0 | 99.4 | 99.2 | 98.2 | **99.0** |
| EasyWAM-Hidden | Dual-DiT | Full-parameter | 99.4 | 100.0 | 97.0 | 97.8 | **98.6** |
| EasyWAM-MoT | Dual-DiT | Full-parameter | 97.8 | 98.4 | 97.6 | 95.6 | **97.4** |
| EasyWAM-Unified | Single-DiT | LoRA | 84.0 | 97.8 | 92.0 | 81.2 | **88.8** |
| EasyWAM-Hidden | Dual-DiT | LoRA | 96.8 | 99.4 | 92.6 | 86.8 | **93.9** |
| EasyWAM-MoT | Dual-DiT | LoRA | 96.8 | 98.8 | 94.4 | 90.4 | **95.1** |



**Findings**

- **Under full-parameter training: Unified (99.0) > Hidden (98.6) > MoT (97.4).** The single-DiT design places video and action tokens in the same self-attention, bidirectionally coupled — action prediction can draw directly on the full visual representation, and video generation is in turn constrained by the action signal. Learning this tight coupling well requires tuning the whole network jointly, which is exactly what full-parameter training provides, so Unified comes out on top in this setting.
- **Under LoRA fine-tuning: MoT (95.1) > Hidden (93.9) > Unified (88.8).** Both MoT and Hidden are dual-DiT architectures, which largely preserve the pretrained Video DiT prior; LoRA only needs to fine-tune a relatively independent Action DiT (plus a small amount of interaction layers) to adapt the action side, at a low cost. Unified's advantage under full-parameter training, by contrast, depends on jointly adjusting the *entire* backbone — once only a low-rank update is available, the "coupling capacity" that joint video-action modeling requires can no longer be captured by LoRA, and this shows up most starkly in performance dropping from 99.0 to 88.8.

---

## RQ2: How well do the different architectures generalize?

**Setup**: [LIBERO-Plus](https://arxiv.org/pdf/2510.13626) takes checkpoints trained on LIBERO and systematically perturbs background, camera viewpoint, language instructions, object layout, lighting, noise, and robot embodiment, testing whether a model has actually learned a generalizable vision-action mapping. The three architectures follow different inference paradigms: MoT does not generate or predict future video at inference time, while both Hidden and Unified do.

**Results (LIBERO-Plus, Avg.)**

| Model | Predicts video at inference? | Background | Camera | Language | Layout | Light | Noise | Robot | **Avg.** |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EasyWAM-Unified | ✅ | 55.8 | 33.7 | 93.7 | 80.6 | 92.2 | 50.2 | 71.4 | **67.5** |
| EasyWAM-Hidden | ✅ | 56.8 | 49.2 | 95.3 | 81.0 | 90.4 | 58.2 | 77.4 | **72.4** |
| EasyWAM-MoT | ❌ | 52.8 | 20.6 | 80.4 | 65.2 | 85.1 | 51.5 | 49.7 | **56.8** |

**Findings**

- **Predicting future video at inference time significantly improves generalization.** Hidden (72.4) and Unified (67.5) both predict future video at inference, and their average scores far exceed MoT (56.8), which doesn't. They lead on nearly every perturbation dimension, with the gap especially large on Camera (49.2/33.7 vs. 20.6) and Robot (77.4/71.4 vs. 49.7). This suggests that "predicting the future" itself provides a stronger supervisory signal than pure action imitation — the model has to model how objects and viewpoints evolve over time in order to generate a plausible future frame, and this process pushes it toward visual representations that are closer to the underlying physics, and more robust to perturbation, rather than simply memorizing the action-observation mapping from the training distribution.
- **Hidden generalizes better than Unified, at the cost of a "video-action alignment tax."** Both predict video at inference, but Hidden (72.4) clearly outperforms Unified (67.5). The reason is that Hidden's dual-DiT structure largely preserves the pretrained Video DiT's prior — the video branch runs essentially as it was pretrained, and only feeds its intermediate features to a separate Action DiT. Unified, on the other hand, has to pull action tokens into the same backbone for joint training, so the video branch's representation is continuously perturbed by action gradients — in effect, it pays a portion of its original pure-vision generalization ability as a tax for aligning the two modalities. This tax is most visible under out-of-distribution perturbation, especially Camera (33.7 vs. 49.2).

---

EasyWAM is a WAM training infrastructure built and continuously evolved together with the community, and we welcome contributions of any kind — we'd love for more people to become contributors. 

If you have other questions about WAM training setups, feel free to open an [Issue](https://github.com/OpenMOSS/EasyWAM/issues) — we'll run targeted experiments and share reproducible analysis. You're also welcome to +1 existing issues; the ones that get more attention will be prioritized. And if you've run similar comparative experiments in your own setting, we'd love for you to share your findings in Issues too — we'll keep adding more models and benchmarks to this systematic evaluation.

---
title: Part B（续）：TD-MPC 与规划机制对比
description: TD-MPC 的时序差分混合规划方案、与 DreamerV3 的对比，以及三种规划机制的综合总结。
lecture: 3
---

# Part B（续）：TD-MPC 与规划机制对比

## 机制三：TD-MPC，两者的桥梁

TD-MPC（Temporal Difference Model Predictive Control）[4] 同时拥有 MPC 的前瞻规划能力和 Actor-Critic 的时序差分学习效率。

**核心设计**：

| 组件 | 作用 |
|------|------|
| 潜在一致性损失 | 训练隐式动力学模型：$\hat{z}_{t+1} = f(z_t, a_t)$ 应与编码器输出的 $\text{sg}(z_{t+1})$ 一致 |
| 时序差分目标 | 用 Bellman 方程更新 Q 函数：$Q(z_t, a_t) = r_t + \gamma \cdot Q(z_{t+1}, \pi(z_{t+1}))$ |
| CEM 规划 | 在每步决策时，用 CEM 在潜在空间中搜索最优动作序列 |

**stop-gradient 的作用**：一致性损失中的 `sg(z_{t+1})` 表示停止梯度。如果编码器的两端都可以被梯度更新，模型可能学到一个"恒等函数"，把所有状态映射到同一个点，使得一致性损失为零，但毫无意义。stop-gradient 固定住目标端，防止这种模式坍塌（mode collapse）。

> **📖 Bellman 方程**：$Q(s_t, a_t) = r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a')$。把无限步的累积奖励问题，转化为只看"一步奖励 + 下一步 Q 值"的形式。**自举（bootstrapping）**：用模型自身的估计（如 $Q(s_{t+1}, a')$）来作为训练目标，"用自己预测自己"。TD 学习用 Bellman 方程做 bootstrapping，使学习可以在每步都进行，无需等到 episode 结束。

TD 学习用 Bellman 方程，以"当前奖励 + 下一步 Q 值估计"代替完整展开，将有效规划深度从"模型精确步数"缩短到"1步 + Q函数的自举"。

**与 DreamerV3 的对比**：

| 维度 | DreamerV3 | TD-MPC2 |
|------|-----------|---------|
| 世界模型形式 | 显式生成（重建像素/观测） | 隐式（只保证价值预测准确） |
| 规划方式 | 潜在空间 Actor-Critic | CEM + TD |
| 适用任务范围 | 视觉复杂任务，需要丰富观测 | 状态观测任务，高效连续控制 |
| 可解释性 | 可以可视化重建 | 潜在空间无直接语义 |

---

## 三种规划机制对比

| 维度 | CEM-MPC | Dreamer Actor-Critic | TD-MPC |
|------|---------|---------------------|--------|
| 规划方式 | 随机搜索 | 策略梯度（可微） | 随机搜索 + TD |
| 是否需要像素重建 | 否 | 是 | 否 |
| 长程规划能力 | 受 $H$ 限制 | 靠 Critic bootstrap | TD + MPC 结合 |
| 计算成本 | 高（大 $N$） | 中（想象展开） | 低-中 |
| 高维动作空间 | 效率低 | 梯度直接优化 | Q函数引导搜索 |
| 模型漏洞风险 | 中（短视） | 高（policy可钻漏洞） | 中（TD抑制累积误差） |
| 典型场景 | 简单连续控制 | 视觉复杂任务 | 高效连续控制 |

---

## 本讲小结

- **六个架构族**代表了突破 GRU 记忆瓶颈的不同方向：RNN/RSSM 计算最轻、Transformer 长程依赖最强、Diffusion 视觉最真实、JEPA 最专注语义、RWM 最专注部署稳定性、WAM 把世界预测和动作规划统一。
- **三种学习范式**决定模型的知识边界：观察型学视觉规律但无法控制，交互型学动作因果但数据昂贵，反事实型学价值推理但可解释性弱。WAM 代表第四范式：视频作为 dense physical supervision 联合训练世界和动作。
- **三种规划机制**决定模型如何被用于决策：CEM 最直白但在高维空间效率低，Actor-Critic 最优雅但有模型漏洞风险，TD-MPC 最务实地兼顾了两者。
- Dreamer = 交互型范式 + RSSM + 潜在 Actor-Critic，是本系列课程的核心参考系统。
- TD-MPC = 反事实型范式 + CEM + TD，将在 P04 中亲手实现并与 Dreamer 对比。

---

## 下一讲

构建和运行世界模型之后，下一个问题是：怎么判断它好不好？L04 针对每种架构给出专属的评估指标，Dreamer 的 FID 和奖励相关性、MuZero 的 MCTS 访问熵、TD-MPC 的潜在一致性损失、STORM 的长程 PSNR，以及一个所有模型都会遇到的通用失效模式：**horizon drift（时程漂移）**。

---

## 延伸阅读

本讲涉及的关键论文，按出现顺序排列：

- [Ha & Schmidhuber (2018) — World Models](https://arxiv.org/abs/1803.10122)：V/M/C 三模块框架与梦中训练的原始论文
- [Hafner et al. (2019) — Dreamer V1](https://arxiv.org/abs/1912.01603)、[V2 (2020)](https://arxiv.org/abs/2010.02193)、[V3 (2023)](https://arxiv.org/abs/2301.04104)：RSSM 与潜在 Actor-Critic 系列
- [Schrittwieser et al. (2020) — MuZero](https://arxiv.org/abs/1911.08265)：隐式世界模型 + MCTS，围棋与 Atari 超人水平
- [Hansen et al. (2022) — TD-MPC](https://arxiv.org/abs/2203.04955)、[TD-MPC2 (2024)](https://arxiv.org/abs/2310.16828)：CEM + TD 混合规划
- [Wu et al. (2022) — DayDreamer](https://arxiv.org/abs/2206.14176)：Dreamer 在真实机器人上的部署
- [Micheli et al. (2022) — IRIS](https://arxiv.org/abs/2209.14430)：VQ-VAE 离散化 + Transformer 自回归世界模型
- [Chen et al. (2023) — STORM](https://arxiv.org/abs/2310.09615)：随机 Transformer 世界模型，保留 RSSM 随机路径
- [Assran et al. (2023) — I-JEPA](https://arxiv.org/abs/2301.08243)、[Bardes et al. (2024) — V-JEPA](https://arxiv.org/abs/2404.08471)、[V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985)：语义空间预测架构系列
- [Bruce et al. (2024) — Genie](https://arxiv.org/abs/2402.15391)：可控交互式世界模型，从视频中学习潜在动作

---

## 参考文献

[1] Robine, J., Harter, M., Karwowski, J., & Tresp, V. [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2310.09615) (STORM). ICLR, 2023.

[2] Micheli, V., Alonso, E., & Fleuret, F. [Transformers are Sample Efficient World Models](https://arxiv.org/abs/2209.14430) (IRIS). ICLR, 2023.

[3] LeCun, Y. *A Path Towards Autonomous Machine Intelligence* — [见 L01 参考文献 [4]]

[4] Hansen, N., Su, H., & Wang, X. [TD-MPC: Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955). ICLR, 2022.

[5] Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., & Silver, D. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) (MuZero). Nature, 2020.

[6] Ho, J., Jain, A., & Abbeel, P. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS, 2020.

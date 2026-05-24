---
title: Part B：潜在动力学（GRU、MDN-RNN、RSSM）
description: 从 GRU 到 MDN-RNN 到 RSSM，理解如何在潜在空间中建模未来状态的动力学，以及先验与后验的分离设计。
lecture: 2
---

# Part B：潜在动力学

## 编码器不够用，我们需要预测未来

有了 VAE 编码器，我们能把当前帧 $\mathbf{o}_t$ 压缩为 $\mathbf{z}_t$。但世界模型的核心任务是**预测未来**：

> 在潜在空间中，给定当前状态 $\mathbf{z}_t$ 和动作 $\mathbf{a}_t$，预测下一时刻的状态 $\mathbf{z}_{t+1}$。

这个预测能力让智能体可以在脑海中"模拟"未来，从而在不真实执行动作的情况下规划决策，这正是世界模型节省样本的关键所在。

---

## 最简单的动力学：GRU

**门控循环单元（GRU，Gated Recurrent Unit）** 是序列建模的基础工具。作为动力学模型，GRU 接收 $(\mathbf{z}_t, \mathbf{a}_t)$ 并预测下一潜在状态：

$$
\mathbf{z}_{t+1} = \text{GRU}(\mathbf{z}_t, \mathbf{a}_t; \theta)
$$

> **📖 GRU 内部机制（简要）**：GRU 通过两个"门"控制信息流：**重置门（reset gate）**决定"遗忘多少过去"，**更新门（update gate）**决定"保留多少旧状态 vs 写入新信息"。门的值介于 0-1 之间，由当前输入和上一隐状态共同决定。这使 GRU 能够选择性地记住长期依赖，同时忘记无关信息，比普通 RNN 更擅长处理较长序列。相比 LSTM，GRU 少一个门（无单独的记忆细胞），参数更少，训练更快。

GRU 的优点是简单、训练稳定；缺点是输出确定性预测，无法表达**不确定性**。真实环境中，同一动作可能导致多种不同结果（例如：推箱子可能成功，也可能被卡住）。

---

## MDN-RNN：建模不确定性

**MDN-RNN（Mixture Density Network + RNN）** 在 Ha & Schmidhuber（2018）[见 L01 参考文献 [2]] 的 World Models 论文中提出，用**混合高斯分布**对下一状态的不确定性建模：

$$
p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{a}_t) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{z}_{t+1}; \mu_k, \sigma_k^2)
$$

- $K$ 个高斯分量，每个有自己的均值 $\mu_k$（分布中心）、方差 $\sigma_k^2$（分布宽度）
- **混合权重** $\pi_k$：第 $k$ 个高斯分量的概率质量，满足 $\sum_{k=1}^K \pi_k = 1$，$\pi_k \geq 0$。可以理解为"第 $k$ 种未来发生的概率"。网络输出 $\pi_k$ 后通过 softmax 函数归一化，确保所有权重之和为 1。

MDN-RNN 能捕捉**多峰分布**：环境可能跳到多个不同的下一状态，模型都能表达。

---

## RSSM：分离确定性与随机性

**RSSM（Recurrent State Space Model，循环状态空间模型）** 是 Dreamer 系列论文的核心创新，它将状态分为两部分：

- **确定性隐藏状态** $\mathbf{h}_t$：由 RNN 维护，汇聚历史轨迹信息，没有随机性
- **随机潜在状态** $\mathbf{z}_t$：从以 $\mathbf{h}_t$ 为条件的分布中采样，表达当前的不确定性

**RSSM 的核心方程**：

$$
\mathbf{h}_t = f_\phi(\mathbf{h}_{t-1},\ \mathbf{z}_{t-1},\ \mathbf{a}_{t-1})
\quad \text{（确定性更新，GRU/RNN）}
$$

$$
\mathbf{z}_t \sim p_\phi(\mathbf{z}_t \mid \mathbf{h}_t)
\quad \text{（先验 prior：不看真实观测，仅凭历史记忆 } h_t \text{ 猜测当前状态；用于纯想象/预测）}
$$

$$
\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t,\ \mathbf{o}_t)
\quad \text{（后验 posterior：在先验基础上，结合真实观测 } o_t \text{ 修正估计；训练时使用）}
$$

> **📖 先验 vs 后验**：这是贝叶斯统计的基本概念。**先验**（prior）是"看到数据之前的信念"，RSSM 依据历史记忆 $h_t$ 对当前状态 $z_t$ 的猜测。**后验**（posterior）是"看到数据之后更新的信念"，用真实观测 $o_t$ 修正先验，得到更准确的估计。训练时用后验产生 $z_t$ 并计算 KL 损失（衡量先验与后验的差距）；推理/想象时只有先验可用（没有真实 $o_t$），RSSM 纯靠先验向前滚动。

**为什么要分离？**

| 状态 | 角色 | 特性 |
|------|------|------|
| $\mathbf{h}_t$ | 记忆 | 确定性，汇聚历史 |
| $\mathbf{z}_t$ | 感知 | 随机性，表达不确定性 |

分离后，模型可以在没有真实观测的情况下，仅用先验 $p(\mathbf{z}_t | \mathbf{h}_t)$ 向前滚动，进行**纯想象中的规划**，这是 Dreamer 高效的根本原因。

PlaNet 论文（Hafner et al., ICML 2019）通过消融实验明确验证了这一设计：纯随机路径（无确定性 $h_t$）难以跨多步可靠地记忆信息，训练优化可能找不到"令某些维度方差趋零以储存长期信息"的解；纯确定性路径（无随机 $z_t$）无法表达环境固有的随机性，想象轨迹与真实轨迹的分布差距更大。**两条路径缺一不可。** 观测模型也因此同时以 $h_t$ 和 $z_t$ 为条件：$o_t \sim p(o_t | h_t, z_t)$，确定性记忆和随机感知共同决定重建的图像。

---
title: 架构模式、学习范式与规划方法
description: 以你在 P02 实现的 RSSM 为基线，横向比较七大世界模型架构族，掌握四种学习范式的知识边界，以及 CEM-MPC、潜在 Actor-Critic 与 TD-MPC 三种规划机制的原理与取舍。
lecture: 3
difficulty: 中高
---

# 架构模式、学习范式与规划方法

你在 P02 构建的 RSSM 就是 RNN 基线——本讲以它为起点，分两个部分展开：

- **Part A · 架构模式**：七大架构族（RNN/RSSM、Transformer、Diffusion、JEPA、RWM、Genie、WAM）的原理对比与选型指南，以及四种学习范式的知识边界
- **Part B · 规划机制**：CEM-MPC 随机搜索 → Dreamer 潜在 Actor-Critic → TD-MPC 混合方案，这是 P03 和 P04 的直接前置

架构部分不要求实现——那是后续项目的工作。重点是理解设计取舍，建立"什么情况下用什么"的判断。规划部分需要认真读。

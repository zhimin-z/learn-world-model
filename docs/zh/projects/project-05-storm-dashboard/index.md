---
title: P05：STORM + 三模型评估仪表盘
description: 把 RSSM 的 GRU 替换为 Transformer 实现 STORM，构建并排展示 Dreamer / TD-MPC / STORM 全部指标的评估仪表盘，并学会在真实部署场景下诊断世界模型。
---

# P05：STORM + 三模型评估仪表盘

**前置**：P03（Dreamer 流水线）、P04（TD-MPC 规划）、L03、L04

**交付物**：STORM 实现 + 覆盖三个模型的评估仪表盘 + 真实部署评测清单

---

## 任务概述

把 P02 的 RSSM 动力学模型从 GRU 替换为 Transformer（即 STORM 架构），然后构建一个并排展示 Dreamer / TD-MPC / STORM 所有核心指标的评估仪表盘。仪表盘应覆盖：

- **重建 FID**（Dreamer）
- **奖励相关性 ρ**（Dreamer）
- **一致性损失**（TD-MPC）
- **Token 预测损失 + 长时域 PSNR**（STORM）
- **FVD**（STORM + Diffusion）
- **可视化的潜在漂移曲线**（三模型共用）

---

## Part A：实现 STORM 动力学

STORM 把 RSSM 的 GRU 换成 Transformer 自注意力：历史帧编码为 token 序列，动作作为额外 token 拼入，Transformer 预测下一个 latent token。

```python
# 最小 STORM 骨架（替换 RSSM 的 recurrent_model）
class STORMDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, num_heads=4, num_layers=2, max_seq_len=50):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_seq, a_seq):
        # z_seq: (B, T, latent_dim), a_seq: (B, T, action_dim)
        T = z_seq.shape[1]
        pos = self.pos_emb(torch.arange(T, device=z_seq.device))
        x = z_seq + pos + self.action_proj(a_seq)
        x = self.transformer(x, mask=nn.Transformer.generate_square_subsequent_mask(T, device=x.device))
        return self.out_proj(x)  # (B, T, latent_dim)
```

**关键超参数**：
- `max_seq_len`：上下文窗口长度，建议从 16 开始，若 token 预测损失在长序列上明显恶化，缩短到 8
- `num_layers`：2-4 层足够 Atari 规模，深了反而过拟合短游戏序列
- 学习率：比 RSSM 低一档（如 `1e-4` → `3e-5`），Transformer 对学习率更敏感

---

## Part B：评估仪表盘

### 指标计算

| 指标 | 模型 | 计算位置 |
|------|------|---------|
| 重建 FID | Dreamer | `eval_dreamer.py: compute_fid()` |
| 奖励相关性 ρ | Dreamer | `eval_dreamer.py: reward_correlation()` |
| 一致性损失 | TD-MPC | `eval_tdmpc.py: consistency_loss()` |
| Token 预测损失 | STORM | `eval_storm.py: token_loss()` |
| 长时域 PSNR | STORM | `eval_storm.py: psnr_over_horizon()` |
| FVD | STORM / Diffusion | `eval_storm.py: compute_fvd()` |

### 漂移曲线

三个模型都需要跑漂移曲线：从一批真实起始状态出发，自回归展开 N 步，每步计算当前预测与真实状态的距离，画出距离随步数的折线图。

```python
def rollout_drift_curve(model, init_states, horizon=30):
    """返回每步的平均 PSNR 或余弦相似度"""
    metrics = []
    state = init_states
    for t in range(horizon):
        pred = model.imagine_step(state)
        real = get_real_state(t)
        metrics.append(compute_metric(pred, real))
        state = pred  # autoregressive
    return metrics
```

理想曲线：前 5-10 步近似线性下降后趋于平台。前 5 步骤降超过 30% 是危险信号。

---

## Part C：真实部署评测

P03 和 P04 的训练都在受控仿真环境里完成。当世界模型进入真实部署时，论文指标不再够用。

### 为什么 FID/PSNR 不够

FID/FVD/PSNR 衡量"预测准不准"，但回答不了：
- Policy 在世界模型里学到的动作，能否被真实硬件执行？
- 传感器延迟会不会让时序假设失效？
- 模型不确定时，系统能否识别并请求人工接管？

真实部署里 world model 只是链条中的一环：

```mermaid
flowchart LR
    A[传感器] --> B[状态估计] --> C[world model] --> D[planner/policy] --> E[低层控制] --> F[执行器]
```

### 部署时应记录的指标

**动力学质量**
- one-step prediction error：短期动力学是否准确
- multi-step rollout error：长程漂移（5/10/20 步）
- contact event accuracy：是否正确预测接触、滑动、掉落

**不确定性与可靠性**
- uncertainty calibration（ECE）：置信度是否与实际准确率对齐

> **📖 校准（calibration）**：模型预测"我有 80% 把握"时，真实准确率是否也接近 80%？ECE = 按置信度分桶后，各桶内置信度与实际准确率差值的加权均值，越低越好。

**策略迁移**
- policy transfer gap：模型里学到的策略迁移到真机后的累计奖励损失（sim-to-real gap）

**人机协作**
- intervention rate：每小时需要多少次人工接管
- failure recovery rate：失败中间态能否恢复

**系统性能**
- latency：从观测到动作是否满足控制频率（real-time factor ≥ 1）

### 七个常见坑

**动作语义不一致**：仿真里 `action` 是理想关节目标，真机要经过 PD 控制器、硬件限位、速度约束和电机响应延迟。用"理想动作"训练的 world model 描述的是一个不存在的完美机器人。

**时间延迟和异步传感器**：相机（30-60Hz）、力传感器（1kHz）、关节状态（250Hz+）、控制命令（可变频率）往往不同步。World model 以为 `o_t` 和 `a_t` 同时采集，实际可能差了几十毫秒。

**接触状态不可见**：视觉里看起来接触了，不代表力已经传上去。纯 RGB 输入的世界模型在抓取、插孔、拧瓶盖等接触密集任务上的预测上限远低于人类期望。

**长时序漂移**：短 rollout（1-5步）看起来好，长序列里物体身份、几何关系、接触状态都会悄悄变形。这是所有架构的共性问题，目前没有完全解决方案。

**Policy 利用模型漏洞（Model Exploitation）**：Policy 是优化器，它会找到世界模型里高奖励但真实世界不成立的动作。检测方法：定期把 policy 学到的高奖励动作序列放到真实环境执行，若"模型里有效但真机无效"的比例超过 20%，需要引入对抗训练或修补世界模型漏洞。

**不确定性没有进入控制决策**：世界模型给出预测但不告诉下游"这里我其实没见过类似状态"。一个简单实现：在潜在空间维护一个训练数据的密度估计器（KDE 或 normalizing flow），新观测密度低于阈值时触发高不确定性标志。

**安全不是 reward shaping 能完全解决的**：家庭和工厂机器人需要硬安全层（关节速度限制、末端执行器力限制、碰撞检测、急停、人工接管协议）。World model 可以做风险预测，但最终的硬安全保证必须来自独立的、不依赖学习的控制层。

### 三种递进落地方式

**1. 旁路评估器（Shadow Evaluator）**：真实 policy 照常执行，world model 在旁边独立预测，和真实结果对比，但不介入控制。用来建立可靠性地图（哪些状态下模型不可靠）。风险最低，适合部署早期。

**2. 动作候选过滤器（Action Filter）**：Policy 提出多个候选动作，world model 预测每个动作的后果，过滤掉预测后果危险或不确定性超阈值的候选，执行剩余中奖励最高的。

**3. 闭环规划器（Closed-loop Planner）**：World model 进入 MPC 的 rollout 或 imagined rollout，直接用于在线规划或 policy 的离线训练。这是 Dreamer 和 TD-MPC 的标准用法。收益最大，风险也最大。只在世界模型经过旁路评估器充分验证后才推荐使用。

---

## 总结对比表

| 模型 | 主要指标 | 常见失效模式 | 诊断方法 |
|------|---------|------------|---------|
| **Dreamer**（RSSM） | 重建 FID、奖励相关性 ρ | 编码器退化、想象奖励失真、KL 崩塌 | FID 上升 → 降低编码器 LR；ρ 下降 → 增大潜在维度；熵趋零 → KL 退火 |
| **MuZero**（隐式） | 价值准确度、MCTS 访问熵 | 价值估计偏差、假置信、表示不稳定 | 准确度低 → 重训奖励模型；稳定性低 → 增大网络宽度或加对比损失 |
| **TD-MPC**（潜在 MPC） | 潜在一致性损失、规划效率 | 表示崩塌、短视规划 | 去掉 stop-gradient 后损失更低 → 崩塌；效率低 → 增大 elite ratio |
| **STORM**（Transformer） | Token 预测损失、长时域 PSNR | Teacher forcing 差距、自回归漂移 | PSNR 骤降 → 缩短上下文窗口；调试用 PSNR，策略评估用 FVD |
| **扩散世界模型**（Diamond） | FVD、物理一致性、动作条件保真度 | 物体永久性丧失、3D 关系颠倒 | 深度违规率高 → 每层注入动作信息；保真度低 → 加深度一致性约束 |

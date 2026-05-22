---
title: "Part B: Planning Mechanisms"
description: CEM-MPC random search, Dreamer latent Actor-Critic, TD-MPC hybrid — three planning mechanisms, their mechanics, comparison, and trade-offs.
lecture: 3
---

# Part B: Planning Mechanisms

Given a world model, how does the agent use it to choose actions? This part is the direct prerequisite for P03 and P04. It introduces three planning mechanisms, from the most intuitive random search, through Dreamer's imagination training, to TD-MPC's hybrid.

There is one more paradigm that takes counterfactual reasoning to the extreme: the **counterfactual paradigm** — don't predict pixels, only make accurate predictions at the abstract level of value or reward. MuZero's (superhuman in Go, chess, Atari) implicit world model maintains three prediction heads:

| Head | Predicts | Role |
|--------|----------|------|
| reward head | Immediate reward $r_t$ | Evaluate the current step |
| value head | Future cumulative value $V(s_t)$ | Guide MCTS search direction |
| policy prior | Action probability $\pi(a \mid s_t)$ | Reduce MCTS branching |

As long as these three heads are accurate, the shape of the latent state $s_t$ doesn't matter — **for the agent, "faithfully reconstructing the world" is not necessarily the optimal goal**.

> **📖 MCTS (Monte Carlo Tree Search)**: starting from the current state, repeatedly perform four steps: ① **Selection**: descend the tree picking the node with the highest UCB score; ② **Expansion**: try a new action at the leaf node; ③ **Simulation/Evaluation**: estimate the new node's value with a neural network (MuZero uses the value head directly); ④ **Backpropagation**: propagate the value estimate back up the path. After hundreds of iterations, the most-visited action is "the action that survives sufficient search."

---

## Mechanism One: CEM Shooting MPC

> **📖 MPC (Model Predictive Control)**: at each time step, use the model to predict $H$ steps ahead, pick the best action sequence, execute only the first action, then replan at the next step. Even with an imperfect model, frequent replanning corrects errors before they accumulate.

**One-line description**: randomly sample a batch of action sequences, "imagine" their rollouts in the model, pick the sequences with the highest expected return, execute only the first action, and repeat.

**Algorithm**:

```
CEM-MPC planning loop (executed at every step)

Input: current state s_t, world model f, reward model r, horizon H, refinement rounds K

1. Initialize action distribution: μ ← 0, σ ← 1

2. FOR k = 1 to K (refinement):
   a. Sample N action sequences from N(μ, σ²): {a^(i)_{t:t+H}}
   b. FOR each sequence i:
        roll imagined trajectory: s^(i)_{t+1} = f(s_t, a^(i)_t), ..., s^(i)_{t+H}
        compute cumulative reward: R^(i) = Σ_{h=0}^{H-1} γ^h · r(s^(i)_{t+h}, a^(i)_{t+h})
   c. Select Top-K sequences (by R^(i) descending)
   d. Refit using Top-K: μ ← mean(Top-K), σ ← std(Top-K)

3. Execute the first action of μ: a_t ← μ[0]
```

The first round covers a wide range with low precision — it finds "roughly where the high-return region is." Later rounds refit the distribution with the elite sequences, narrowing the sampling toward high-return regions.

**Limitation**: random search is extremely inefficient in high-dimensional continuous action spaces (e.g. controlling all 7 joints of a robot arm simultaneously). This is the core problem TD-MPC addresses: guide the search with a Q function instead of blind sampling.

**Advantages**: simple, gradient-free, easy to implement; no differentiability requirement on the world model.

---

## Mechanism Two: Actor-Critic in Latent Space (Dreamer's Approach)

> **📖 Actor-Critic architecture**: two networks — the **Actor** (policy network $\pi_\theta(a|s)$) is in charge of "decisions," the **Critic** (value network $V_\phi(s)$) is in charge of "evaluation." The baseline provided by the Critic greatly reduces the variance of gradient estimation and makes training more stable.

Dreamer's core insight: instead of collecting massive data in the real environment to train the policy, train it in the **world model's imagined trajectories** — fast, risk-free, and differentiable.

**Training flow**:
1. **Imagination rollout**: starting from the current latent state $z_t$, the Actor samples actions and RSSM rolls forward $H$ steps.
2. **Critic estimates value**: compute $V(z_h)$ for each imagined state, use $\lambda$-return to construct training targets.
3. **Actor optimization**: the Actor maximizes the cumulative value predicted by the Critic via backprop (across the whole imagined trajectory).
4. **World model update**: update RSSM and encoder with real-environment data (reconstruction loss + KL).

**$\lambda$-return intuition**: pure Monte Carlo waits until the episode ends to get the true return — high variance; pure TD looks only one step — high bias. $\lambda$-return interpolates — construct $k$-step returns from "first $k$ real rewards + Critic estimate at step $k+1$" and weight-average over all $k$. $\lambda \to 1$ trusts the real rollout; $\lambda \to 0$ trusts the Critic.

**Why differentiability matters**: the Actor's gradient flows directly through RSSM's differentiable dynamics, far more accurate than Monte Carlo policy-gradient estimation.

**Model exploitation problem**: the policy may find actions with high reward in the model but invalid in reality — e.g. high-frequency jitter that scores well in the world model but only damages the motors on a real robot. Dreamer addresses this by periodically updating the world model with real-environment data and limiting imagination horizon length, but the problem is not fundamentally solved.

---

## Mechanism Three: TD-MPC — The Bridge

TD-MPC (Temporal Difference Model Predictive Control) combines MPC's look-ahead planning with Actor-Critic's TD-learning efficiency.

**Core design**:

| Component | Role |
|------|------|
| Latent consistency loss | Train an implicit dynamics model: $\hat{z}_{t+1} = f(z_t, a_t)$ should match $\text{sg}(z_{t+1})$ from the encoder |
| TD target | Update the Q function via Bellman: $Q(z_t, a_t) = r_t + \gamma \cdot Q(z_{t+1}, \pi(z_{t+1}))$ |
| CEM planning | At each decision step, run CEM in latent space to search for the best action sequence |

**Why stop-gradient**: the `sg(z_{t+1})` in the consistency loss stops the gradient. If both ends of the encoder can be updated, the model can learn an "identity function" — map every state to the same point, making the consistency loss zero but meaningless. Stop-gradient fixes the target side, preventing this mode collapse.

> **📖 Bellman equation**: $Q(s_t, a_t) = r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a')$. Converts the infinite-horizon cumulative reward problem into "one-step reward + Q at the next step." **Bootstrapping**: use the model's own estimates (e.g. $Q(s_{t+1}, a')$) as training targets — "predict yourself with yourself." TD learning uses Bellman to bootstrap, so updates happen at every step without waiting for episode end.

TD learning uses Bellman to substitute "current reward + next-step Q estimate" for a full rollout, shortening the effective planning depth from "the model's accurate horizon" to "1 step + Q bootstrap."

**Compared to DreamerV3**:

| Dimension | DreamerV3 | TD-MPC2 |
|------|-----------|---------|
| World-model form | Explicit generation (reconstruct pixels/observations) | Implicit (only value prediction must be accurate) |
| Planning method | Latent-space Actor-Critic | CEM + TD |
| Task scope | Visually complex tasks, requires rich observations | State-observation tasks, efficient continuous control |
| Interpretability | Reconstructions are visualizable | Latent space has no direct semantics |

---

## Three-Way Comparison

| Dimension | CEM-MPC | Dreamer Actor-Critic | TD-MPC |
|------|---------|---------------------|--------|
| Planning method | Random search | Policy gradient (differentiable) | Random search + TD |
| Needs pixel reconstruction? | No | Yes | No |
| Long-horizon planning | Capped by $H$ | Via Critic bootstrap | TD + MPC combined |
| Compute cost | High (large $N$) | Medium (imagination rollouts) | Low–medium |
| High-dim action space | Inefficient | Gradient-based optimization | Q-guided search |
| Model exploitation risk | Medium (myopic) | High (policy can exploit) | Medium (TD suppresses accumulation) |
| Typical use | Simple continuous control | Visually complex tasks | Efficient continuous control |

---

## Lecture Summary

- **The six architecture families** represent different ways to break the GRU memory bottleneck: RNN/RSSM is the lightest, Transformer has the strongest long-range deps, Diffusion has the highest visual fidelity, JEPA is the most semantic, RWM focuses on deployment stability, and WAM unifies world prediction and action planning.
- **The three learning paradigms** define the model's knowledge boundary: observational learns visual regularities but cannot control, interactive learns action causality but data is expensive, counterfactual learns value reasoning but is hard to interpret. WAM represents a fourth paradigm: jointly training world and action with video as dense physical supervision.
- **The three planning mechanisms** define how the model is used for decisions: CEM is most direct but inefficient in high dimensions, Actor-Critic is most elegant but has model-exploitation risk, TD-MPC most pragmatically combines the two.
- Dreamer = interactive paradigm + RSSM + latent Actor-Critic, the core reference system of this course.
- TD-MPC = counterfactual paradigm + CEM + TD, which you will implement in P04 and compare with Dreamer.

---

## Next Lecture

After building and running a world model, the next question is: how do we tell whether it is good? L04 gives each architecture its dedicated evaluation metrics — Dreamer's FID and reward correlation, MuZero's MCTS visit entropy, TD-MPC's latent consistency loss, STORM's long-horizon PSNR — and one universal failure mode all models face: **horizon drift**.

---

*References:*

- *Ha & Schmidhuber 2018 — World Models*
- *Hafner et al. 2019–2023 — Dreamer V1/V2/V3*
- *Hansen et al. 2022/2024 — TD-MPC / TD-MPC2*
- *Wu et al. 2022 — DayDreamer*
- *Schrittwieser et al. 2020 — MuZero*
- *Bruce et al. 2024 — Genie / Genie 2*
- *Chen et al. 2023 — STORM*
- *Micheli et al. 2022 — IRIS*
- *Assran et al. 2023 — I-JEPA*
- *Bardes et al. 2024/2025 — V-JEPA / V-JEPA 2*
- *DreamZero 2025 / Motus 2025 / DreamDojo 2026*

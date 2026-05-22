---
title: Metrics for Dreamer, MuZero, and TD-MPC
description: Dedicated evaluation metrics and diagnostic rules for the three architectures — RNN/RSSM, implicit world model, and latent MPC.
lecture: 4
---

# Metrics for Dreamer, MuZero, and TD-MPC

## Why "Evaluate by Model"?

**Counter-example**: scoring MuZero with FID. MuZero does not generate pixel images at all — its world model is implicit, and FID is meaningless. Likewise, scoring Dreamer with "token prediction loss" only misleads you into thinking Dreamer is a language model.

Different world models break in different places:

| Architecture | Most common failure point |
|---------|----------------|
| RNN/RSSM (Dreamer) | Encoder degeneration, imagined-reward distortion, KL collapse |
| Implicit (MuZero) | Biased value estimates, fragile representations, degenerate search trees |
| Latent MPC (TD-MPC) | Inconsistent latents, representation collapse, inefficient planning |
| Transformer dynamics (STORM) | Teacher-forcing/free-running gap, long-horizon token drift |
| Diffusion world model (Diamond) | Physics consistency collapse, objects appearing/vanishing |

---

## Dreamer (RNN/RSSM)

*In P03 you implemented the full Dreamer pipeline: encoder → RSSM prediction → latent Actor-Critic → action execution.*

### Reconstruction FID (Fréchet Inception Distance)

FID uses **Inception-v3** (an image classifier pretrained on ImageNet) to extract deep features from real and reconstructed frames, then computes the **Fréchet distance** between the two feature distributions. **Lower is better.**

> **📖 How FID is computed**: ① extract a feature vector from each of many real and generated images using an intermediate layer of Inception-v3; ② fit a multivariate Gaussian to each set of features (mean $\mu$, covariance $\Sigma$); ③ compute the Fréchet distance (a.k.a. Wasserstein-2 distance) between the two Gaussians — a "distance between distributions" that captures perceptual quality better than per-pixel MSE. Why Inception-v3? Its feature space approximates human perceptual judgment — two images close in this feature space also look close to the human eye.

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

- `μ_r, Σ_r`: mean vector and covariance matrix of real-image features
- `μ_g, Σ_g`: mean vector and covariance matrix of generated-image features
- `Tr(·)`: matrix trace (sum of diagonal entries)

**Diagnostic rule**: FID suddenly rises mid-training → the encoder is suffering representation collapse; conv weights have degenerated into a constant output. **Mitigation**: lower the encoder learning rate or add LayerNorm after the encoder.

**FID is necessary, not sufficient.** FID measures only generation quality, but Dreamer's true failure point is not always visual — sometimes the visual reconstruction is fine while the "imagined reward" has quietly drifted. That means the Actor-Critic is training inside a world that "looks right but has distorted reward signals," producing a policy that performs poorly in reality. So a normal FID is just the start; you must combine it with reward correlation.

### Reward Correlation

Dreamer rolls out trajectories in the "imagined space" and predicts rewards. These imagined rewards must correlate highly with the real rewards returned by the environment, or the Actor-Critic won't learn a useful policy.

$$\rho = \text{Pearson}(r_{\text{imagined}},\, r_{\text{real}})$$

In practice, take a batch of trajectories (e.g. 1000 steps), compute the Pearson correlation between imagined and real reward sequences, aim for `ρ ≥ 0.8`.

**Diagnostic rule**: `ρ` persistently below 0.5 → RSSM's stochastic state `z_t` is not encoding enough reward information; try increasing the latent dimension or extending the KL annealing schedule.

**Recommended experiment**: visualize imagined vs. real reward curves. From the same initial state, have Dreamer imagine forward 20 steps and execute the same action sequence in the real environment, then plot both reward curves on the same chart. If trends match (even if not exactly), the world model is honestly reflecting the environment; if the imagined reward stays higher than the real reward and the trends diverge, the "world model is lying" — the policy is learning tricks on a fake objective, exactly the root of model exploitation.

### Imagined Trajectory Entropy

An easily overlooked but very important early-warning indicator. RSSM's stochastic state `z_t` is sampled from a Gaussian; in principle it should have some variance at each step, reflecting environment randomness and model uncertainty.

$$H_{\text{traj}} = \mathbb{E}_t\!\left[H\!\left(q(z_t \mid h_t, o_t)\right)\right] = \mathbb{E}_t\!\left[\tfrac{1}{2}\sum_i \left(1 + \log \sigma^2_i\right)\right]$$

**Diagnostic rule**: if RSSM returns nearly identical `z_t` during imagination rollouts (all variances `σ²` near 0), the stochastic variable has degenerated into a deterministic one — an early sign of **KL collapse**. When KL collapses, the KL term in the loss becomes near-zero, the encoder stops injecting information into `z_t`, and RSSM degenerates into a pure RNN.

**Mitigations**:
- KL annealing: ramp the KL weight from 0 to its target slowly, giving the encoder time to learn reconstruction first and then learn to encode randomness.
- KL free bits: enforce a floor on the KL term so it cannot collapse too early.

---

## MuZero (Implicit World Model)

MuZero does not reconstruct pixels — its world model lives entirely inside the representation, dynamics, and prediction networks. Evaluating it means looking at **search quality** and **representation stability**.

### Value Accuracy

Before MCTS, the network outputs an initial value estimate `V₀` at the root; after search completes, the Q values weighted by visit counts give a refined estimate `V*`. Their mean squared error measures how accurate the network can be "without searching."

$$\text{ValueAcc} = 1 - \frac{\text{MSE}(V_0, V^*)}{\text{Var}(V^*)}$$

The closer to 1 the better. In a mature MuZero, `V₀` should be very close to `V*`, with search acting as "verification" rather than "correction."

**Diagnostic rule**: value accuracy persistently below 0.6 → the reward model (the `r̂` from the dynamics function) needs retraining, or the replay buffer holds too many out-of-distribution samples from old policies. Try increasing the priority replay weight so recent data is sampled more often.

### MCTS Visit Entropy

In the search tree, the visit counts `n_i` of child nodes form a distribution. The entropy is:

$$H = -\sum_i \frac{n_i}{N} \log \frac{n_i}{N}$$

**High entropy** → the model is uncertain across multiple actions, broad search; **low entropy** → the model is very confident in one action.

**Important context dependence**: high entropy is not necessarily bad. In stochastic games (e.g. opening positions in Go), positions objectively have many near-equal-value choices, and high entropy correctly reflects "multiple actions are valuable." Low entropy only signals healthy convergence in late game (when the position is decided). In early training or stochastic positions, abnormally low entropy means the model has formed bias too early — a signal of insufficient coverage, not convergence.

**Diagnostic rule**: persistent very low entropy combined with low value accuracy → "pseudo-confidence" — the model's judgment lacks real grounding. Usually need to add exploration noise (Dirichlet noise) or expand replay diversity.

### Representation Stability

A MuZero-specific diagnostic for the representation network's robustness.

$$\text{Stability} = \mathbb{E}_o\!\left[\cos\_\text{sim}(h(o),\, h(o + \varepsilon))\right]$$

> **📖 Cosine similarity**: measures the directional similarity of two vectors, independent of their magnitudes: $\cos\_sim(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}||\mathbf{v}|}$. Range $[-1, 1]$: 1 means same direction, 0 means orthogonal, -1 means opposite. We use it instead of Euclidean distance because the absolute length of representation vectors doesn't matter — what matters is whether they "point in the same direction" in high-dimensional space, which means the model assigns similar inputs similar semantics.

Add a tiny random perturbation `ε` (e.g. Gaussian noise on image observations with std ~1% of the pixel range) to a position `o`; the representation network's latent state should be very close. Target cosine similarity `> 0.95`.

**Why it matters**: if the representation is unstable, MCTS will make wildly different decisions on physically similar positions, causing the policy to behave erratically under small perturbations. For real-world robots or game AIs, this means the policy is overly sensitive to sensor noise and unreliable.

**Diagnostic rule**: stability below 0.9 → the representation network's training data lacks diversity in similar positions, or the network capacity is too small and produces nonlinear jumps on similar inputs. Widening the network or adding a contrastive loss (e.g. SimCLR-style positive pairs) usually helps.

---

## TD-MPC (Latent MPC)

*In P04 you implemented TD-MPC with CEM shooting and compared rewards against P03's Dreamer Actor-Critic.*

TD-MPC's core requirement: the encoder `h = enc(o)` must produce **consistent** representations across timesteps, otherwise MPC cannot plan effectively in latent space.

### Latent Consistency Loss

$$\mathcal{L}_{\text{consist}} = \|\text{sg}(h_{t+1}) - f(h_t, a_t)\|^2$$

where `f` is the dynamics function and `sg` is stop-gradient. This loss measures the distance between "the next-state prediction from the dynamics function" and "the latent encoded directly from the next observation."

**Why stop-gradient is critical**: `sg` blocks gradient flow at the target end, forcing the dynamics function `f` to actively "chase" the encoder, rather than letting the encoder degenerate to a trivial solution that minimizes the loss.

**Stop-gradient diagnostic**: if removing `sg` lowers the consistency loss, the model is learning an "identity mapping" — encoder maps all states to one point, dynamics outputs the same point, loss is zero, but it's meaningless. This is a form of **representation collapse** and one of TD-MPC's most dangerous failure modes. Formal diagnostic: on a validation set, check whether the encoded latent vectors have enough variance (the covariance rank should approach the latent dimension); if the covariance matrix's rank is near 1, collapse has occurred.

**Diagnostic rule (training oscillation)**: consistency loss oscillating (unstably going up and down) instead of monotonically decreasing → learning rate too high, or the encoder and dynamics function's gradient scales mismatched. Lower the learning rate by an order of magnitude, or give the encoder its own (smaller) learning rate.

### Latent Space Visualization

A powerful qualitative diagnostic that provides intuition beyond quantitative metrics.

**Method**: on a 2D continuous-control task (e.g. Pendulum or HalfCheetah), collect a batch of state-action trajectories, encode all observations into latent space, then project to 2D with t-SNE and visualize.

**A healthy TD-MPC latent space should satisfy**:
- "Physically close states" (e.g. similar pole angles) map to "nearby points in latent space" — local isomorphism.
- Trajectories from the same state under different actions move in consistent directions in latent space — action predictability.
- As rollout steps increase, trajectories move smoothly through latent space rather than jumping randomly.

**Diagnostic rule**: in the t-SNE plot, similar states (e.g. "pole upright") are scattered rather than clustered → the latent space's geometric structure is broken; MPC's planning in this space has no physical meaning.

### Plan Efficiency

Defined as: the number of MPC planning steps required to reach a **target reward threshold** (e.g. 80% of optimal) starting from a randomly initialized policy. Fewer steps means higher efficiency.

**Diagnostic rule**: low plan efficiency (many steps to converge) → CEM's elite ratio is set too low, or the planning horizon is too short, causing myopic planning to miss long-horizon rewards.

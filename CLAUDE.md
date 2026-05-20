# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

A VitePress documentation site for a World Models curriculum — 6 lectures + 5 projects teaching world models in AI/ML from first principles to full Dreamer/TD-MPC/STORM pipelines.

## Commands

```sh
npm install
npm run docs:dev        # Dev server with hot reload
npm run docs:build      # Production build
npm run docs:preview    # Preview built site
```

## Repository Structure

- `docs/` — VitePress site
- `docs/.vitepress/config.mts` — Nav/sidebar config (EN + ZH locales)
- `docs/zh/lectures/` — 6 Chinese lecture pages
- `docs/en/lectures/` — 6 English lecture pages (keep in sync with ZH)
- `docs/zh/projects/` / `docs/en/projects/` — 5 project pages
- `external/world-model-tutorial/` — PyTorch source code referenced by projects
- `external/world-model-tutorial/references.md` — 4-era history + architecture survey

## Writing Style

These rules apply to all lecture and project markdown files:

- **No AI flavor**: avoid "接下来我们…", "用大白话理解", "理解了X，我们可以更清楚地回答Y" — cut these transition phrases entirely
- **No `💡 直觉` callout boxes**: inline the content into surrounding prose instead
- **No "认知负荷：高/中/低" labels**: replace with one concrete sentence about what the reader needs before starting
- **No "下一讲预告" as a section title**: use "下一讲" and lead with the concrete problem the next lecture solves
- **Analogies**: avoid overused ones (飞行模拟器之于飞行员, 婴儿学走路, 撒网捕鱼, 雕塑家). Use engineering language or drop the analogy
- **`📖` definition callouts**: keep these — they define terms for deep-learning-only readers. Don't remove them
- **"读者思考" sections**: rename to "留给你" or similar; don't pad with three numbered questions if two suffice
- **Paragraph rhythm**: vary sentence length. If every paragraph is ~4 sentences, something is off

---

## World Models Curriculum

### Learning Objectives

By the end of this curriculum, students will be able to:
1. Explain what a world model is and why it reduces sample complexity — grounded in history and intuition (L01)
2. Implement a VAE encoder and chain it into an RSSM latent dynamics model (L02, P01, P02)
3. Compare 4 architectural families — using their own RSSM as the RNN baseline — and select the right one for a task (L03)
4. Describe three learning paradigms, implement CEM-MPC and latent actor-critic, and assemble a complete Dreamer pipeline (L04, P03, P04)
5. Select model-appropriate evaluation metrics, benchmark Dreamer vs TD-MPC vs STORM, and diagnose latent drift (L05, P05)

---

### Lectures

Each lecture lives in `docs/en/lectures/<slug>/index.md` and `docs/zh/lectures/<slug>/index.md`.

| # | Slug | Title | Core Concepts | Source |
|---|------|-------|---------------|--------|
| L01 | `lecture-01-internal-simulation` | Internal Simulation & Historical Context | Craik's mental models, predictive coding, 4 eras of WM evolution (1950s RNN → 2018 Ha&Schmidhuber → 2019 Dreamer → 2023 JEPA/Sora) | `references.md` §1 |
| L02 | `lecture-02-encode-and-dynamics` | Observation Encoding & Latent Dynamics | VAE → CNN encoder → ELBO. GRU → MDN-RNN → RSSM (deterministic + stochastic). Encoder as the bridge into Dreamer. | `tutorial/03-observation-encoder/`, `tutorial/04-latent-dynamics/` |
| L03 | `lecture-03-architecture-patterns` | Architecture Patterns & Tradeoffs | Student's RSSM as RNN baseline. Transformer (STORM/IRIS) → Diffusion (Diamond/Sora) → JEPA. Tradeoffs table. RWM + WAM as frontier. | `references.md` §2 |
| L04 | `lecture-04-learning-paradigms-and-planning` | Learning Paradigms & Planning | Part A: Observation-only / Interaction-based / Counterfactual paradigms. Part B: CEM-MPC → latent Actor-Critic → TD-MPC as bridge. | `references.md` §2, `tutorial/05-policy-learning/`, `tutorial/06-mpc-control/` |
| L05 | `lecture-05-evaluation-by-model` | Evaluation Metrics by World Model | Per-model metrics: Dreamer (FID, reward correlation), MuZero (value accuracy, visit entropy), TD-MPC (consistency loss, plan efficiency), STORM (token loss, PSNR), Diffusion (physics consistency). Horizon drift as universal failure mode. | `references.md` conclusion |
| L06 | `lecture-06-frontier-debates` | Frontier Debates: Language, Vision & the Boundary of the World | 5 open debates anchored in Xie Saining / LeCun / Sutton viewpoints. No answers given. | Xie Saining interview, LeCun 2022, Sutton Bitter Lesson |

---

### Projects

Source code in `external/world-model-tutorial/src/`. Reference chain: **Dreamer (RSSM)** → **TD-MPC** → **STORM**.

| # | Title | Prereq | Deliverable |
|---|-------|--------|-------------|
| P01 | Train a VAE Encoder | L01, L02 Part A | VAE compressing 64×64 → latent `z`; reconstruction loss curve; latent slider demo |
| P02 | Build a Latent Dynamics Model | P01, L02 Part B | GRU → RSSM predicting next latent; 1-step vs 5-step prediction error plot |
| P03 | Full Dreamer Pipeline | P02, L04 Part A | End-to-end: encode → RSSM → latent Actor-Critic → act; reward curve + FID/ρ/entropy self-eval |
| P04 | Implement TD-MPC Planning | P03, L04 Part B | CEM-MPC + latent consistency loss; compare vs Dreamer reward curve |
| P05 | STORM + Three-Model Evaluation Dashboard | P03, P04, L03, L05 | Swap GRU → Transformer (STORM-style); side-by-side dashboard for Dreamer/TD-MPC/STORM |

---

### Curriculum Flow

```
L01 (History + Intuition)
        ↓
L02 Part A (VAE Encoder) → P01 (Build & visualize VAE)
        ↓
L02 Part B (GRU → RSSM)  → P02 (Train dynamics, measure drift)
        ↓
L03 (Architecture Patterns — anchored to P02 RSSM baseline)
        ↓
L04 Part A (Paradigms: why)
        ↓
L04 Part B (Planning: how) → P03 (Full Dreamer pipeline)
                                    ↓
                             P04 (TD-MPC — compare vs P03)
                                    ↓
L05 (Evaluation vocabulary) → P05 (STORM + Three-Model Dashboard)
                                    ↓
                             L06 (Frontier Debates — no code)
```

**Suggested path**: L01 → L02 → P01 → P02 → L03 → L04 → P03 → P04 → L05 → P05 → L06

---

### Content Placement in VitePress

- Lecture pages: `docs/en/lectures/lecture-0N-<slug>/index.md` + `docs/zh/lectures/lecture-0N-<slug>/index.md`
- Project pages: `docs/en/projects/project-0N-<slug>/index.md` + `docs/zh/projects/project-0N-<slug>/index.md`
- World model landing: `docs/en/world-model/index.md` + `docs/zh/world-model/index.md`
- Sidebar group: `enWorldModelItems` / `zhWorldModelItems` arrays in `docs/.vitepress/config.mts`

### Reference Sources

- `external/world-model-tutorial/README.md` — tutorial structure, installation, demos
- `external/world-model-tutorial/references.md` — 4-era history, architecture taxonomy, application domains
- [liyang.page/wm-tutorial](https://liyang.page/wm-tutorial/) — primary external reference
- Pioneer papers: Ha & Schmidhuber 2018, Dreamer V1 (Hafner 2019), MuZero (DeepMind 2020), JEPA (LeCun 2023)

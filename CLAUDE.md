# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Learn Harness Engineering is a project-based course on building reliable coding environments for AI agents. The repo contains a VitePress documentation site plus hands-on project code.

## Commands

```sh
# Documentation site
npm install
npm run docs:dev        # Dev server with hot reload (VitePress)
npm run docs:build      # Production build
npm run docs:preview    # Preview built site

# Run lecture code examples
npx tsx docs/lectures/<lecture-dir>/code/<file>.ts

# Project Electron apps (from each project directory)
cd projects/project-NN/starter  # or solution/
npm install
npm run dev              # Build + launch Electron (via scripts/dev.js)
npm run check            # Type-check both tsconfig.json and tsconfig.node.json
npm run test             # Vitest run (single run)
npm run test:watch       # Vitest watch mode
```

## Repository Structure

- `docs/` — VitePress documentation site (lectures, projects, resources)
- `docs/.vitepress/config.mts` — Nav/sidebar config for both EN and ZH locales
- `docs/lectures/` — 12 lectures, each with `index.md` + `code/` examples
- `docs/projects/` — 6 project descriptions
- `docs/resources/` — Bilingual (en/zh) templates, references, OpenAI advanced pack

## Architecture

The course revolves around an Electron knowledge-base desktop app that evolves across 6 projects:
- **Main process** (`src/main/`): Window management, IPC handlers, service initialization
- **Preload** (`src/preload/`): contextBridge exposing typed API to renderer
- **Renderer** (`src/renderer/`): React UI with document list, Q&A panel, status bar
- **Services** (`src/services/`): DocumentService, IndexingService, QaService, PersistenceService
- **Shared types** (`src/shared/types.ts`): Cross-boundary interfaces and IPC channel constants

Each project's starter/solution is a complete copy of the Electron app at that evolutionary stage. P(N+1) starter is derived from P(N) solution. The shared foundation is in `projects/shared/`.

## Key Patterns

- IPC channels defined as constants in `src/shared/types.ts` (IPC_CHANNELS) — single source of truth
- All data stored locally as JSON/text files (no database)
- Mock Q&A returns structured answers with citations (no real LLM API)
- Harness files in project roots: AGENTS.md, CLAUDE.md, feature_list.json, init.sh, claude-progress.md
- Progressive disclosure: short AGENTS.md entrypoint linking to focused docs
- Each project has two tsconfigs: `tsconfig.json` (renderer) and `tsconfig.node.json` (main/preload)

## Bilingual Content

All content exists in both English and Chinese. Documentation lives in shared `docs/lectures/` and `docs/projects/` dirs (content is bilingual within each file). Resources have separate `docs/resources/en/` and `docs/resources/zh/` directories. Keep both in sync.

---

## World Models Curriculum

A code-first, project-driven curriculum teaching World Models in AI/ML. Based on `external/world-model-tutorial/` (hands-on PyTorch pipeline) and `external/world-model-tutorial/references.md` (historical + architectural survey).

### Learning Objectives

By the end of this curriculum, students will be able to:
1. Explain what a world model is and why it reduces sample complexity
2. Implement a VAE observation encoder and train it on image observations
3. Build and compare latent dynamics models (GRU, MDN-RNN, RSSM)
4. Understand 4 architectural families (RNN, Transformer, Diffusion, JEPA) and their tradeoffs
5. Apply MPC planning over a learned latent model
6. Evaluate trajectory quality for horizon drift, safety, and control utility

---

### Lectures (Theory + Code Walkthrough)

Each lecture lives in `docs/en/lectures/<slug>/index.md` and `docs/zh/lectures/<slug>/index.md`.

| # | Slug | Title | Core Concepts | Source |
|---|------|-------|---------------|--------|
| L01 | `lecture-01-internal-simulation` | Internal Simulation & Historical Context | Craik's mental models, predictive coding, 4 eras of WM evolution (1950s→2026) | `references.md` §1 |
| L02 | `lecture-02-observation-encoder` | Observation Encoder | VAE, CNN encoder, latent compression, ELBO loss | `tutorial/03-observation-encoder/` |
| L03 | `lecture-03-latent-dynamics` | Latent Dynamics Models | GRU, MDN-RNN, RSSM (deterministic + stochastic), $s_{t+1} = \mathcal{T}(s_t, a_t, h_t)$ | `tutorial/04-latent-dynamics/` |
| L04 | `lecture-04-architecture-patterns` | Architecture Patterns | RNN vs Transformer vs Diffusion vs JEPA — tradeoffs table | `references.md` §2 |
| L05 | `lecture-05-learning-paradigms` | Learning Paradigms | Observation-only (Sora), Interaction-based (Dreamer), Counterfactual (MuZero) | `references.md` §2 |
| L06 | `lecture-06-planning-and-control` | Planning & Control | MPC, CEM shooting, policy learning in latent space, actor-critic | `tutorial/05-policy-learning/`, `tutorial/06-mpc-control/` |
| L07 | `lecture-07-trajectory-evaluation` | Trajectory Evaluation | Horizon drift, safety violations, physics consistency, control utility | `references.md` conclusion |

---

### Projects (Hands-on Implementation)

Each project builds on the previous. Source code lives in `external/world-model-tutorial/src/`.

| # | Title | Deliverable | Key Skills |
|---|-------|-------------|------------|
| P01 | **Train a VAE Encoder** | Working VAE that compresses 64×64 images to latent `z`; reconstruction loss curve | `src/vae_model.py`, `demos/vae-visualizer.html` |
| P02 | **Build a Latent Dynamics Model** | GRU or MDN-RNN predicting next latent from $(z_t, a_t)$; compare prediction error | `src/` dynamics module |
| P03 | **Implement MPC Planning** | CEM-based MPC controller over the trained latent model; toy environment rollout | `src/mpc_controller.py`, `demos/mpc-controller.html` |
| P04 | **Full World Model Pipeline** | End-to-end: encode → predict → plan → act loop; training with `src/train.py` | `python -m src.train --demo` |
| P05 | **Architecture Comparison** | Benchmark RNN vs Transformer dynamics head on the same dataset; ablation table | Custom experiment |
| P06 | **Trajectory Evaluation Dashboard** | Visualize latent rollout drift, flag safety violations, score control utility | `demos/latent-dynamics.html` + custom metrics |

---

### Curriculum Flow

```
L01 (Why WMs?) → L02 (Encode) → P01 (Build VAE)
                                      ↓
L03 (Dynamics) → P02 (Train Dynamics)
                      ↓
L04 (Architectures) → L05 (Paradigms) → P03 (MPC) → P04 (Full Pipeline)
                                                           ↓
L06 (Planning) → L07 (Evaluation) → P05 (Ablation) → P06 (Dashboard)
```

**Suggested path for newcomers**: L01 → L02 → P01 → L03 → P02 → L06 → P03 → P04

---

### Content Placement in VitePress

- Lecture pages: `docs/en/lectures/lecture-0N-<slug>/index.md` + `docs/zh/lectures/lecture-0N-<slug>/index.md`
- Project pages: `docs/en/projects/project-0N-<slug>/index.md` + `docs/zh/projects/project-0N-<slug>/index.md`
- World model landing: `docs/en/world-model/index.md` + `docs/zh/world-model/index.md`
- Sidebar group: `enWorldModelItems` / `zhWorldModelItems` arrays in `docs/.vitepress/config.mts`
- Nav item: "World Models" → `/en/world-model/` and "世界模型" → `/zh/world-model/`

### Reference Sources

- `external/world-model-tutorial/README.md` — tutorial structure, installation, demos
- `external/world-model-tutorial/references.md` — 4-era history, architecture taxonomy, application domains
- Pioneer papers: Ha & Schmidhuber 2018, Dreamer V1 (Hafner 2019), MuZero (DeepMind 2020), JEPA (LeCun 2023)

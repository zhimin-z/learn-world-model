---
title: Architecture Patterns, Learning Paradigms, and Planning
description: Using the RSSM you built in P02 as the baseline, compare six world-model architecture families, three learning paradigms, and three planning mechanisms — CEM-MPC, latent Actor-Critic, and TD-MPC.
lecture: 3
difficulty: Intermediate-Advanced
---

# Architecture Patterns, Learning Paradigms, and Planning

The RSSM you built in P02 is the RNN baseline — this lecture starts from there and splits into two parts:

- **Part A · Architecture patterns**: comparison and selection guide for six architecture families (RNN/RSSM, Transformer, Diffusion, JEPA, RWM, WAM), plus the knowledge boundaries of three learning paradigms.
- **Part B · Planning**: CEM-MPC random search → Dreamer latent Actor-Critic → TD-MPC hybrid. This is the direct prerequisite for P03 and P04.

The architecture section does not require implementation — that is left for the projects. Focus on understanding the design trade-offs and building judgment for "which to use when." Read the planning section carefully.

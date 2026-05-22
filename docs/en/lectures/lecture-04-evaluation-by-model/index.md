---
title: "Evaluation by Model: Metrics and Diagnostics"
description: A systematic look at Dreamer-, MuZero-, TD-MPC-, STORM-, and diffusion-WM-specific evaluation metrics; understand horizon drift as the universal failure mode; learn real-deployment evaluation; diagnose world models by symptom.
lecture: 4
difficulty: Advanced
---

# L04 · Evaluation by Model: Metrics and Diagnostics

Many tutorials list a universal metric set — FID, PSNR, reward curves — and score all models with the same yardstick. It looks fair, but it hides each architecture's real failure modes.

**Core principle**: *metrics must be aligned with the architecture's failure modes.*

This lecture has three parts:

- **Per-model metrics**: Dreamer (FID + reward correlation), MuZero (value accuracy + visit entropy), TD-MPC (consistency loss), STORM (token loss + long-horizon PSNR), diffusion WM (physics consistency).
- **Universal failure mode**: horizon drift and mitigation strategies.
- **Real deployment**: beyond paper metrics — seven common pitfalls and three deployment strategies.

Read this after finishing P03–P05 — once you've seen your own numbers, many diagnostic rules click immediately.

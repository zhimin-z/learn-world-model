---
title: Observation Encoding and Latent Dynamics
description: Learn how to compress high-dimensional pixels into a compact latent representation, and how to model the dynamics of future states in that latent space.
lecture: 2
difficulty: Intermediate
---

# Lecture 2: Observation Encoding and Latent Dynamics

Dreamer's core question splits in two: **how to compress perception**, and **how to predict the future**. This lecture has two parts, each solving one of them.

- **Part A · Observation Encoding**: why compress at all? The encoder–decoder structure of the VAE, the intuition behind ELBO, and the CNN encoder.
- **Part B · Latent Dynamics**: from the simplest GRU, through MDN-RNN's uncertainty modeling, to RSSM's dual deterministic/stochastic design.

Read both parts before starting the projects. Part A is the direct prerequisite for P01; Part B is the direct prerequisite for P02.

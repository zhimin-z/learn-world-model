---
title: "Debates 3 & 4: AGI as Pseudo-Problem, and Will the Two Paths Converge?"
description: LeCun and Xie's challenge to "general intelligence," the chimpanzee world-model experiment, and whether LLMs and world models will end up at the same place.
lecture: 5
---

# Debates 3 & 4: AGI as Pseudo-Problem, and Will the Two Paths Converge?

## Debate 3: Is AGI a Goal, or a Pseudo-Problem?

### LeCun's Skepticism

Yann LeCun has a less-quoted argument on this; Xie restates the core in his interview:

A human has 2 million visual nerve fibers, but what we can actually process and see is close to zero — strictly bounded by consciousness and neural bandwidth. Outside the center of your field of view, you basically see nothing; your brain "fabricates" the illusion of a complete scene.

Human intelligence is a **very specialized intelligence**, not a general one. Humans cannot survive at low temperatures, cannot multiply three-digit numbers in their heads, cannot attend to several focal points at once. Our intelligence is what evolution optimized for a specific ecological niche.

The very definition of "general intelligence" is human-centric — we take our own intelligence as the standard for "general" and then ask "can machines reach human level?" But if human intelligence isn't general in the first place, the question itself is wrong.

Xie adds a perspective:

> "The evolution of intelligence is a continuum; humans are not unique."

He quotes Sutton (the same Sutton from the Bitter Lesson):

> "Building the intelligence of a squirrel — that's the truly hard problem."

### The Chimpanzee Experiment: Animals Have World Models

To support the "intelligence is a continuum" view, Xie cites a classic experiment from animal cognition:

The experimenter puts food in one of two boxes while a chimpanzee watches, then pulls the chimpanzee out of the room for a long time and brings it back. Key step: before returning, the chimpanzee sees the experimenter eating food from one of the boxes.

The chimpanzee's response: it walks straight to **the other** box, opens it, and takes out the food.

What does this mean? The chimpanzee maintained an internal model of the world's state while it wasn't present: it remembered which box had food, it understood "being eaten" means food disappears, and it inferred the other box still has food. This is a form of mental reasoning (Theory of Mind) — internal representation and inference about the world state.

Animals have world models. Intelligence is not a human monopoly. "General" may never have been a clear goal to begin with.

### Counter: Generalization Is the Essential Difference

AGI supporters reply that "general" is a matter of degree, not a binary. Humans' cross-domain generalization is unprecedented: you can learn any language, play any game, work in any profession, adapt to any culture. Squirrels can't write code, can't learn calculus, can't form social institutions. This cross-domain generalization is the essential feature of human intelligence and the real target of AGI.

Defining "general" as "squirrel level" is deliberately lowering the bar. If an AI system can achieve human-level cross-domain generalization, call it AGI or not — that's the goal.

### Over to You

If even squirrel-level intelligence is hard to fully replicate, is it possible we reach "general" before we ever truly understand it?

Or flip the question: how do you define "general"? Is there a definition that isn't human-centric?

---

## Debate 4: Are World Models and LLMs Competitors or Complements?

### Xie's Division of Labor

Xie is surprisingly moderate on this point. He does not say "LLMs are a dead end." He says:

> "Without LLMs, vision could not have expanded into the truly multimodal large-scale intelligence we see now."

His AMI Labs is not hostile to LLMs — he explicitly thanks LLMs for the progress they brought, because LLMs solved language understanding and instruction following, giving vision systems a "language interface" through which the world model can accept natural-language instructions.

His judgment: the two operate on different dimensions of information and each has its own stage:

| Paradigm | Domain |
|----------|--------|
| LLM | Reasoning in digital space, code, text generation, knowledge retrieval |
| World model | Prediction of the physical world, robot control, industrial perception, embodied AI |
| WAM | Unification: joint video + action modeling — understand physics and accept language instructions |

This "WAM" (World Action Model) concept deserves its own note. The core idea: jointly train video understanding and action prediction. Not just "understand the world," but "understand the world and know how to act in it."

If you organize existing world models by a matrix, with "consumes action signals" on the x-axis and "what it predicts" on the y-axis:
- **Dreamer**: takes actions, predicts latent state — narrow world model, active decision-maker.
- **WAM**: takes actions, also understands semantics — the unification.

### The Key Tension: Will They Eventually Converge?

Here's a question uncomfortable for both sides:

If world models eventually take language (accept natural-language instructions), and LLMs eventually take vision (process images and video), will the two ultimately converge into the same thing?

GPT-4o can see images, hear sound, generate images. V-JEPA 2 doesn't generate pixels — it predicts semantic representations — but it also needs language to describe tasks. Two paths are approaching the same point from two directions.

Maybe the "LLM vs world model" debate is really about **core design philosophy**, not the final product:
- Do you start from language and expand into physical understanding?
- Or start from physical perception and expand into a language interface?

Different starting points may produce fundamentally different system architectures, training data, and evaluation metrics — even if the end-state functions look similar.

### Over to You

V-JEPA 2 doesn't generate pixels; only predicts semantic representations. GPT-4o sees images and generates images. Do you think these two paths will still be separate in five years?

If they converge, who "wins" — the language camp or the world-model camp? Or is the question itself wrong?

---
title: "Debate 5: Where the Data Comes From, and Closing Remarks"
description: '"The era of downloading humanity": the data acquisition challenge for the physical world, AMI Labs'' data strategy, and the bet of those who are not believed.'
lecture: 5
---

# Debate 5: Where the Data Comes From, and Closing Remarks

## Debate 5: The Era of "Downloading Humanity" — Where Does the Data Come From?

### Saining Xie's Summary

Xie has a macro framing of the current stage of AI:

> "It used to be the era of downloading the internet. Now it's the era of downloading humanity."

The first stage, "downloading the internet," refers to LLM training data: Common Crawl, Wikipedia, GitHub, books — everything humans left as text in digital space. The data is enormous, but it has a fundamental limit: it has only language, no physics.

The second stage, "downloading humanity," refers to the data world models need: first-person human operation video, industrial sensor data, robot demonstration data, surgical video... This is data of how humans **act in the physical world**, not how they describe action.

Xie offers a startling order-of-magnitude comparison: a 4-year-old child, over their 4 years of life, has received more visual information — measured in frames and pixels — than all the text tokens used to train GPT-4 combined.

And crucially: this data won't be uploaded to YouTube. It lives in hospital operating rooms, factory floors, agricultural fields, home kitchens — scattered across countless private spaces, neither digitized nor labeled.

### AMI Labs' Data Strategy

Xie describes his team's response: a grassroots alliance. Find the companies that have real-world data — industrial manufacturing, healthcare, agriculture — and let them trade data for model capability, jointly building world models.

He draws an analogy from finance: Mastercard and Visa succeeded not because they had a lot of money themselves, but because they built a network all banks were willing to join. The world model data strategy may need similar network effects — no single company can accumulate enough physical-world data alone, but an alliance can.

### Deep Challenges

This vision carries severe challenges Xie does not dodge:

**Annotation cost is enormous**: labeling a single video of an industrial robot manipulation — joint states, object properties, action intent — is far more complex than labeling an image class.

**Privacy and ownership**: medical imagery has patient privacy, factory data has trade secrets, home video has personal privacy. Circulating this data requires solving thorny legal and ethical problems.

**Ethical boundaries**: "downloading humanity" is an unsettling phrase. Human behavior data, decision patterns, body movements — all to be used to train machines. Where does that boundary lie? Who decides?

### Over to You

If a world model ultimately requires sensor data from every person's daily life — your AR glasses, your home cameras, your work activity logs — would you trade that privacy for a smarter AI assistant?

A deeper question: is this choice really yours to "agree" or "refuse"? Or will it be like smartphones — if you don't participate, you're excluded — a piece of infrastructure?

---

## Closing: The Bet of Those Who Are Not Believed

Xie knows what he's doing isn't mainstream:

> "You can choose not to believe us — fine, let's see how it plays out. I've already gone all-in on this path. Are you in?"

Hinton in 2012 had this tone. Sutton sticking with reinforcement learning in 2016 had this tone — going all-in on a direction most don't believe.

LeCun's optimism is grander but aligned:

> "This is exactly what happened with deep learning and neural networks. There's always a small group that can clearly see the trajectory of the world."

Together these are both a manifesto and a risk statement. Historically, yes, a small group did get it right — but more often, small groups went all-in and never saw the turning point.

The world-model researchers' bet: language is not the substrate of thought; prediction and understanding of the physical world are the core of intelligence. If they're right, the next decade of AI's center will not be Silicon Valley's data centers but the sensor networks in factories, hospitals, and farms.

If they're wrong, the Scaling Law continues to work, and LLMs gradually approach physical understanding through more data and bigger models — not through world models, but through language.

---

## Three Questions to Take With You

**Question 1**: Do you think language is a "shortcut" or a "fast lane" for world models?

A "shortcut" means you take a road that avoids the real challenge and end up in a dead end; a "fast lane" means you take a more efficient road and still arrive at the same destination. The difference between the two determines where LLM's ceiling lies.

**Question 2**: If Sutton's Bitter Lesson is right, will world models have their own Bitter Lesson moment too?

Will some day, a "simpler, more general" method surpass the carefully designed world model architectures in one stroke — as AlphaZero surpassed Deep Blue, as Transformer surpassed LSTM? Will the Bitter Lesson itself have a Bitter Lesson?

**Question 3**: Is the world model the endpoint everyone arrives at, or one fork in the road?

Maybe the final answer isn't "who wins" but that different application scenarios take different technical paths: language generation and code assistance follow LLMs; robotics, industrial control, and autonomous driving follow world models; and some tasks — perhaps the ones you and I use most — live forever in the gray zone between.

If that's the case, the meaning of this debate is not in who wins, but in making clearer for us all: **what problem are we trying to solve, and where does the road we are on lead?**

---

## Further Reading and Resources

- **Saining Xie interview, *The Business Interviews*** (Zhang Xiaojun, 2024) — most of the first-hand views in this lecture come from here; strongly recommend listening to the full episode.
- **LeCun, Y. *A Path Towards Autonomous Machine Intelligence*** (2022) — LeCun's full argument for the Joint Embedding Predictive Architecture (JEPA), the world-model camp's manifesto.
- **Sutton, R. *The Bitter Lesson*** (2019) — under 2,000 words, one of the most contested short essays in recent AI; read the original.
- **Ha & Schmidhuber. *World Models*** (2018) — the founding paper of modern world-model research; first proposed the VAE + MDN-RNN + Controller architecture.
- **Frans de Waal. *Are We Smart Enough to Know How Smart Animals Are?*** — a classic popular book on animal cognition that will reshape your intuition for "intelligence."
- **Douglas Hofstadter. *Gödel, Escher, Bach*** — a philosophical magnum opus on intelligence, self-reference, and consciousness; the conceptual backbone for understanding why "general intelligence" is hard.
- **Robert Pirsig. *Zen and the Art of Motorcycle Maintenance*** — not about AI directly, but its reflections on "quality" and "understanding" resonate strangely with this lecture's core tension.

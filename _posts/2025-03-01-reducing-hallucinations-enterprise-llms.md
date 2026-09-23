---
layout: post
description: "How retrieval optimization, prompt engineering, and evaluation reduced enterprise LLM hallucinations without changing the underlying model."
title: Reducing Hallucinations by 60% Without Changing the Model
subtitle: Retrieval optimization, prompt engineering, and A/B testing for enterprise LLMs
---

When you build an AI product on top of a third-party LLM, you don't control the model weights. You control everything around it. That constraint turns out to be more of an advantage than a limitation.

After launching Slackbot, a context-aware conversational AI built into Slack, early beta feedback made one thing clear: accuracy was the top concern. For an enterprise workplace tool, this is existential. One bad answer erodes trust, and trust is hard to rebuild. We needed to systematically reduce hallucinations without waiting for the next model release.

We brought the hallucination rate from 12% down to around 5%. Here's how.

### Measuring what matters

You can't improve what you can't measure, and measuring hallucination is harder than it sounds. A hallucination isn't just a wrong answer. It's an answer that sounds right but isn't grounded in the source material.

We built a measurement pipeline:

- **Sample 500-1000 responses per week** from production traffic
- **LLM-as-judge for automated classification** across hallucination categories
- **Human calibration** on 100 samples per week, achieving 87% agreement with the automated judge
- **Track by category over time** to understand which types of hallucinations are improving and which aren't

The category breakdown matters. Simple factual Q&A has a hallucination rate of 1-5%. Complex reasoning and summarization can run as high as 30%. Treating "hallucination rate" as a single number hides the variance that actually drives your roadmap.

### The biggest lever: retrieval

Most people assume hallucinations come from the model making things up. In our case, 60% of hallucinations were retrieval failures: wrong messages retrieved, missing recent context, or irrelevant content diluting the signal. The model was faithfully synthesizing bad inputs.

We attacked this on multiple fronts:

**Hybrid search.** We combined semantic search (embeddings) with keyword search (BM25) to catch both conceptual matches and exact terms. Semantic search alone misses cases where a user asks about a specific project name or acronym. BM25 alone misses paraphrasing and intent.

**Query reformulation.** Before hitting the search index, we use an LLM to reformulate the user's query into a more search-friendly form. A question like "what did Sarah say about the Q3 launch timeline?" gets decomposed into targeted retrieval queries.

**Recency boosting.** If keywords like "recent" or "latest" appear, we boost messages from the last 7 days. This is simple but effective. Most workplace questions are about what happened recently, and retrieving a message from six months ago about a different project with a similar name is a common failure mode.

### Prompt engineering as a systematic practice

With retrieval fixed, about 30% of remaining hallucinations were model-side. Our baseline prompt was minimal: just context and a query. We hypothesized that explicit instructions would help, but we needed to test it rigorously.

We designed an A/B test for prompts. This is different from traditional A/B testing because LLM outputs are stochastic and we need to track multiple competing metrics.

**Experiment setup:**

- Randomization at user level (not request level) to ensure a consistent experience
- 50/50 split, 2,000 users per arm, 20,000+ responses per arm
- 2-week duration to account for novelty effects

**What we tested:**

The control prompt was basic: "Answer the user's question based on the messages provided." The treatment added three explicit instructions:

1. "For every claim, cite the source message with author and timestamp"
2. "Only use information from provided messages"
3. "If the answer isn't clear, say 'I couldn't find this'"

**Metrics framework:**

- Primary: hallucination rate (LLM-as-judge, calibrated against human review)
- Secondary: thumbs-up rate, regenerate rate, response length
- Guardrails: P99 latency < 5s, hallucination rate < 12% (auto-pause if violated)

We aggregated to user level to account for clustering (users average ~15 queries each) and used clustered standard errors for proper variance estimation. The primary metric was pre-registered to avoid multiple comparison issues.

**Results:**

- 25% relative decrease in hallucination rate
- 9% increase in P99 latency (200ms), within guardrail
- 10% increase in thumbs-up rate
- 33% decrease in regenerate rate

The latency cost was negligible for the quality gain. We rolled out globally.

### Post-processing: the last 20%

The final layer was output validation. We added checks for common failure patterns: responses that reference messages not in the provided context, claims without citations, and confident assertions about topics with no supporting evidence. These get caught and either reformulated or returned with an explicit uncertainty qualifier.

### The breakdown

Across our full effort, the improvement attribution was roughly:

- **45% from retrieval fixes** (getting the right context to the model)
- **30% from prompt engineering** (telling the model how to use the context)
- **20% from post-processing** (catching failures after generation)
- **5% from model version updates** (new releases during the period)

When you don't control the model, you control the input, the instructions, and the output validation. That's enough to cut hallucinations by more than half.

There's still a long way to go. Task-specific variance remains high, and the hardest cases involve complex multi-step reasoning where the model needs to synthesize across many messages. But the framework is in place: measure systematically, identify the failure category, fix the layer that's failing, and validate with rigorous experiments.

The takeaway for anyone building on top of LLMs: don't wait for a better model. Fix the pipeline.

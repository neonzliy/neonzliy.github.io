---
layout: post
description: "Evaluate AI products across quality, latency, cost, and user value with a measurement framework that connects model scores to product outcomes."
title: Evaluating AI Products Beyond Accuracy
subtitle: A practical measurement stack for quality, cost, latency, and product value
date: 2026-07-14 09:00:00 -0700
permalink: /2026-09-17-evaluating-ai-products-beyond-accuracy/
---

AI products are often evaluated as if the central question were simple:

> Did the model produce a good answer?

That question matters, but it is rarely enough.

A response can be accurate and still fail to help someone complete a task. It can be useful but arrive too slowly. It can increase engagement while creating more confusion. It can perform well in an evaluation set while producing weak results for the highest-volume use cases. It can improve a local quality score while increasing cost or reducing trust elsewhere in the product.

The hard part of evaluating AI products is not choosing a better score. It is connecting the score to the product outcome we actually care about.

I think about this as a measurement stack. The stack should connect business outcomes, user behavior, system performance, and component diagnostics. Each layer answers a different question. Together, they help us move from "the model looks good" to "the product is useful, reliable, and worth operating."

### The AI Product Evaluation Stack

A practical evaluation system has four layers:

```text
Business outcome
  ↓
User behavior and workflow completion
  ↓
System quality, latency, cost, and safety
  ↓
Component diagnostics and experiment inputs
```

The direction matters. We should begin with the outcome and work downward.

At the top, we might care about retention, revenue, productivity, task completion, or reduced support burden. These are product outcomes.

Below that are user behaviors that provide evidence of value. Did the user return? Did they complete the workflow? Did they accept or edit the result? Did they take the next action?

The third layer contains system-level properties such as answer quality, response time, cost per successful interaction, reliability, and safety.

The bottom layer contains the specific mechanisms that influence those properties. Examples include retrieval quality, context selection, ambiguity handling, tool selection, response classification, and prompt or model behavior.

A metric becomes useful when it has a clear place in this stack. It should also have a definition, a source, an owner, and a decision that it supports.

Without those connections, measurement becomes a collection of numbers rather than a system for learning.

### Start with the use case

Aggregate quality scores are attractive because they are easy to summarize. They are also easy to misunderstand.

An AI product usually serves several different use cases. Users may ask it to find information, summarize a document, answer a question, transform content, perform an action, or explain how the product works. These tasks have different failure modes and different expectations.

A system may perform well on writing assistance while struggling with retrieval. It may answer general questions effectively while failing when a user refers to a specific folder, file, or screen element. A single average can hide these differences.

The first step is to create a useful taxonomy of tasks. This does not need to be perfect. It needs to be stable enough to support comparison.

For each use case, measure at least four things:

1. How often the use case occurs.
2. How well the system performs.
3. How important the use case is to the product.
4. What kind of failure it creates.

This creates a prioritization model. A small use case with poor quality may not be the first place to invest. A high-volume use case with slightly below-target quality may represent a much larger opportunity.

The right question is not:

> What is the average quality score?

It is:

> Which use cases create the most value, and which failures most limit that value?

Use-case measurement also helps with evaluation-set design. If the production distribution is highly uneven, a balanced benchmark may not represent the actual product experience. We need both broad coverage and realistic weighting.

### Behavior is not the same as success

Behavioral signals are useful, but they are ambiguous.

A user who asks a follow-up question may be engaged. They may also be correcting a failed answer. A user who clicks a source may trust the response, or they may be checking whether it is wrong. A user who returns next week may have found durable value, or may simply be repeating an unresolved task.

This is why behavioral metrics should not be interpreted in isolation.

Consider multi-turn usage. It can indicate that the product supports a productive conversation. It can also indicate that the first response was incomplete, confusing, or poorly grounded. The same observed behavior can represent success or failure depending on the surrounding quality signal.

A stronger approach is to combine behavioral signals with outcome evidence:

- Did the user complete the intended workflow?
- Did they accept, edit, or discard the output?
- Did they take a meaningful next action?
- Was the result grounded in an appropriate source?
- Did the user need to repeat or reframe the request?
- Did the user return because the product was useful?

The goal is not to eliminate ambiguity from every signal. That is usually impossible. The goal is to make the ambiguity visible and reduce it through triangulation.

I prefer describing these metrics as evidence rather than truth. A click, a follow-up, or a return visit is a clue. It becomes stronger when multiple signals point in the same direction.

### Quality should be tied to the task

Quality evaluation works best when it reflects what success means for a specific use case.

For a retrieval task, relevant measures may include whether the correct source was found, whether the answer was supported by that source, and whether the user took the next action.

For a transformation task, we may care about whether the output preserved the original intent, followed the requested format, and required minimal editing.

For an action-taking system, answer quality is not enough. We need to know whether the correct action was selected, whether it executed successfully, and whether the user could recover from an error.

This leads to a layered quality model:

- **Task quality:** Did the response or action satisfy the user's goal?
- **Grounding quality:** Was the result based on the right information?
- **Interaction quality:** Did the system handle ambiguity and follow-up appropriately?
- **Execution quality:** Did tools or actions behave correctly?
- **Safety quality:** Did the system avoid unacceptable outcomes?

A model score can be useful at one layer while failing at another. For example, a response may sound convincing while relying on the wrong context. A tool call may be technically valid while performing the wrong action.

Evaluation should therefore be designed around the product task, not only the model output.

### Latency and cost are product constraints

Latency and cost are sometimes treated as infrastructure concerns that sit outside product quality. For AI systems, that separation is too artificial.

A response that arrives after the user has abandoned the workflow is not fully successful. A feature that produces excellent outputs at unsustainable cost is not ready to scale. A system that improves quality by adding several expensive calls may create a poor tradeoff for the product.

Latency should be measured at multiple levels:

- Time to first response
- Time to final response
- Time spent retrieving context
- Time spent calling tools
- Time spent in model inference
- Time spent coordinating multiple steps

The average is not enough. Tail latency often determines the experience of the users who encounter the most complex requests.

Cost also needs more than one lens. We can measure infrastructure cost, cost per request, cost per active user, or cost per successful workflow. The last measure is often the most meaningful because it connects operating expense to delivered value.

A cheaper response is not necessarily better. A more expensive response is not necessarily wasteful. The right comparison is whether additional cost produces enough improvement in quality, completion, trust, or another important outcome.

This is why cost and latency belong inside the evaluation framework. They shape whether the product can provide value consistently and sustainably.

### Instrument the journey, not just the event

Good evaluation depends on good instrumentation.

If we want to understand a workflow, we need to connect the events that make up that workflow. An isolated response event cannot tell us whether the user saw the result, clicked a source, completed an action, or abandoned the task.

Instrumentation should work backward from the product question.

Suppose the question is:

> What percentage of search sessions lead to a successful file action?

The system needs enough information to connect:

1. The search session.
2. The query or intent.
3. The results shown.
4. The result selected.
5. The file action attempted.
6. The terminal outcome.

This requires stable analytical containers and identifiers. A browser session, search session, conversation, and workflow are not interchangeable. Each represents a different unit of analysis.

Instrumentation should also distinguish exposure, interaction, and outcome. Seeing a recommendation is different from selecting it. Selecting it is different from completing the resulting action.

When these distinctions are missing, teams are forced to infer success from incomplete signals. That makes evaluation slower and creates disagreement about what the metrics mean.

A practical instrumentation review should ask:

- What product question is this event intended to answer?
- Which events belong to the same journey?
- What identifiers connect them?
- What properties are required for segmentation?
- What is the terminal state?
- Can the metric be validated independently?

The purpose of instrumentation is not to collect everything. It is to make important decisions measurable.

### Use guardrails to prevent local optimization

Any metric can be optimized in a way that damages the broader product.

Improving answer length might increase perceived helpfulness while increasing latency. Increasing tool use might improve task coverage while increasing cost. Increasing multi-turn conversations might appear to improve engagement while reflecting more failed first attempts.

Guardrails make these tradeoffs visible.

A quality metric might be paired with:

- Latency
- Cost
- Reliability
- User retention
- Workflow completion
- Safety incidents
- Negative feedback
- Unsuccessful action rate

The purpose of a guardrail is not to prevent change. It is to prevent us from declaring success too early.

A useful launch review should define the primary success metric and the conditions under which the launch should pause, roll back, or receive further investigation. These conditions should be defined before the results arrive.

Guardrails are particularly important for AI systems because behavior can change in unexpected ways. Users adapt to the product. They discover workarounds. They ask different kinds of questions. They may use a feature in contexts that were not represented in the original evaluation.

### Be disciplined about causal claims

Measurement can show that two things moved together. It does not automatically show that one caused the other.

If quality and retention rise at the same time, several explanations may be possible. The quality improvement may have caused the retention change. A different product change may have affected both. The user population may have changed. The measurement process may have changed.

Causal claims require a stronger design. Depending on the situation, that may involve randomized experiments, quasi-experimental methods, carefully constructed cohorts, or a credible comparison group.

Even then, we need to define the estimand clearly. Are we measuring the effect of a feature on all eligible users, exposed users, or users who actively used the feature? Are we measuring immediate behavior or durable outcomes?

A good analysis should state what the evidence supports and what it does not support.

For example:

- "The treatment group showed higher completion" is an observational result.
- "The feature caused higher completion" is a causal claim.
- "The result is consistent with a positive effect, but additional validation is needed" is a disciplined interpretation when evidence is incomplete.

This language may sound cautious, but it improves decision quality. It prevents teams from turning weak signals into strong narratives.

### An operational checklist

Before launching or scaling an AI product, I would ask:

#### Outcome

- What user or business outcome are we trying to improve?
- What does successful completion mean?
- What is the time horizon for value?

#### Use cases

- Which tasks make up most of the product experience?
- Are we measuring quality separately by use case?
- Which failure modes matter most?

#### Behavior

- Which signals indicate value?
- Which signals are ambiguous?
- What evidence distinguishes success from rework?

#### System performance

- What are the response-time distributions?
- What is the cost per request and per successful workflow?
- What reliability and safety constraints apply?

#### Instrumentation

- Can we connect exposure, interaction, and terminal outcomes?
- Are the relevant identifiers and properties present?
- Is there an independent way to validate the metric?

#### Experimentation

- What is the comparison group?
- What potential confounders exist?
- What claim can the design actually support?

#### Guardrails

- Which metrics must not regress?
- What would trigger a pause or rollback?
- Who owns the decision?

The best evaluation systems do not produce one definitive score. They create a shared language for deciding what to build, what to fix, and when to stop.

AI products are not only model outputs. They are workflows operating under constraints. Their value depends on whether users can accomplish meaningful goals with a level of quality, speed, cost, and trust that can be sustained.

That is the standard I would use: evaluate the model, measure the system, understand the user journey, and connect all of it back to the outcome.

---
layout: post
description: "Design coding-agent workflows around exit criteria, evaluation, context, and human review to turn plausible answers into completed work."
title: Claude Code Is a Workflow, Not a Chat Box
subtitle: What sustained agentic work taught me about exit criteria, evaluation, context, and review
date: 2026-09-09 09:00:00 -0700
permalink: /2026-09-17-claude-code-is-a-workflow-not-a-chat-box/
---

The least useful way to evaluate a coding agent is to ask whether it produced an answer.

Answers are easy. Completed work is harder.

A plausible patch can still misunderstand the system. A passing test can miss the behavior that matters. A confident explanation can rest on a capability the tool does not actually have. An agent can remain busy for a long time without moving the outcome closer to done.

After using Claude Code across recurring analytical, product, and engineering workflows, I stopped thinking of it primarily as a chat interface. I began treating it as an operating environment.

That shift changes the design question. The goal is no longer to write the cleverest prompt. It is to create a workflow in which the agent can discover the right context, act within clear boundaries, produce inspectable evidence, and know when to stop.

The model still matters. But reliability increasingly comes from the system around it.

### The misleading unit of progress

Conversational interfaces encourage us to measure progress in turns. We ask for something, receive a response, and decide whether to ask again.

That is a reasonable interaction model for brainstorming. It is a weak operating model for consequential work.

In a real task, the unit of progress is not the message. It is the state change.

Did the relevant behavior improve? Did the analysis answer the decision question? Did the test exercise the failure mode? Can another person inspect the result? Is the work safe to use?

This distinction sounds obvious, but it changes how an agent behaves. If success is defined as producing code, the agent will optimize for code. If success is defined as demonstrating a verified outcome, code becomes only one possible step.

The same principle applies outside software. An analysis is not complete because a query returned rows. A document is not complete because it has paragraphs. A workflow is not successful because it ran without an exception.

The output is evidence of activity. Completion requires evidence of the intended outcome.

### Write the exit condition first

The most effective improvement I have found is to define completion before the agent starts.

An exit condition should describe what must be true, not what the agent must produce.

Weak exit conditions look like this:

- Write the implementation.
- Analyze the metric.
- Create the document.
- Fix the bug.

Stronger exit conditions look like this:

- The user-visible behavior works in the affected states, and the previous behavior still works elsewhere.
- The metric is calculated from a verified population, agrees with an independent check, and is labeled with its remaining uncertainty.
- The document answers the intended decision, names its evidence, and separates facts from recommendations.
- The failure can be reproduced before the change, cannot be reproduced after it, and a regression check protects the boundary.

Good exit conditions change the agent's search strategy. They create a reason to inspect existing behavior, test assumptions, and gather evidence rather than stopping at the first plausible artifact.

They also make handoff easier. A reviewer does not need to infer what "done" meant after the work is complete. The standard was visible from the beginning.

### Interrogate the harness

Agents operate through a harness: commands, tools, permissions, context rules, connectors, and interface conventions. Many failures begin when we assume the harness behaves differently from how it actually works.

Before designing a new workflow, I now verify the environment.

What can the tool already do? Which commands exist? Which sources are connected? What can be changed, and what is read-only? Which events or logs are trustworthy? Where does context come from? What survives between sessions?

This often prevents unnecessary work. A capability that appears missing may already exist under a different interface. A proposed automation may duplicate a built-in behavior. A metric may already be available, but its name or semantics may differ from the assumption behind the plan.

The lesson is broader than Claude Code. Never build a system around a presumed gap until the gap has been verified.

There is also a product-design implication. The harness is part of the user experience. A powerful model with unclear permissions, inconsistent context, or invisible state will feel unreliable. A more constrained model with legible tools and predictable boundaries can be far more useful.

### Use a rubric before the run

It is difficult to evaluate an output fairly after seeing it.

Once a polished result exists, people tend to rationalize its weaknesses. The prose sounds good. The code looks substantial. The diagram feels coherent. Standards that were vague at the beginning become flexible at the end.

A pre-written rubric reduces that bias.

For agentic work, I usually want criteria across several dimensions:

- **Correctness:** Does the result satisfy the actual task?
- **Evidence:** What demonstrates that it works?
- **Scope:** Did the work stay inside the authorized boundary?
- **Safety:** Were destructive or sensitive actions handled appropriately?
- **Maintainability:** Can another person understand and modify the result?
- **Efficiency:** Was unnecessary complexity, cost, or context avoided?
- **Handoff:** Are the remaining uncertainty and next action clear?

The rubric should match the risk of the task. A small copy edit needs very little ceremony. A change that affects data access, external communication, or shared infrastructure deserves a stronger gate.

The purpose is not to turn every task into a process. It is to make the quality standard explicit before the output can influence it.

### Make review adversarial

Review is most valuable when it tries to disprove the claim.

An agent that created a plan or implementation already has a narrative about why it should work. Asking the same context to "double-check" often produces confirmation rather than scrutiny.

For material work, I prefer a fresh reviewer with a narrow contract:

> Attempt to refute the claimed outcome. Do not fix the work. Return the strongest evidence for and against it.

This framing has caught problems that normal review missed: an acceptance condition that contradicted the intended behavior, a publication flow that did not preserve meaningful consent, and an apparently useful automation whose premise had not been verified.

The value is not that a second agent is automatically correct. The value is independence. A fresh context is less invested in the path that produced the result.

Adversarial review is especially useful before implementation. Finding a broken assumption in a plan is much cheaper than discovering it after code, documentation, and dependencies have accumulated around it.

### Control context deliberately

More context is not always better context.

Large instruction files and reference collections can create the impression of completeness while making the important rules harder to find. They consume attention on every task, including tasks that do not need most of the material.

I have found a layered context model more effective:

1. A small, always-loaded core containing identity, safety rules, and operating principles.
2. Task-specific guidance loaded only when the work enters that domain.
3. Detailed references retrieved when a concrete question requires them.
4. Durable lessons written back only when they are likely to change future behavior.

This is similar to information architecture in a product. The most important content should be visible at the moment it becomes relevant. Everything else should be discoverable without competing for attention.

Selective loading also improves maintenance. A focused reference can have a clear owner and purpose. A giant context file tends to collect stale instructions because removing anything feels risky.

Context should earn its place. If a rule never changes a decision, it may not belong in the always-loaded layer.

### Keep a compact decision trace

Agentic workflows need memory, but not every intermediate thought needs to become permanent.

What matters is a concise decision trace:

- What question was being answered?
- What evidence changed the decision?
- What uncertainty remains?
- What action was taken?
- Who owns the next step?

This trace is different from a transcript. A transcript records activity. A decision trace records the information another person needs to understand and challenge the outcome.

The distinction is important for both usability and privacy. Storing every exploratory branch creates noise and may preserve information that never needed to leave the working context. Storing no trace makes the result difficult to audit or improve.

A compact record gives the next person a place to begin without forcing them to reconstruct the entire session.

This is also where attribution belongs. If an individual workflow becomes a shared capability, the contributor should be able to review the exact artifact, understand how it will be presented, and approve the version that others will use.

Automating discovery is helpful. Automating consent is not.

### Retire what does not earn its keep

Agentic systems accumulate quickly. Prompts become skills. Skills become templates. Templates gain wrappers, monitoring, and documentation. Soon, maintaining the workflow costs more than running it ever saved.

Every reusable artifact should therefore have a path to retirement.

Useful questions include:

- Is anyone using it repeatedly?
- Does it still solve the original problem?
- Does it require constant support?
- Has the underlying tool made it redundant?
- Is the owner still willing to maintain it?
- Does its value justify its context, compute, and review cost?

Process liveness is not outcome health. A scheduled job can keep running long after its result has stopped mattering. An agent can complete every step in a workflow that should no longer exist.

Sunsetting is part of design. It keeps the system legible and directs attention toward the workflows that continue to create value.

### From better prompts to better environments

Prompt quality still matters. Clear language, examples, constraints, and role definitions can improve a run.

But the largest gains I have seen come from environment design:

- Define the outcome before the work begins.
- Verify the harness before proposing new infrastructure.
- Pre-commit the evaluation criteria.
- Separate creation from adversarial review.
- Load context when it becomes relevant.
- Preserve decisions without preserving unnecessary reasoning.
- Require human approval where publication, attribution, or accountability changes.
- Retire workflows that do not become useful habits.

These practices make an agent less magical and more dependable.

That is a worthwhile trade. The point of an agentic workflow is not to create the feeling that a model can do anything. It is to help people complete real work with a level of evidence, control, and clarity they can trust.

Claude Code becomes much more powerful when it stops being treated as a place to ask for answers and starts being designed as a system for reaching verified outcomes.

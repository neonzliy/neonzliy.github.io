---
layout: post
description: "Validate decision models by testing assumptions, separating correlation from causation, and making uncertainty visible before acting on results."
title: Trustworthy Decision Systems Begin With Model Skepticism
subtitle: How to validate signals, separate mechanisms from artifacts, and make uncertainty useful
date: 2026-03-18 09:00:00 -0700
permalink: /2026-09-17-trustworthy-decision-systems-begin-with-model-skepticism/
---

I have learned to distrust models that look impressive too quickly.

A strong coefficient, a polished dashboard, or a compelling narrative can create the feeling that we understand a system. Often, we understand only one slice of it. The result may be driven by a shared time trend, a mechanical relationship in the data, a population mismatch, or a field that means something different from what we assumed.

This is not an argument against models. It is an argument for taking the surrounding decision system as seriously as the model itself.

A trustworthy decision system makes its assumptions visible. It distinguishes observation from explanation, correlation from causation, and a useful hypothesis from an operational recommendation. It gives people a way to challenge the result before the result changes a product, allocates resources, or becomes part of an automated workflow.

That discipline has become even more important as teams adopt AI-enabled tools. AI can accelerate analysis and make sophisticated workflows available to more people. It can also accelerate a wrong interpretation, hide an unsupported assumption, or turn an unreviewed artifact into a shared dependency.

The goal is not to eliminate uncertainty. The goal is to make uncertainty useful.

### The seductive failure mode

The most dangerous analytical failures are rarely obvious. They usually look like progress.

A model fits historical data. A metric moves in the expected direction. A dashboard updates successfully. An AI-generated summary sounds coherent. Each of these can be true while the underlying decision is still unsupported.

One recurring pattern is a model that explains the past but does not identify a reliable lever. Several variables may rise together because of seasonality, product growth, a policy change, or a common external event. The model attributes the shared movement to a particular feature, even though changing that feature may not produce the expected result.

Another pattern is mechanical correlation. A variable may be mathematically downstream from the outcome it is supposed to explain. Including both in a regression can produce a very strong relationship that says little about what would happen if a team changed the proposed driver.

A third pattern is population mismatch. A metric may be calculated for one group while the decision concerns another. A query can run without error and still answer a different question.

These failures are seductive because they are often accompanied by real technical work. The query ran. The chart rendered. The model converged. The work is not wasted, but the conclusion needs to change.

A failed validation is evidence. It tells us which story the data cannot support.

### Start with the decision, not the model

Before selecting a method, I try to write down the decision the analysis is meant to inform.

That means specifying:

- What action might change?
- Who or what is affected?
- What intervention is under consideration?
- What outcome matters?
- Over what time period?
- What level of uncertainty is acceptable?

This step prevents a common mistake: treating a descriptive relationship as an intervention recommendation.

For example, "users who do X tend to have higher retention" is descriptive. "Increasing X will improve retention" is causal. The second claim requires a stronger design, such as an experiment, a credible quasi-experiment, or a carefully justified natural experiment.

The distinction matters even when the recommendation feels obvious. A feature can be associated with successful users because it helps them succeed. It can also be used more often by users who were already likely to succeed. The same observation supports two very different explanations.

I find it useful to label conclusions explicitly:

- **Observed:** Directly measured pattern.
- **Associated:** Relationship that persists after basic controls.
- **Candidate mechanism:** Plausible explanation supported by additional evidence.
- **Causal evidence:** Result from a design that identifies the effect of an intervention.
- **Operational recommendation:** Action justified by the strength and relevance of the evidence.

These labels make it harder for a tentative finding to become a definitive sentence during handoff.

### Validate the substrate

Model validation begins before the model. It begins with the data substrate.

I want to know what each field means, how it is produced, and at what grain it is defined. I verify names, timestamps, joins, filters, population definitions, and event semantics before building a narrative around the output.

A schema check is not administrative overhead. It is part of analytical correctness.

A query can fail loudly because a column does not exist. That is easy to catch. More dangerous is a query that succeeds against a similarly named field with a subtly different meaning. The output may look plausible while answering the wrong question.

The same applies to joins. Joining a user-day table to a user-week table without carefully handling grain can multiply records. Joining a population table on user ID without the corresponding date can attach the wrong eligibility state to an observation. Neither error necessarily produces a database exception.

My minimum validation questions are:

1. What is one row?
2. What is the authoritative source?
3. Which date controls eligibility?
4. Are the joins one-to-one, one-to-many, or many-to-many?
5. Which records are excluded, and why?
6. Are the fields measured before or after the outcome?
7. Does the query return the expected magnitude and coverage?

I also compare the result with an independent source or a simpler calculation. If two paths disagree, that disagreement is more informative than a single polished number.

### Measure distance from the mechanism

Not all predictors are equally useful.

A variable close to the decision or value event may be easier to interpret than a variable several process steps upstream. This is not a universal law, but it is a useful diagnostic.

Suppose a business outcome is produced by a cascade of steps. Some variables are near the final value moment. Others influence intermediate states, and still others are broad environmental indicators. A distant variable may correlate strongly with the outcome because it shares the same trend. It may also be difficult to change or impossible to attribute.

I call this **mechanism distance**.

The closer a variable is to the intervention and outcome, the more carefully I ask whether it represents a mechanism, a consequence, or a proxy. A downstream variable may be highly predictive but unusable as a lever because it is already part of the outcome pathway. An upstream variable may be actionable but too noisy to support a decision.

A useful analysis maps each candidate variable onto the process:

- What event does it represent?
- Where does it sit in the causal or operational chain?
- Can the proposed intervention change it?
- Is it a cause, a consequence, or a shared indicator?
- What other variables mechanically depend on it?

This mapping often changes the question. Instead of asking, "Which variable has the largest coefficient?" I ask, "Which measurable step is both actionable and close enough to the mechanism that evidence can support a decision?"

### Detrend, hold out, and try to refute the story

A trustworthy analysis should contain deliberate attempts to disprove its own headline.

The first test I reach for is detrending. If two variables rise over time, their raw correlation may be high even if their week-to-week movements are unrelated. Removing shared temporal movement gives a more honest view of whether the relationship persists.

Next comes heterogeneity. I look across segments, surfaces, cohorts, or other meaningful units. An average effect can hide opposite relationships. A result that exists only in one segment may be valuable, but it needs a narrower claim.

Then I use holdout periods or split samples. Strong in-sample performance is not enough. If a relationship disappears when evaluated on later data, it is not ready to become a reliable operating lever.

I also inspect sign stability. If a coefficient changes direction when the specification changes modestly, that is a signal to reduce confidence. It may indicate collinearity, sparse data, omitted variables, or a relationship that is not structurally stable.

The practical sequence is:

1. Establish the simple relationship.
2. Check the time structure.
3. Remove or account for shared trends.
4. Test relevant segments.
5. Evaluate on held-out data.
6. Compare alternative specifications.
7. Attempt to explain the result through the operational process.
8. Downgrade the claim when the evidence does not survive.

This process often produces a less dramatic story. It also produces a story that is more likely to remain useful after the next data refresh.

### Use evidence tiers

I prefer evidence tiers to a binary distinction between "proven" and "not proven."

A practical tiering system might look like this:

- **Tier A, robust candidate:** Stable direction, plausible mechanism, survives detrending and holdout checks.
- **Tier B, conditional signal:** Real relationship, but dependent on a segment, specification, or limited evidence.
- **Tier C, diagnostic only:** Useful for understanding the system, but not suitable as an intervention lever.
- **Rejected:** Contradicted by validation or explained by a mechanical artifact.

The exact labels matter less than the habit of making confidence visible.

Evidence tiers also improve communication. A decision-maker can act differently on a Tier A candidate than on a Tier B hypothesis. A Tier C finding can still guide investigation without being promoted into a roadmap commitment.

Most importantly, this structure prevents the strongest-sounding result from winning simply because it is easiest to summarize.

### Make lineage and review part of the product

A metric or model becomes more trustworthy when people can inspect how it was made.

For each important output, I want a compact lineage record:

- Definition
- Source tables or events
- Population and grain
- Transformation logic
- Known exclusions
- Validation tests
- Confidence tier
- Owner
- Last reviewed date

This does not require a giant governance system. A small, consistent template is often enough.

The same principle applies to AI-enabled workflows. A reusable prompt, skill, agent, or analysis template is an executable artifact. It should have an owner, a version, a purpose, and a way to review changes.

One useful design is to connect publication and attribution to the same approval step. The person responsible for an artifact reviews the exact version that will be shared. This is stronger than asking for informal approval of a description, because the reviewer can see the actual diff and the actual behavior.

Automation can discover candidate artifacts, check their structure, and prepare a review. It should not silently publish names, content, or access paths without explicit approval.

This is a broader product lesson: automate the parts that reduce clerical work, but preserve human gates at points where consent, interpretation, or accountability changes.

### Lessons for AI-enabled workflows

AI makes it easier to produce analyses, summaries, code, and reusable workflows. It also makes it easier to produce them at a scale where small errors become systemic.

I have found several safeguards especially useful.

First, define a measurable exit condition. "The agent produced an answer" is not the same as "the task is complete." A workflow should state what evidence must exist before it can finish.

Second, use a grading rubric before generation. If quality criteria are written only after seeing the output, evaluation tends to become subjective and forgiving.

Third, preserve concise decision traces. A decision trace should record the question, key evidence, uncertainty, and next action. It does not need to expose private reasoning or every intermediate step. Its purpose is to make the operational conclusion inspectable.

Fourth, retire unused workflows. Reusable artifacts accumulate quickly. If nobody owns or uses one, it can become stale, misleading, or unnecessarily expensive to maintain.

Finally, treat AI-native advocacy as environment design. The important question is not simply whether a team uses AI. It is whether the team has workflows that help people evaluate, challenge, reuse, and improve AI-assisted work.

### A practical checklist

Before turning an analysis into a decision, I ask:

#### Decision

- What decision will this change?
- What action is actually available?
- What would success and failure look like?

#### Data

- What does one row represent?
- Are the schema, grain, timestamps, joins, and filters verified?
- Is the population aligned with the decision?

#### Evidence

- Is the result descriptive, associated, mechanistic, or causal?
- Does it survive detrending?
- Does it hold across relevant segments?
- Does it generalize to held-out data?
- Is the direction stable?

#### Mechanism

- Where does the variable sit in the operational chain?
- Is it actionable?
- Is it a cause, consequence, or proxy?
- Could a mechanical relationship explain the result?

#### Workflow

- Is the output reproducible?
- Is its lineage documented?
- Is the confidence tier visible?
- Who reviewed the artifact?
- What should happen if the evidence weakens?

#### Communication

- Are facts separated from hypotheses?
- Are limitations stated next to the recommendation?
- Would a reasonable reader know what is still uncertain?

The strongest decision systems I have worked on are not the ones with the most elaborate models. They are the ones that make it easy to ask better questions after the first answer arrives.

Model skepticism is not pessimism. It is a design principle. It keeps attractive artifacts from becoming false certainty, while preserving the useful signal inside imperfect evidence.

That is how uncertainty becomes something a team can work with.

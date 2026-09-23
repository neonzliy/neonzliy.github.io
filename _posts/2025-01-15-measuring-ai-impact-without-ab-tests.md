---
layout: post
description: "How Leon Zhao measured AI feature impact using exact matching and difference-in-differences when randomized A/B tests were unavailable."
title: Measuring AI Impact When You Can't A/B Test
subtitle: Using quasi-experimental methods to evaluate feature value at scale
---

If you ship fast enough, you'll eventually outrun your ability to measure. That's exactly what happened with Slack AI.

Over the course of a year, we launched AI features at roughly one per month. The goal was clear: help users navigate information overload. But because Slack users are clustered in teams and workspaces, and because the launch pace was aggressive, many features shipped without clean A/B tests. By the time leadership asked "is AI actually working?", most features were already rolled out to the majority of users.

No holdout group. No randomization. Just a question that needed an answer.

### The problem with observational data

The naive approach would be to compare AI users to non-users. But that comparison is meaningless without accounting for selection bias. Users who adopt AI features are fundamentally different: they tend to be more active, more overloaded, and on larger enterprise plans. Comparing them directly would be like concluding that umbrellas cause rain.

We needed causal inference from non-random data.

### Designing the study

I designed a quasi-experimental study using matching methods. The idea is straightforward: if we can find non-AI users who look statistically identical to AI users on every observable dimension, then the remaining difference in outcomes can be attributed to AI usage.

First, I built a feature set to characterize users and overcome data sparsity:

- Channel count and message volume
- Mention frequency and thread participation
- Team characteristics (size, plan type)
- Historical engagement patterns (pre-AI baseline)

I took a 1% stratified sample for computational speed, then built a logistic regression model to understand who actually adopts AI. This served double duty: it revealed adoption drivers and generated propensity scores for matching.

### Why exact matching beat propensity scores

My initial approach was propensity score matching, but I found that covariate balance was poor on key dimensions like plan type and team size. These variables have outsized influence on both adoption and outcomes, and collapsing them into a single score lost too much information.

I switched to exact matching on categorical variables (plan type, team size bucket) combined with k-nearest neighbors on continuous features. This gave substantially better covariate balance. I matched 2,000 AI users to similar non-users, then tracked engagement four weeks before and after first AI use.

The critical validation step: checking parallel pre-trends. If matched users show similar engagement trajectories before AI adoption, we can be more confident that post-adoption divergence reflects a real treatment effect rather than pre-existing differences.

With parallel trends confirmed, I estimated sustained effects using difference-in-differences.

### What we found

Three findings shaped the product roadmap:

**AI is adopted by the overloaded.** The users who gravitate toward AI features are the ones drowning in notifications, navigating dozens of channels, and fielding constant mentions. This wasn't surprising, but quantifying it mattered.

**Enterprise users benefit the most.** Users with 50+ channels and high mention volume see the largest engagement effects. Enterprise users make up only 16% of active users but account for over 70% of revenue. The AI investment was disproportionately serving the highest-value segment.

**Consolidation is the opportunity.** Only 25% of opted-in users had tried any Slack AI feature. Of those who adopted, 68% used only one feature, and 90% used at most two per month. The story was consistent across all AI features. Users weren't discovering the breadth of what was available.

This last finding was the most actionable. Rather than building more point features, we advocated for moving toward a single conversational entry point where users could access all AI capabilities. That became Slackbot.

### The outcome

Slackbot launched in beta as a context-aware conversational AI interface. After two months: 86% weekly retention, 14 average messages per user, and a 74% positive feedback rate.

For context, industry benchmarks for enterprise SaaS AI products show 50-70% weekly retention and 60-75% positive feedback. We were above the ceiling on retention and competitive on satisfaction, and the product was still in beta.

None of this would have been possible without the quasi-experimental work. When you can't randomize, you match. When you can't match perfectly, you validate your assumptions. The methods aren't as clean as a randomized experiment, but they gave leadership the confidence to make a major product bet.

Sometimes the most impactful analysis isn't a model or a dashboard. It's the study that answers "should we keep going?" when no one else can.

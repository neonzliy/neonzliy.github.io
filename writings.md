---
layout: editorial
title: "Writing on AI Evaluation, Workflows & Data Science"
description: "Practical articles by Leon Zhao on AI product evaluation, LLM quality, AI-native workflow design, experimentation, and causal inference."
permalink: /writings/
---

Practical notes from building and evaluating AI products. Explore quality, cost, latency, workflow adoption, and the evidence behind product decisions.

{% for post in site.posts %}
  <div class="post-preview">
    <h2 class="post-title">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h2>
    <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
    {% if post.subtitle %}
      <p class="post-excerpt"><em>{{ post.subtitle }}</em></p>
    {% else %}
      <p class="post-excerpt">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
    {% endif %}
    <p><a href="{{ post.url | relative_url }}">Read More →</a></p>
    <hr>
  </div>
{% endfor %}

---
layout: page
title: Writings
permalink: /writings/
css: '/assets/css/home.css'
---

{% for post in site.posts %}
  <div class="post-preview">
    <h2 class="post-title">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h2>
    <p class="post-excerpt">{{ post.excerpt }}</p>
    <p><a href="{{ post.url | relative_url }}">Read More →</a></p>
    <hr>
  </div>
{% endfor %}

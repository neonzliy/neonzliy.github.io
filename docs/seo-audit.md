# SEO and AI search audit — 2026-09-22

Site: https://leonz.site/  
Repository: https://github.com/neonzliy/neonzliy.github.io

## Observed live baseline

- GitHub Pages address redirects permanently (301) to the HTTPS custom domain; the homepage returns 200.
- Canonical links, server-rendered content, sitemap.xml, and a robots.txt sitemap declaration already exist.
- The shared head ignores `page.description`. The writing archive has no deliberate summary.
- No JSON-LD describes the author, website, profile, or articles.
- Article templates do not visibly identify their author or exact publication date. Open Graph article property names use an incorrect `og:article:` prefix.
- The sitemap includes the alternative `/v7/` design and the empty `/tags/` page.
- Homepage writing links are a fixed selection of older articles.
- RSS has no channel description and is not advertised in page heads.

## Implemented locally

- Use explicit page descriptions, social descriptions, subtitles, then a safe excerpt/site fallback. Add unique, topic-specific summaries to seven posts and the writing archive.
- Clarify the homepage title around Leon Zhao, AI product evaluation, and data science while preserving its visual identity.
- Add a linked JSON-LD graph: Person, WebSite, WebPage, ProfilePage, CollectionPage, and BlogPosting where appropriate. Author identity links use the existing GitHub and LinkedIn profiles. Dates come from existing source front matter; no invented credentials, claims, reviews, or modification dates.
- Display linked author bylines and publication dates. Correct article Open Graph properties. Move the editorial stylesheet into the HTML head.
- Mark the preview, empty tag index, and error page `noindex, follow`; remove them from the generated sitemap. Leave crawling allowed so crawlers can see noindex.
- Link the three newest posts from the homepage. Retain the complete writing archive.
- Enable RSS discovery and a feed description. Exclude development scripts and type stubs from published artifacts.
- Add `scripts/check-seo.py` to validate generated output after a Jekyll build.

## AI search approach

Google says its existing SEO fundamentals apply to AI Overviews and AI Mode; no special AI text file or schema is required. The work above improves crawlability, authorship, machine-readable relationships, and discoverability without promises of ranking or citation.

Reference: https://developers.google.com/search/docs/appearance/ai-features

Existing robots.txt does not disallow crawlers. This audit does not change AI-training crawler permissions or add speculative AI directives. `llms.txt` is not required for Google's AI search eligibility and was not added.

## Verification

- Production Jekyll 3.10 build with strict front matter passed locally (Ruby 2.6); deployment workflow uses Ruby 3.3 and was not run.
- Generated output checks passed for all 10 indexable pages: unique descriptions, canonical and social URLs, JSON-LD parsing and author relationships, article publication metadata, internal link targets, sitemap exclusions, RSS XML, and robots.txt sitemap reference.
- Headless Chrome checks at 1440 px and 390 px with JavaScript disabled passed for home, profile, archive, and an article: visible content, stylesheet loading, byline output, and no horizontal overflow. The mobile article screenshot was visually inspected.
- With JavaScript enabled, the homepage reached `is-webgl is-ready`, linked the three latest posts, and successfully navigated to the writing archive without page errors.
- Independent verifier verdict: CONFIRMED. All 13 generated HTML files were scanned for local link targets with zero broken links; no introduced findings were reported.
- Live observations are HTTP/source inspections, not proof of current search indexing or ranking.

## After publication

1. Verify ownership of `leonz.site` in Google Search Console and Bing Webmaster Tools if not already done. Submit `https://leonz.site/sitemap.xml` and inspect the homepage, `/me/`, and key articles. This requires access to the owner's accounts.
2. Review actual impressions, queries, clicks, indexing reports, and referrals over time. Rankings, AI citations, and crawl schedules cannot be guaranteed.
3. Run PageSpeed Insights on the deployed homepage and an article. The homepage loads a WebGL scene and roughly 740 KB of Three.js source plus 204 KB of scene data before compression. That is a performance investigation target, not a measured Core Web Vitals failure; no performance score is claimed here.
4. Strengthen article evidence with attributable public references, methodology limitations, and substantive case studies where disclosure is appropriate. Review existing numerical claims and publication dates for accuracy; the audit preserves them and stable URLs.
5. Add genuine `last_modified_at` values when articles are substantially revised. Do not automatically label every build as a content update.

The owner approved publication on 2026-09-22. Deployment uses the existing GitHub Pages workflow on pushes to `master`.

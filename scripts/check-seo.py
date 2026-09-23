"""Validate generated public pages after `bundle exec jekyll build`."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / '_site'
ORIGIN = 'https://leonz.site'

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.meta, self.links, self.scripts = {}, [], []
        self.h1 = 0
        self.json_text = None
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('name', attrs.get('property'))] = attrs.get('content')
        if tag in ('a', 'link'):
            self.links.append(attrs)
        if tag == 'h1':
            self.h1 += 1
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.json_text = ''
    def handle_data(self, data):
        if self.json_text is not None:
            self.json_text += data
    def handle_endtag(self, tag):
        if tag == 'script' and self.json_text is not None:
            self.scripts.append(json.loads(self.json_text))
            self.json_text = None

def file_for(url):
    path = unquote(urlsplit(url).path).lstrip('/')
    result = ROOT / path
    return result / 'index.html' if result.is_dir() else result

urls = [el.text for el in ET.parse(ROOT / 'sitemap.xml').iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
assert len(urls) == 10, urls  # home, profile, writing archive, seven articles
assert len(set(urls)) == len(urls)
descriptions = set()
for url in urls:
    assert url.startswith(ORIGIN + '/'), url
    page = Page(file_for(url).read_text())
    assert page.h1 == 1, url
    assert 'noindex' not in page.meta['robots'], url
    description = page.meta['description']
    assert description and description not in descriptions, (url, description)
    assert '{%' not in description and '{{' not in description, url
    descriptions.add(description)
    assert page.meta['og:description'] == description, url
    assert page.meta['twitter:description'] == description, url
    canonicals = [a['href'] for a in page.links if a.get('rel') == 'canonical']
    assert canonicals == [url], (url, canonicals)
    assert page.meta['og:url'] == url, url
    assert len(page.scripts) == 1, url
    graph = {item['@type']: item for item in page.scripts[0]['@graph']}
    assert graph['Person']['name'] == 'Leon Zhao', url
    assert graph['Person']['url'] == ORIGIN + '/me/', url
    if urlsplit(url).path.startswith('/20'):
        article = graph['BlogPosting']
        assert article['datePublished'] == page.meta['article:published_time'], url
        assert any(a.get('rel') == 'author' and a.get('href') == '/me/' for a in page.links), url
    if url == ORIGIN + '/me/':
        assert graph['ProfilePage']['mainEntity']['@id'] == graph['Person']['@id']
    for a in page.links:
        href = a.get('href', '')
        if href.startswith('/') and not href.startswith('//'):
            assert file_for(href).exists(), (url, href)
for path in ('v7/', 'tags/', '404.html'):
    url = ORIGIN + '/' + path
    assert url not in urls, url
    assert 'noindex' in Page(file_for(url).read_text()).meta['robots'], url
assert 'Sitemap: ' + ORIGIN + '/sitemap.xml' in (ROOT / 'robots.txt').read_text()
ET.parse(ROOT / 'feed.xml')
assert not (ROOT / 'scripts').exists()
assert not (ROOT / 'typings').exists()
print(f'PASS: {len(urls)} indexable pages; metadata, JSON-LD, canonical URLs, local links, exclusions, sitemap, robots.txt and RSS.')

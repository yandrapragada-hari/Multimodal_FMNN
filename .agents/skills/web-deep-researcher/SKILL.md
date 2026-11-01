---
name: web-deep-researcher
description: >-
  Browses the internet, queries web search APIs, renders dynamic JavaScript web pages using Playwright,
  and extracts clean markdown content for academic research and documentation.
---

# Web Deep Researcher Skill

## Capabilities
- Automates web browsing using Playwright headless Chromium or HTTPX.
- Bypasses client-side rendering hurdles to scrape full DOM articles, papers, and docs.
- Cleans and converts raw HTML into formatted Markdown, eliminating navigation bars, sidebars, and ads.

## Available Helpers
- `scripts/fetch_page.py`: Fetches any URL (either via fast HTTPX or headless browser) and returns sanitized text/markdown.

## Usage Guide
```bash
python ~/.gemini/config/skills/web-deep-researcher/scripts/fetch_page.py --url "https://arxiv.org/abs/2103.00020" --output "page.md"
```

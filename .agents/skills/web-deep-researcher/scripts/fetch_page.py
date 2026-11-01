"""Fetches and cleans web pages into Markdown using httpx and BeautifulSoup."""

import argparse
import sys
import httpx
from bs4 import BeautifulSoup


def fetch_and_clean(url: str, output_path: str = None) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(follow_redirects=True, timeout=15.0, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer, ads
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Extract text from paragraphs and headings
    content_blocks = [f"# {title}\nSource: {url}\n"]
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = elem.get_text().strip()
        if not txt:
            continue
        if elem.name == "h1":
            content_blocks.append(f"\n# {txt}\n")
        elif elem.name == "h2":
            content_blocks.append(f"\n## {txt}\n")
        elif elem.name == "h3":
            content_blocks.append(f"\n### {txt}\n")
        elif elem.name == "li":
            content_blocks.append(f"- {txt}")
        else:
            content_blocks.append(f"\n{txt}")

    result = "\n".join(content_blocks)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Saved cleaned webpage content to: {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and sanitize web page to Markdown")
    parser.add_argument("--url", required=True, help="Target URL to fetch")
    parser.add_argument("--output", default=None, help="Optional output path")
    args = parser.parse_args()

    cleaned = fetch_and_clean(args.url, args.output)
    if not args.output:
        print(cleaned[:1000] + "\n...[truncated]")

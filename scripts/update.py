#!/usr/bin/env python3
"""AI Pulse — data update script.

Fetches RSS feeds, deduplicates recent items, asks Claude to score
AI company momentum, and writes data/feed.json + data/history.json.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
import xml.etree.ElementTree as ET

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic feedparser")

# ── Config ───────────────────────────────────────────────────────────────────

FEEDS = [
    "https://openai.com/news/rss.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://hnrss.org/frontpage?q=AI+LLM+Claude+GPT+DeepSeek&count=20",
]

LOOKBACK_HOURS = 12
MAX_ITEMS = 40
SIMILARITY_THRESHOLD = 0.6

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
FEED_FILE = DATA_DIR / "feed.json"
HISTORY_FILE = DATA_DIR / "history.json"

SYSTEM_PROMPT = """You are an AI news analyst. Given a list of recent AI news headlines \
and summaries, return a JSON object with this exact structure:
{
  "updated_at": "<ISO datetime>",
  "scores": [
    { "name": "OpenAI",     "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Anthropic",  "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "DeepSeek",   "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Perplexity", "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" }
  ],
  "signals": [
    { "company": "...", "text": "...one line max...", "tag": "..." },
    ... 3-5 items
  ],
  "winner": { "name": "...", "text": "...one sentence, slightly ironic..." },
  "loser":  { "name": "...", "text": "...one sentence, slightly ironic..." }
}
Score logic: higher = more positive news momentum today.
Delta = change from neutral baseline (50) if no prior data is available.
Return only valid JSON, no markdown."""


# ── RSS helpers ───────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 15) -> bytes | None:
    headers = {"User-Agent": "AI-Pulse/1.0 (https://github.com/melnikvl/aifund)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (URLError, Exception) as exc:
        print(f"  warn: failed to fetch {url}: {exc}", file=sys.stderr)
        return None


def parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    # Try common RSS date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_text(element) -> str:
    """Return text from an XML element, stripping HTML tags."""
    if element is None:
        return ""
    text = element.text or ""
    # Strip simple HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def parse_feed(raw: bytes) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items

    ns = {}
    # Detect Atom vs RSS
    tag = root.tag
    if "atom" in tag or "feed" in tag.lower():
        # Atom feed
        atom_ns = "http://www.w3.org/2005/Atom"
        for entry in root.findall(f"{{{atom_ns}}}entry"):
            title = extract_text(entry.find(f"{{{atom_ns}}}title"))
            summary = extract_text(entry.find(f"{{{atom_ns}}}summary")) or \
                      extract_text(entry.find(f"{{{atom_ns}}}content"))
            pub = entry.find(f"{{{atom_ns}}}published") or entry.find(f"{{{atom_ns}}}updated")
            pub_text = pub.text if pub is not None else None
            items.append({"title": title, "summary": summary, "published": pub_text})
    else:
        # RSS 2.0
        for item in root.iter("item"):
            title = extract_text(item.find("title"))
            desc  = extract_text(item.find("description")) or \
                    extract_text(item.find("summary"))
            pub   = item.find("pubDate") or item.find("published")
            pub_text = pub.text if pub is not None else None
            items.append({"title": title, "summary": desc, "published": pub_text})

    return items


def fetch_recent_items(lookback_hours: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    all_items: list[dict] = []

    for url in FEEDS:
        print(f"  fetching {url}", file=sys.stderr)
        raw = fetch_url(url)
        if raw is None:
            continue
        parsed = parse_feed(raw)
        for item in parsed:
            pub = parse_date(item.get("published"))
            # Include items without a parseable date — they might be recent
            if pub is None or pub >= cutoff:
                all_items.append(item)

    return all_items


# ── Deduplication ─────────────────────────────────────────────────────────────

def title_tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(items: list[dict]) -> list[dict]:
    seen: list[set] = []
    unique: list[dict] = []
    for item in items:
        tokens = title_tokens(item.get("title", ""))
        if any(jaccard(tokens, s) >= SIMILARITY_THRESHOLD for s in seen):
            continue
        seen.append(tokens)
        unique.append(item)
    return unique


# ── Claude analysis ───────────────────────────────────────────────────────────

def build_news_block(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items[:MAX_ITEMS], 1):
        title   = item.get("title",   "").strip()
        summary = item.get("summary", "").strip()
        if title:
            line = f"{i}. {title}"
            if summary:
                line += f" — {summary[:200]}"
            lines.append(line)
    return "\n".join(lines)


def analyse_with_claude(news_block: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    user_msg = (
        f"Current UTC time: {now_iso}\n\n"
        f"Recent AI news items:\n{news_block}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_feed(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEED_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    generate_static_html(data)
    print(f"  wrote {FEED_FILE}", file=sys.stderr)


def save_history(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            history = []

    # Keep a compact snapshot (timestamps + scores only)
    snapshot = {
        "updated_at": data.get("updated_at"),
        "scores": [
            {"name": s["name"], "score": s["score"], "delta": s["delta"], "badge": s["badge"]}
            for s in data.get("scores", [])
        ],
    }
    history.append(snapshot)

    # Retain last 60 snapshots (~30 days at twice-daily cadence)
    history = history[-60:]
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(f"  wrote {HISTORY_FILE}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    print("1/4  fetching RSS feeds…", file=sys.stderr)
    items = fetch_recent_items(LOOKBACK_HOURS)
    print(f"     collected {len(items)} raw items", file=sys.stderr)

    print("2/4  deduplicating…", file=sys.stderr)
    items = deduplicate(items)
    print(f"     {len(items)} unique items", file=sys.stderr)

    if not items:
        sys.exit("No items found — check feed URLs or lookback window.")

    print("3/4  analysing with Claude…", file=sys.stderr)
    news_block = build_news_block(items)
    data = analyse_with_claude(news_block)

    print("4/4  saving output…", file=sys.stderr)
    save_feed(data)
    save_history(data)

    print("done.", file=sys.stderr)



def generate_static_html(data: dict):
    signals_html = ""
    for s in data.get("signals", []):
        signals_html += f'<li><strong>{s.get("company","")}</strong>: {s.get("text","")}</li>\n'

    scores_html = ""
    for info in data.get("scores", []):
        name = info.get("name", info.get("company", ""))
        scores_html += f'<li>{name}: {info.get("score","?")} ({info.get("badge","")})</li>\n'

    winner = data.get("winner", {})
    loser = data.get("loser", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Pulse — Daily AI Industry Momentum</title>
<meta name="description" content="Daily momentum scores for OpenAI, Anthropic, DeepSeek and Perplexity based on latest AI news.">
<link rel="canonical" href="https://app.ai-groundtruth.com/">
</head>
<body>
<h1>AI Pulse — AI Industry Momentum</h1>
<p>Updated: {data.get("updated","")}</p>
<h2>Scores</h2>
<ul>{scores_html}</ul>
<h2>Top Signals</h2>
<ul>{signals_html}</ul>
<h2>Winner of the day: {winner.get("name","")}</h2>
<p>{winner.get("reason","")}</p>
<h2>Loser of the day: {loser.get("name","")}</h2>
<p>{loser.get("reason","")}</p>
<p><a href="/">View live dashboard</a></p>
</body>
</html>"""

    with open("data/seo.html", "w") as f:
        f.write(html)
    print("  static seo.html generated")

if __name__ == "__main__":
    main()


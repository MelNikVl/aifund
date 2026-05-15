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
    # Core AI labs
    "https://openai.com/news/rss.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
    "https://huggingface.co/blog/feed.xml",
    # Google
    "https://blog.google/technology/ai/rss/",
    # Meta / xAI / Mistral via community RSS mirrors
    "https://raw.githubusercontent.com/0xSMW/rss-feeds/main/feeds/feed_meta_ai.xml",
    "https://raw.githubusercontent.com/0xSMW/rss-feeds/main/feeds/feed_xai_news.xml",
    "https://raw.githubusercontent.com/0xSMW/rss-feeds/main/feeds/feed_mistral_news.xml",
    # Decentralized AI
    "https://blog.bittensor.com/feed",                          # Bittensor official
    "https://what-is-gonka.hashnode.dev/rss.xml",               # Gonka blog
    "https://www.reddit.com/r/bittensor_.rss",                  # Bittensor community
    "https://www.reddit.com/r/GonkaAI/.rss",                    # Gonka community
    "https://hnrss.org/frontpage?q=Bittensor+TAO+Gonka+Cocoon+decentralized+AI&count=15",
    # Tech media
    "https://www.theverge.com/rss/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://wired.com/feed/rss",
    # Reddit AI communities
    "https://www.reddit.com/r/MachineLearning/.rss",
    "https://www.reddit.com/r/artificial/.rss",
    "https://www.reddit.com/r/OpenAI/.rss",
    "https://www.reddit.com/r/LocalLLaMA/.rss",
    # Hacker News
    "https://hnrss.org/frontpage?q=AI+LLM+Claude+GPT+DeepSeek+Gemini+Llama+Mistral&count=30",
]

LOOKBACK_HOURS = 48
MAX_ITEMS = 60
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
    { "name": "Google",     "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Meta",       "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "DeepSeek",   "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Mistral",    "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "xAI",        "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Perplexity", "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Bittensor",  "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" },
    { "name": "Gonka",      "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" },
    { "name": "Cocoon",     "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" }
  ],
  "signals": [
    { "company": "...", "text": "...one line max...", "tag": "...", "date": "YYYY-MM-DD" },
    ... 5-8 items
  ],
  "winner": { "name": "...", "text": "...one sentence, slightly ironic..." },
  "loser":  { "name": "...", "text": "...one sentence, slightly ironic..." }
}
Score logic: higher = more positive news momentum today.
Delta = change from neutral baseline (50) if no prior data is available.
For Bittensor/Gonka/Cocoon: score based on network activity, token news, new subnets, partnerships, technical milestones.
For signals, use the actual publication date of the news item if available, otherwise use today's date.
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


def format_date_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d")


def extract_text(element) -> str:
    if element is None:
        return ""
    text = element.text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def parse_feed(raw: bytes) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items

    tag = root.tag
    if "atom" in tag or "feed" in tag.lower():
        atom_ns = "http://www.w3.org/2005/Atom"
        for entry in root.findall(f"{{{atom_ns}}}entry"):
            title = extract_text(entry.find(f"{{{atom_ns}}}title"))
            summary = extract_text(entry.find(f"{{{atom_ns}}}summary")) or \
                      extract_text(entry.find(f"{{{atom_ns}}}content"))
            pub = entry.find(f"{{{atom_ns}}}published")
            if pub is None:
                pub = entry.find(f"{{{atom_ns}}}updated")
            pub_text = pub.text if pub is not None else None
            items.append({"title": title, "summary": summary, "published": pub_text})
    else:
        # RSS 2.0 — fixed DeprecationWarning
        for item in root.iter("item"):
            title = extract_text(item.find("title"))
            desc  = extract_text(item.find("description")) or \
                    extract_text(item.find("summary"))
            pub = item.find("pubDate")
            if pub is None:
                pub = item.find("published")
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
            # Make timezone-aware if naive
            if pub is not None and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub is None or pub >= cutoff:
                # Attach parsed date as ISO string for Claude
                item["date"] = format_date_iso(pub)
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
        date    = item.get("date", "")
        if title:
            line = f"{i}. [{date}] {title}" if date else f"{i}. {title}"
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
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ── Prediction Markets ───────────────────────────────────────────────────────

BETS_FILE = DATA_DIR / "bets.json"

BETS_GENERATE_PROMPT = """You are an AI industry analyst. Given recent AI news, generate 3 fresh prediction market questions.
Rules:
- Questions must be YES/NO answerable
- Resolvable within 7-14 days based on observable facts
- About specific AI companies, product launches, regulatory events, or market moves
- Must be directly connected to the news provided
- No duplicates of existing open questions listed below

Existing open questions (do not duplicate):
{existing}

Return ONLY valid JSON, no markdown:
[
  {{
    "id": "q_<6 random hex chars>",
    "text": "Will X do Y by <specific date>?",
    "created_at": "<ISO datetime now>",
    "resolves_at": "<ISO datetime 7-14 days from now>",
    "status": "open",
    "yes_votes": 0,
    "no_votes": 0,
    "resolution": null,
    "resolution_text": null
  }},
  ...
]"""

BETS_RESOLVE_PROMPT = """You are an AI industry analyst. Given recent AI news and a list of open prediction questions,
resolve any questions whose resolution date has passed OR whose outcome is clearly determinable from the news.

News summary:
{news}

Questions to evaluate:
{questions}

For each question return its id and one of:
- resolution: "yes" or "no" (if clearly resolvable)
- resolution: null (if still uncertain)
- resolution_text: one sentence explaining the resolution

Return ONLY valid JSON, no markdown:
[
  {{"id": "q_xxx", "resolution": "yes"|"no"|null, "resolution_text": "..."}},
  ...
]"""


def load_bets() -> dict:
    if BETS_FILE.exists():
        try:
            return json.loads(BETS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"questions": []}


def save_bets(bets: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BETS_FILE.write_text(json.dumps(bets, indent=2, ensure_ascii=False))
    print(f"  wrote {BETS_FILE}", file=sys.stderr)


def generate_bets(news_block: str, existing_questions: list) -> list:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    existing_texts = "\n".join(f"- {q['text']}" for q in existing_questions) or "none"
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prompt = BETS_GENERATE_PROMPT.format(existing=existing_texts)
    user_msg = f"Current UTC time: {now_iso}\n\nRecent news:\n{news_block[:3000]}"

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def resolve_bets(news_block: str, open_questions: list) -> list:
    if not open_questions:
        return []
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    questions_text = json.dumps(
        [{"id": q["id"], "text": q["text"], "resolves_at": q["resolves_at"]} for q in open_questions],
        indent=2
    )
    prompt = BETS_RESOLVE_PROMPT.format(
        news=news_block[:3000],
        questions=questions_text
    )

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="You are an AI industry analyst. Return only valid JSON arrays.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def update_bets(news_block: str) -> None:
    bets = load_bets()
    now = datetime.now(timezone.utc)

    # 1. Resolve open questions
    open_qs = [q for q in bets["questions"] if q["status"] == "open"]
    if open_qs:
        print("  resolving open bets…", file=sys.stderr)
        try:
            resolutions = resolve_bets(news_block, open_qs)
            for r in resolutions:
                for q in bets["questions"]:
                    if q["id"] == r["id"] and r.get("resolution") in ("yes", "no"):
                        q["resolution"] = r["resolution"]
                        q["resolution_text"] = r.get("resolution_text", "")
                        q["status"] = "resolved"
        except Exception as e:
            print(f"  warn: resolution failed: {e}", file=sys.stderr)

    # 2. Expire overdue unresolved questions
    for q in bets["questions"]:
        if q["status"] == "open":
            try:
                resolves_at = datetime.fromisoformat(q["resolves_at"].replace("Z", "+00:00"))
                if resolves_at < now - timedelta(days=2):
                    q["status"] = "expired"
            except Exception:
                pass

    # 3. Generate new questions (only if fewer than 5 open)
    open_count = sum(1 for q in bets["questions"] if q["status"] == "open")
    if open_count < 5:
        print("  generating new bets…", file=sys.stderr)
        try:
            new_qs = generate_bets(news_block, [q for q in bets["questions"] if q["status"] == "open"])
            bets["questions"] = bets["questions"] + new_qs
            # Keep last 50 questions total
            bets["questions"] = bets["questions"][-50:]
        except Exception as e:
            print(f"  warn: bet generation failed: {e}", file=sys.stderr)

    save_bets(bets)


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

    snapshot = {
        "updated_at": data.get("updated_at"),
        "scores": [
            {"name": s["name"], "score": s["score"], "delta": s["delta"], "badge": s["badge"]}
            for s in data.get("scores", [])
        ],
    }
    history.append(snapshot)
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

    print("5/5  updating prediction markets…", file=sys.stderr)
    update_bets(news_block)

    print("done.", file=sys.stderr)


def generate_static_html(data: dict):
    signals_html = ""
    for s in data.get("signals", []):
        date_str = f" <span style='color:#888;font-size:0.85em'>({s.get('date', '')})</span>" if s.get("date") else ""
        signals_html += f'<li><strong>{s.get("company","")}</strong>{date_str}: {s.get("text","")}</li>\n'

    scores_html = ""
    for info in data.get("scores", []):
        name = info.get("name", info.get("company", ""))
        delta = info.get("delta", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        scores_html += f'<li>{name}: {info.get("score","?")} ({delta_str}) — {info.get("badge","")}</li>\n'

    winner = data.get("winner", {})
    loser = data.get("loser", {})
    updated = data.get("updated_at", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Pulse — Daily AI Industry Momentum</title>
<meta name="description" content="Daily momentum scores for OpenAI, Anthropic, Google, Meta, DeepSeek, Mistral, xAI and Perplexity based on latest AI news.">
<link rel="canonical" href="https://app.ai-groundtruth.com/">
</head>
<body>
<h1>AI Pulse — AI Industry Momentum</h1>
<p>Updated: {updated}</p>
<h2>Scores</h2>
<ul>{scores_html}</ul>
<h2>Top Signals</h2>
<ul>{signals_html}</ul>
<h2>Winner of the day: {winner.get("name","")}</h2>
<p>{winner.get("text","")}</p>
<h2>Loser of the day: {loser.get("name","")}</h2>
<p>{loser.get("text","")}</p>
<p><a href="/">View live dashboard</a></p>
</body>
</html>"""

    seo_path = DATA_DIR / "seo.html"
    seo_path.write_text(html)
    print("  static seo.html generated", file=sys.stderr)


if __name__ == "__main__":
    main()
